import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from src import utils
from typing import Callable, Union, Optional,Tuple
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import zeros_
from torch_scatter import scatter
from src.gvp_model import GVPNetwork

class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented.")

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)






class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf,edges_in_d=0,
                 activation='silu',attention=False, 
                 normalization_factor=100, aggregation_method='sum', normalization=None):
                   
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.edge_mlp = nn.Sequential(
            DenseLayer(input_edge + edges_in_d, hidden_nf,activation=activation),
            nn.BatchNorm1d(hidden_nf),
            DenseLayer(hidden_nf, hidden_nf,activation=activation))

        if normalization is None:
            self.node_mlp = nn.Sequential(
                DenseLayer(hidden_nf + input_nf, hidden_nf,activation=activation),
                DenseLayer(hidden_nf, output_nf,activation=None))
            
        elif normalization == 'batch_norm':
            self.node_mlp=nn.Sequential(
                DenseLayer(hidden_nf + input_nf, hidden_nf,activation=activation),
                nn.BatchNorm1d(hidden_nf),
                DenseLayer(hidden_nf, output_nf,activation=None),
                nn.BatchNorm1d(output_nf))
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        out = torch.cat([h[col], h[row], edge_attr], dim=1)
        edge_feat = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(edge_feat)
            edge_feat = edge_feat * att_val
        agg=scatter(edge_feat, col, dim=0, reduce=self.aggregation_method)/self.normalization_factor
        
        agg=torch.cat([h, agg], dim=1)
        out = h + self.node_mlp(agg)
        return out


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, activation='silu', tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
           DenseLayer(input_edge, hidden_nf,activation=activation),
           nn.BatchNorm1d(hidden_nf),
           DenseLayer(hidden_nf, hidden_nf,activation=activation),
           nn.BatchNorm1d(hidden_nf),
           layer)
            
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method


    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, ligand_diff=None):
        row, col = edge_index
        input_tensor = torch.cat([h[col], h[row], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        agg=scatter(trans, col, dim=0, reduce=self.aggregation_method)/self.normalization_factor
        if ligand_diff is not None:
            agg = agg * ligand_diff
        coord = coord + agg
        return coord

       


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2,device='cpu',
                 activation='silu',n_layers=2, 
                 attention=True,norm_diff=True, tanh=False,
                 coords_range=15, norm_constant=1,
                 sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum',
                 normalization=None):  
                 
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              activation=activation, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              normalization=normalization,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, activation=activation, tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, x, edge_index, ligand_diff=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h= self._modules["gcl_%d" % i](h, edge_index, edge_attr)
        x = self._modules["gcl_equiv"](
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            ligand_diff=ligand_diff,
        )
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', activation='silu', n_layers=3, attention=False,
                 tanh=False, norm_constant=1,inv_sublayers=2,sin_embedding=False,normalization_factor=100, aggregation_method='sum',
                 norm_diff=True, out_node_nf=None,  coords_range=15, normalization=None
                  ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = DenseLayer(self.hidden_nf*2, self.hidden_nf,activation=activation)
        
        self.embedding_out = DenseLayer(self.hidden_nf, out_node_nf,activation=None)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               activation=activation, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               normalization=normalization,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        if torch.cuda.is_available():
            self.to(self.device)
        else:
            self.to('cpu')

    def forward(self, h, x, edge_index, ligand_diff):
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, 
                ligand_diff=ligand_diff,
                edge_attr=distances
            )
        h = self.embedding_out(h)
        return h, x


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[col] - x[row]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff




class Dynamics(nn.Module):
    def __init__(
            self, in_node_nf,n_dims,  ligand_group_node_nf, 
            hidden_nf=32, activation='silu', n_layers=2,attention=False,tanh=True,
            norm_constant=0.00001, inv_sublayers=2, sin_embedding=False,
            normalization_factor=100,aggregation_method='sum',  drop_rate=0.0,
            device='cpu',model='egnn_dynamics',normalization='batch_norm',condition_time=True
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.ligand_group_node_nf = ligand_group_node_nf+1
        self.model = model

        self.ligand_group_embedding=DenseLayer(ligand_group_node_nf+1,hidden_nf,activation=activation)
        self.h_embedding=DenseLayer(in_node_nf,hidden_nf,activation=activation)
        in_node_nf = in_node_nf + ligand_group_node_nf + condition_time
        self.h_embedding_out=DenseLayer(hidden_nf, in_node_nf,activation=None)
        
        if self.model == 'egnn_dynamics':
            self.dynamics = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf, device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                normalization=normalization,
            )
        elif self.model == 'gvp_dynamics':
            self.dynamics = GVPNetwork(
                in_dims=(hidden_nf*2, 0), # (scalar_features, vector_features)
            out_dims=(hidden_nf, 1),
            hidden_dims=(hidden_nf, hidden_nf//2),
            drop_rate=drop_rate,
            vector_gate=True,
            num_layers=n_layers,
            attention=attention,
            normalization_factor=normalization_factor,
            )
        else:
            raise NotImplementedError


    def forward(self,xh, t,  ligand_diff, ligand_group,batch_seg ):

        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)
        edge_index = radius_graph(x, r=1e+50, batch=batch_seg, loop=False,max_num_neighbors=100)
        
        # conditioning on time 
        if np.prod(t.size()) == 1:
            # t is the same for all elements in batch.
            h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
        else:
            # t is different over the batch dimension.
            h_time = t[batch_seg]  

        ligand_group_with_time=torch.cat([ligand_group,h_time],dim=-1)

        ligand_group_with_time=self.ligand_group_embedding(ligand_group_with_time) #(B*N, hidden_nf)
        h=self.h_embedding(h) #(B*N, hidden_nf)
        h=torch.cat([h,ligand_group_with_time],dim=-1)

        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(h,x,edge_index,ligand_diff)
            vel = (x_final - x) 
    
        elif self.model == 'gvp_dynamics':
            h_final, vel = self.dynamics(h,x, edge_index)
            h_final=self.h_embedding_out(h_final)
            vel=vel.squeeze()
            
        else:
            raise NotImplementedError

        # Slice off ligand_group 
        if ligand_group is not None:
            h_final = h_final[:, :-self.ligand_group_node_nf]
        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            raise utils.FoundNaNException(vel, h_final)

        return torch.cat([vel, h_final], dim=1)

    


