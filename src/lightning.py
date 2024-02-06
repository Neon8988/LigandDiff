import numpy as np
import pandas as pd
import os
import pytorch_lightning as pl
import torch
import wandb
import time
from rdkit import Chem
from src import  utils
from src import const
from src.egnn import Dynamics
from src.edm import EDM
from src.visualizer import visualize_chain
from src.SA_Score.sascorer import compute_sa_score   
from src.molecule_builder import BasicLigandMetrics, build_mol,extract_ligand,sanitycheck,write_xyz_file
from typing import Dict, List, Optional
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add


class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(self,
        data_path, train_data, val_data,
        in_node_nf, n_dims, ligand_group_node_nf,
        hidden_nf,attention,n_layers,normalization_factor,normalize_factors,
        
        drop_rate,

        activation, tanh, norm_constant,
        inv_sublayers, sin_embedding,  aggregation_method,normalization,
        
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        lr,batch_size,torch_device, model,test_epochs,
        samples_dir=None, center_of_mass='context',clip_grad=False,
    ):
        super(DDPM, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data = train_data
        self.val_data = val_data
        
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.test_epochs = test_epochs
        self.samples_dir = samples_dir
        self.center_of_mass = center_of_mass
        self.loss_type = diffusion_loss_type
        self.T=diffusion_steps
        self.clip_grad=clip_grad
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            self.gradnorm_queue.add(3000)
        
        self.ligand_metrics=BasicLigandMetrics()
        
        # save targets in each batch to compute metric overall epoch
        self.training_step_outputs = []   
        self.validation_step_outputs = []
        self.test_step_outputs = []

        dynamics = Dynamics(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            ligand_group_node_nf=ligand_group_node_nf,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            device=torch_device,
            model=model,
            normalization=normalization,
            drop_rate=drop_rate
        )

        self.edm = EDM(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset=torch.load(f'{self.data_path}/{self.train_data}.pt',map_location=self.torch_device)
            self.val_dataset=torch.load(f'{self.data_path}/{self.val_data}.pt',map_location=self.torch_device)
        elif stage == 'val':   
            self.val_dataset = torch.load(f'{self.data_path}/{self.val_data}.pt',map_location=self.torch_device)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size)

    
    def forward(self, data):
        x = data['pos']
        h = data['one_hot']
        context = data['context'].view(-1,1)
        ligand_diff = data['ligand_diff'].view(-1,1)
        ligand_group=data['ligand_group']
        batch_seg=data.batch
        batch_size=int(torch.max(batch_seg))+1
        
        #Removing COM of context from the atom coordinates
        if self.center_of_mass == 'context':
            x = utils.remove_partial_mean_with_mask(x,context,batch_seg)
        elif self.center_of_mass == 'ligand_diff':
            x = utils.remove_partial_mean_with_mask(x,ligand_diff,batch_seg)
        else:
            raise ValueError(f'Unknown center_of_mass: {self.center_of_mass}')
        
        delta_log_px, error_t, SNR_weight,loss_0_x, loss_0_h, neg_log_const_0,\
        kl_prior= self.edm.forward(
            x=x,
            h=h,
            context=context,
            ligand_diff=ligand_diff,
            batch_seg=batch_seg,
            batch_size=batch_size,
            ligand_group=ligand_group
        )
        if self.loss_type == 'l2' and self.training:
            #normalize loss_t
            normalization=(self.n_dims + self.in_node_nf)*EDM.inflate_batch_array(ligand_diff, batch_seg)
            error_t=error_t/normalization
            loss_t=error_t
            #normaliza loss_0
            loss_0_x=loss_0_x/self.n_dims*EDM.inflate_batch_array(ligand_diff, batch_seg)
            loss_0=loss_0_x+loss_0_h
        
        else:
            loss_t = self.T * 0.5 * SNR_weight * error_t
            loss_0 = loss_0_x + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        nll=loss_t + loss_0 + kl_prior

        if not (self.loss_type == 'l2' and self.training):
            nll=nll-delta_log_px
        
        metrics={'error_t':error_t.mean(0),
        'SNR_weight': SNR_weight.mean(0),
        'loss_0':loss_0.mean(0),
        'kl_prior':kl_prior.mean(0),
        'delta_log_px':delta_log_px.mean(0),
        'neg_log_const_0': neg_log_const_0.mean(0)}
        return nll, metrics
            
    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def training_step(self, data, *args):
        try:
            nll, metrics = self.forward(data)
        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e

        loss = nll.mean(0)
        metrics['loss']=loss
        self.log_metrics(metrics, 'train', batch_size=int(torch.max(data.batch))+1)
        self.training_step_outputs.append(metrics)
        return metrics

    def _shared_eval_step(self, data, prefix, *args):
        nll, metrics = self.forward(data)
        loss = nll.mean(0)
        metrics['loss'] = loss
        self.log_metrics(metrics, prefix, batch_size=torch.max(data.batch)+1,sync_dist=True)
        if prefix == 'val':
            self.validation_step_outputs.append(metrics)
        else:
            self.test_step_outputs.append(metrics)
        return metrics
    
    def validation_step(self, data, *args):
        self._shared_eval_step(data, 'val', *args)

    def test_step(self, data, *args):
        self._shared_eval_step(data, 'test', *args)

    def on_train_epoch_end(self):
        for metric in self.training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)
        print(f"current epoch on validation:{self.current_epoch} and current val_loss:{self.metrics['loss/val'][-1]}")
        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.val_dataset,animation=True)
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/val', metric_value, prog_bar=True)
                self.metrics.setdefault(f'{metric_name}/val', []).append(metric_value)

            best_metrics, best_epoch = self.compute_best_validation_metrics()
            self.log('best_epoch', int(best_epoch), prog_bar=True, batch_size=self.batch_size)
            for metric, value in best_metrics.items():
                self.log(f'best_{metric}', value, prog_bar=True, batch_size=self.batch_size)
        self.validation_step_outputs.clear()
    def test_epoch_end(self):
        for metric in self.test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.test_step_outputs, metric)
            #self.metrics.setdefault(f'{metric}/test_epoch', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

        # if (self.current_epoch + 1) % self.test_epochs == 0:
        #     sampling_results = self.sample_and_analyze(self.test_dataloader())
        #     for metric_name, metric_value in sampling_results.items():
        #         self.log(f'{metric_name}/test', metric_value, prog_bar=True)
        #         self.metrics.setdefault(f'{metric_name}/test', []).append(metric_value)
        self.test_step_outputs.clear()
    
    
    def generate_animation(self, chain_batch, batch_i,batch_seg,metals):
        
        idx2atom = const.IDX2ATOM
        idx2metals=const.idx2metals
        #here we only visualize the second complex in the batch
        pos = chain_batch[:,batch_seg==1,:3]
        onehot = chain_batch[:,batch_seg==1,3:]
        metal=metals[1]
        n_atoms =pos.shape[1]
        name = f'mol_{batch_i}'
        chain_output = os.path.join(self.samples_dir, f'epoch_{self.current_epoch}', name)
        os.makedirs(chain_output, exist_ok=True)
        for j in range(self.FRAMES):
            f = open(os.path.join(chain_output, f'{name}_{j}.xyz'), "w")
            f.write("%d\n\n" % n_atoms)
            atoms = torch.argmax(onehot[j], dim=1)
            for atom_i in range(n_atoms):
                if atom_i==0:
                    atom=idx2metals[metal.item()]
                else:
                    atom = idx2atom[atoms[atom_i].item()]
                f.write("%s %.5f %.5f %.5f\n" % (
                    atom, pos[j][atom_i, 0], pos[j][atom_i, 1], pos[j][atom_i, 2]
                ))
            f.close()
        visualize_chain(chain_output, wandb=wandb, mode=name)
    
    
    @torch.no_grad()
    def sample_and_analyze(self, dataset,batch_size=None,outdir='generated_samples',animation=False):
        batch_size=self.batch_size if batch_size is None else batch_size
        os.makedirs(outdir, exist_ok=True)
        valid_comp=0
        valid_ligand=0
        connected_ligand=0
        connected_mols=[]
        sa_score=0
        num=0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for b, data in enumerate(dataloader):
            pos_original=data['pos']
            batch_seg=data.batch
            batch_size=torch.max(batch_seg)+1
            ligand_diff = data['ligand_diff'].view(-1,1)
            context=data['context'].view(-1,1)
            metals=[data['nuclear_charges'][batch_seg==i][0] for i in range(batch_size)]
            fixed_mean=scatter_add(pos_original*context, batch_seg, dim=0)/scatter_add(context, batch_seg, dim=0).view(-1,1)
            natoms=data['num_atoms']
            try:
                chain_batch = self.sample_chain(data, keep_frames=self.FRAMES)
            except utils.FoundNaNException as e:
                continue
                
            if animation and self.samples_dir is not None and b in [0, 1]:
                self.generate_animation(chain_batch=chain_batch, batch_i=b,batch_seg=batch_seg,metals=metals)
            # Get final complexes from chains – for computing metrics
            x = chain_batch[0][ :, :3]
            x=x+fixed_mean[batch_seg]
            one_hot = chain_batch[0][ :, 3:]
            assert one_hot.shape[1]==self.in_node_nf
            ligands=extract_ligand(x,one_hot,ligand_diff,batch_seg)
            rdmols=[build_mol(*graph) for graph in ligands]
            (validity, connectivity), (valid, connected_mol,connected_index) = self.ligand_metrics.evaluate_rdmols(rdmols)
            connected_mols.extend([Chem.MolToSmiles(mol) for mol in connected_mol])
            valid_ligand+=validity
            connected_ligand+=connectivity
            if connectivity!=0:
                assert max(connected_index)<=batch_size
                for i,mol in zip(connected_index,connected_mol):
                    positions=x[batch_seg==i]
                    atom_types=one_hot[batch_seg==i].argmax(dim=1)
                    metal=metals[i]
                    overlapping,liglist=sanitycheck(positions, atom_types,metal)
                    total_atoms=sum(len(lig) for lig in liglist)+1
                    #check if there exists atoms overlapping in the complex or some ligands are not coordinated to the metal
                    if not overlapping and total_atoms ==natoms[i].item():
                        valid_comp+=1
                        sa_score += compute_sa_score(mol)
                        
                        write_xyz_file(positions, atom_types,f'{outdir}/{b}_{i}', metal)


        train_smiles=pd.read_csv('../data/train_smiles.csv')
        train_smiles=train_smiles['smiles'].tolist()
        for i in connected_mols:
            if i in train_smiles:
                num+=1
        
        
        metrics={'valid_ligand':valid_ligand/len(dataset),
                 'connected_ligand':connected_ligand/valid_ligand,
                 'valid_complex':valid_comp/len(dataset),
                 'uniqueness':len(set(connected_mols))/connected_ligand,
                 'novelty':1-(num/connected_ligand),
                 'sa_score':sa_score/valid_comp}
        
        print(f'Finish sampling on  {len(dataset)} samples')
        print('Metrics for sampling:')
        print(metrics)
        return metrics               

    def sample_chain(self, data, keep_frames=None):

        x = data['pos']
        h = data['one_hot']
        context = data['context'].view(-1,1)
        ligand_diff = data['ligand_diff'].view(-1,1)
        batch_seg=data.batch
        batch_size=int(torch.max(batch_seg))+1
        ligand_group=data['ligand_group']

        if self.center_of_mass == 'context':
            x= utils.remove_partial_mean_with_mask(x,context,batch_seg)
        elif self.center_of_mass == 'ligand_diff':
            x= utils.remove_partial_mean_with_mask(x,ligand_diff,batch_seg)
        else:
            raise ValueError(f'Unknown center_of_mass: {self.center_of_mass}')
        
        chain = self.edm.sample_chain(
            x=x,
            h=h,
            context=context,
            ligand_diff=ligand_diff,
            batch_seg=batch_seg,
            batch_size=batch_size,
            ligand_group=ligand_group,
            keep_frames=keep_frames)
        
        return chain 

    def configure_optimizers(self):
        return torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-6)


    def configure_gradient_clipping(self, optimizer,gradient_clip_val, gradient_clip_algorithm):
                                
        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
            

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'valid_complex/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        }
        return best_metrics, best_epoch

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
    


   

