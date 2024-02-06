import argparse
from pathlib import Path
import os
import numpy as np
import torch
from src import const
from src.lightning import DDPM
from torch_geometric.data import Data


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=Path)
parser.add_argument('--model', type=Path)
parser.add_argument('--dataset', type=Path)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ligand_sizes', type=str, default='random')

num_atom_types=const.NUMBER_OF_ATOM_TYPES
def get_ligand_size(ligand_size='random'):
    if ligand_size == 'random':
        ligand_size=np.random.choice(range(6,21),1)[0]
    else:
        ligand_size=int(ligand_size)
    return ligand_size

def reform_data(dataset,device,ligand_sizes='random'):
    new_data=[]
    for i in dataset:
        x=i['pos'][i['context']==1]
        one_hot=i['one_hot'][i['context']==1]
        ligand_group=i['ligand_group'][i['context']==1]
        nuclear_charges=i['nuclear_charges'][i['context']==1]
        ligand_group_linker=i['ligand_group'][i['ligand_diff']==1]
        index=torch.nonzero(ligand_group_linker)[0][-1]
        ligand_size=get_ligand_size(ligand_sizes)
        new_ligand_linker=torch.zeros(ligand_size,6)
        new_ligand_linker[:,index]=1
        new_x_linker=torch.zeros(ligand_size,3)
        new_onehot_linker=torch.zeros(ligand_size,num_atom_types)
        new_x=torch.cat([x,new_x_linker],dim=0)
        new_context_mask=torch.cat([torch.ones(x.shape[0]),torch.zeros(ligand_size)],dim=0)
        new_ligand_diff_mask=torch.cat([torch.zeros(x.shape[0]),torch.ones(ligand_size)],dim=0)
        new_charges=torch.cat([nuclear_charges,torch.zeros(ligand_size)],dim=0)
        assert new_x.shape[0]==new_charges.shape[0]
        new_ligand_group=torch.cat([ligand_group,new_ligand_linker],dim=0)
        new_onehot=torch.cat([one_hot,new_onehot_linker],dim=0)
        natoms=new_x.shape[0]
        data = Data(pos=new_x.to(device),label=i['label'],nuclear_charges=new_charges.to(device), context=new_context_mask.to(device), ligand_diff=new_ligand_diff_mask.to(device), ligand_group=new_ligand_group.to(device), one_hot=new_onehot.to(device), num_atoms=natoms)
        new_data.append(data)
    return new_data


def main(outdir,model,dataset,batch_size=64,ligand_sizes='random'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataset=torch.load(dataset)
    new_data=reform_data(dataset,device,ligand_sizes=ligand_sizes)
    batch_size=batch_size if batch_size is not None else ddpm.batch_size
    with torch.no_grad():
        ddpm.sample_and_analyze(new_data, batch_size=batch_size,outdir=outdir,animation=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.outdir,args.model,args.dataset,args.batch_size,args.ligand_sizes)




