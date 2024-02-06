import argparse
import os
import numpy as np
import tempfile
import torch
from src import const
from src import utils
from src.lightning import DDPM
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sampling import reform_data
from torch_scatter import scatter_add
from src.molecule_builder import build_mol,extract_ligand,sanitycheck,write_xyz_file
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--complex', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--ligand_sizes', type=str, default='random')

atom2idx=const.ATOM2IDX 
idx2atom=const.IDX2ATOM 
charges=const.CHARGES
num_atoms_type=const.NUMBER_OF_ATOM_TYPES
metal_list=const.metals

def reform_pos(xyz_file):
    metal_index = None
    with open(xyz_file, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(tuple(metal_list)):
            metal_index = i
            break
    if metal_index is not None:
        lines.insert(2, lines.pop(metal_index))
        with open(f'{xyz_file[:-4]}_re.xyz', 'w') as file:
            file.writelines(lines)

def parse_complex(filename):
    label=filename[:-4]
    data_list=[]
    ele=[]
    pos=[]
    nuclear_charges=[]
    H_list=[]# store H atoms, maybe add them back later
    noH_list=[]
    with open(filename, 'r') as f:
        lines=f.readlines()
    
    for i in lines[3:]:
        if i.split()[0] =='H':
            H_list.append(i)
        else:
            noH_list.append(i)
            ele.append(atom2idx[i.split()[0]])
            nuclear_charges.append(charges[i.split()[0]])
            pos.append([float(j) for j in i.split()[1:]])
    noH_list.insert(0,lines[2])
    pos.insert(0,[float(j) for j in lines[2].split()[1:]]) # add metal position
    nuclear_charges.insert(0,charges[lines[2].split()[0]]) # add metal charge
    one_hot=torch.zeros(len(ele),8)
    one_hot[range(len(ele)),ele]=1
    one_hot=torch.cat([torch.zeros(8).view(1,-1),one_hot],dim=0)
    num_atoms=len(pos)
    pos=torch.tensor(pos)
    nuclear_charges=torch.tensor(nuclear_charges)

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        with open(f'{tmp_file}.xyz', 'w') as file:
            file.write(f"{num_atoms}\n\n")
            for sublist in noH_list:
                file.write(f"{sublist}")
    mol=mol3D()
    mol.readfromxyz(f'{tmp_file}.xyz')
    liglist,ligdents,ligcon=ligand_breakdown(mol,silent=True,BondedOct=True)
    f_group=torch.zeros(num_atoms)
    for i in range(len(liglist)):
        f_group[liglist[i]]=i+1   

    ligand_group=torch.zeros((num_atoms,7) )
    ligand_group[range(len(f_group.long())),f_group.long()]=1

    for k in range(len(liglist)):
        ligand=torch.zeros(num_atoms)
        for i in liglist[k]:
            ligand[i]=1
        context = 1-ligand
        data = Data(pos=pos,label=label,  context=context,  nuclear_charges=nuclear_charges,ligand_diff=ligand, num_atoms=num_atoms, one_hot=one_hot,ligand_group=ligand_group[:,1:])
        data_list.append(data)
    return data_list






def read_molecule(filename):
    if not filename.endswith('.xyz'):
        raise Exception('Unknown file extension, only .xyz files are supported')
    
    with open(filename, 'r') as file:
        metal = file.readlines()[2]
        if metal.split()[0] not in metal_list:
            reform_pos(filename)
            print(f'Metal is not located at the begining of the coordinates.The {filename} is rearranged and saved to {filename[:-4]}_re.xyz')
            return parse_complex(f'{filename[:-4]}_re.xyz')
        else:
            return parse_complex(filename)
        


def main(outdir,model,complex,batch_size=64,n_samples=1,ligand_sizes='random'):
    """
    Generate multiple samples for each ligand in a given complex
    Args:
        outdir: path to save generated complexes
        n_samples: number of samples to generate for each ligand
        ligand_size: fix the ligand size or randomly sample from 6 to 20
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataset=read_molecule(complex)*n_samples
    print(f'{len(dataset)} samples will be generated')
    data=reform_data(dataset,device,ligand_sizes=ligand_sizes)
    batch_size=min(batch_size,len(dataset))
    os.makedirs(outdir, exist_ok=True)
    num=0
    with torch.no_grad():
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        for b, data in enumerate(dataloader):
            pos_orginal=data['pos']
            batch_seg=data.batch
            batch_size=torch.max(batch_seg)+1
            ligand_diff = data['ligand_diff'].view(-1,1)
            context=data['context'].view(-1,1)
            metals=[data['nuclear_charges'][batch_seg==i][0] for i in range(batch_size)]
            fixed_mean=scatter_add(pos_orginal*context, batch_seg, dim=0)/scatter_add(context, batch_seg, dim=0).view(-1,1)
            natoms=data['num_atoms']
            try:
                chain_batch = ddpm.sample_chain(data, keep_frames=100)
            except utils.FoundNaNException as e:
                continue
  
            x = chain_batch[0][ :, :3]
            x=x+fixed_mean[batch_seg]
            one_hot = chain_batch[0][ :, 3:]
            assert one_hot.shape[1]==ddpm.in_node_nf
            ligands=extract_ligand(x,one_hot,ligand_diff,batch_seg)
            rdmols=[build_mol(*graph) for graph in ligands]
            (validity, connectivity), (valid, connected_mol,connected_index) = ddpm.ligand_metrics.evaluate_rdmols(rdmols)
            if connectivity!=0:
                assert max(connected_index)<=batch_size
                for i in connected_index:
                    positions=x[batch_seg==i]
                    atom_types=one_hot[batch_seg==i].argmax(dim=1)
                    metal=metals[i]
                    overlapping,liglist=sanitycheck(positions, atom_types,metal)
                    total_atoms=sum(len(lig) for lig in liglist)+1
                    if not overlapping and total_atoms ==natoms[i].item():
                        num+=1
                        write_xyz_file(positions, atom_types,f'{outdir}/{b}_{i}', metal)

    print('Done')
    print(f'Totally {num} valid complexes are generated and saved in {outdir}')

        


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.outdir,args.model,args.complex,args.batch_size,args.n_samples,args.ligand_sizes)

    

    