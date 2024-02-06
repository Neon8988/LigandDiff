import os
import torch
import numpy as np
import tempfile
from rdkit import Chem
from openbabel import openbabel
from src import const
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown
import warnings
warnings.filterwarnings("ignore")

def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond

def extract_ligand(x,onehot,ligand_diff,batch_seg):
    unique_indices = torch.unique(batch_seg)
    ligands=[]
    for idx in unique_indices:
        ligand_diffs = ligand_diff[batch_seg == idx]
        indices = (ligand_diffs == 1).nonzero(as_tuple=True)[0]
        pos = x[batch_seg == idx][indices]
        hs = onehot[batch_seg == idx][indices]
        atoms = torch.argmax(hs, dim=1)
        ligands.append(list((pos, atoms)))
    return ligands

def write_xyz_file(coords, atom_types,filename,metal):
    idx2atom = const.IDX2ATOM
    idx2metals=const.idx2metals
    f=open(f'{filename}.xyz','w')
    assert len(coords) == len(atom_types)
    f.write("%d\n\n" % len(coords))
    if metal==None:
        for i in range(len(coords)):
            atom=idx2atom[atom_types[i].item()]
            f.write(f"{atom} {coords[i, 0]:.5f} {coords[i, 1]:.5f} {coords[i, 2]:.5f}\n")
        f.close()

    else:
        for i in range(len(coords)):
            if i ==0:          
                atom=idx2metals[metal.item()]
            else:
                atom=idx2atom[atom_types[i].item()]
            f.write(f"{atom} {coords[i, 0]:.5f} {coords[i, 1]:.5f} {coords[i, 2]:.5f}\n")
        f.close()

def build_mol(positions, atom_types,use_openbabel=True):
                   
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types)
                                
    else:
        raise NotImplementedError

    return mol



def make_mol_openbabel(positions, atom_types):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
    Returns:
        rdkit molecule
    """
    openbabel.obErrorLog.StopLogging()

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file,metal=None)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")     
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, f'{tmp_file}.xyz')
        obConversion.WriteFile(ob_mol, f'{tmp_file}.sdf')
        # Read sdf file with RDKit
        tmp_mol = Chem.SDMolSupplier(f'{tmp_file}.sdf', sanitize=False)[0]
    # Build new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))

    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                bond.GetBondType())

    return mol
   


class BasicLigandMetrics(object):
    def __init__(self,connectivity_thresh=1.0):
                 
        self.connectivity_thresh = connectivity_thresh

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        if len(generated) < 1:
            return [], 0.0

        valid = []
        for index,mol in enumerate(generated):
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue

            valid.append((index,mol))

        return valid, len(valid)

    def compute_connectivity(self, valid):
        """ Consider molecule connected if its largest fragment contains at
        least x% of all atoms, where x is determined by
        self.connectivity_thresh (defaults to 100%). """
        if len(valid) < 1:
            return [],[], 0.0

        connected = []
        connected_index=[]
        for index,mol in valid:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            if len(mol_frags) == 1:
                connected.append(mol_frags[0])
                connected_index.append(index)

        return connected,connected_index, len(connected_index)


    def evaluate_rdmols(self, rdmols):
        valid, validity = self.compute_validity(rdmols)

        connected,connected_index, connectivity = \
            self.compute_connectivity(valid)

        return [validity, connectivity], [valid, connected,connected_index]



def sanitycheck(positions, atom_types,metal):
    """
    Using molsimplify to check whether atoms are overlapping
    """
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        write_xyz_file(positions, atom_types, tmp_file,metal)
    
    mol=mol3D()
    mol.readfromxyz(f'{tmp_file}.xyz')
    overlapping=mol.sanitycheck(silence=True)[0]
    liglist,ligdents,ligcon=ligand_breakdown(mol,silent=True,BondedOct=True)
    return overlapping,liglist
    
    
