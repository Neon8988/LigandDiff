import torch

from rdkit import Chem


TORCH_FLOAT = torch.float32
TORCH_INT = torch.int8


# Atom idx for one-hot encoding
ATOM2IDX = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'Br': 4, 'Cl': 5, 'P': 6, 'F': 7}
IDX2ATOM = {0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'Br', 5: 'Cl', 6: 'P', 7: 'F'}
idx2metals={24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',44:'Ru',46:'Pd',78:'Pt'}
metals=['Cr','Mn','Fe','Co','Ni','Cu','Zn','Ru','Pd','Pt']

CHARGES = {'C': 6, 'O': 8, 'N': 7, 'S': 16, 'Cl': 17, 'P': 15, 'Br': 35, 'F': 9,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ru':44,'Pd':46,'Pt':78}
# One-hot atom types
NUMBER_OF_ATOM_TYPES = len(ATOM2IDX)



# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
BONDS_1 = {
    'H': {
        'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
        'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
        'Cl': 127, 'Br': 141, 'I': 161
    },
    'C': {
        'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
        'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
        'I': 214
    },
    'N': {
        'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177
    },
    'O': {
        'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194
    },
    'F': {
        'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187
    },
    'B': {
        'H':  119, 'Cl': 175
    },
    'Si': {
        'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
        'F': 160, 'Cl': 202, 'Br': 215, 'I': 243,
    },
    'Cl': {
        'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
        'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
        'Br': 214
    },
    'S': {
        'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
        'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
        'I': 234
    },
    'Br': {
        'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
        'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222
    },
    'P': {
        'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
        'S': 210, 'F': 156, 'N': 177, 'Br': 222
    },
    'I': {
        'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
        'S': 234, 'F': 187, 'I': 266
    },
    'As': {
        'H': 152
    }
}

BONDS_2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

BONDS_3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}

BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND2IDX = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

ALLOWED_BONDS = {
    'H': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'B': 3,
    'Al': 3,
    'Si': 4,
    'P': [3, 5],
    'S': 4,
    'Cl': 1,
    'As': 3,
    'Br': 1,
    'I': 1,
    'Hg': [1, 2],
    'Bi': [3, 5]
}

MARGINS_EDM = [10, 5, 2]

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8','C9','C10']

RADII = [0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,0.77,0.77]

