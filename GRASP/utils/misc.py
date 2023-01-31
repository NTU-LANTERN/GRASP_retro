# encoding=utf8

import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# from scipy.spatial.distance import rogerstanimoto, jaccard

RDLogger.DisableLog('rdApp.*')

def tokenizer(smiles):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>>?|\*|\$|\%\([0-9]{3}\)|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smiles)]
    assert smiles == ''.join(tokens), smiles + "\n" + ''.join(tokens)
    return ' '.join(tokens)

def detokenizer(smiles):
    return smiles.strip().replace(' ', '')


def remove_atom_mapping(smiles, canonical=True, isomeric=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    else:
        return ''


def remove_rxn_atom_mapping(rxn, canonical=True, isomeric=True):
    reactant, reagent, product = rxn.split(">")
    reactant = remove_atom_mapping(reactant, canonical=canonical, isomeric=isomeric)
    reagent = remove_atom_mapping(reagent, canonical=canonical, isomeric=isomeric)
    product = remove_atom_mapping(product, canonical=canonical, isomeric=isomeric)
    return reactant + '>' + reagent + '>' + product


def process_smiles(smiles, canonical=True, isomeric=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    else:
        return ''


def process_rxn_smiles(rxn, canonical=True, isomeric=True):
    reactant, reagent, product = rxn.split(">")
    reactant = process_smiles(reactant, canonical=canonical, isomeric=isomeric)
    reagent = process_smiles(reagent, canonical=canonical, isomeric=isomeric)
    product = process_smiles(product, canonical=canonical, isomeric=isomeric)
    return reactant + '>' + reagent + '>' + product


def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)

    return arr

def batch_smiles_to_fp(s_list, fp_dim=2048):
    fps = []
    for s in s_list:
        fps.append(smiles_to_fp(s, fp_dim))
    fps = np.array(fps)

    assert fps.shape[0] == len(s_list) and fps.shape[1] == fp_dim

    return fps
    
def canonicalize_basic(smi: str):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), True)


def canonicalize(smi: str) -> str:
    try:
        res = ".".join(sorted(canonicalize_basic(s) for s in smi.split(".")))
        return res
    except:
        res = ''
        return res

def Rogerstanimoto(u, v):
    if u.dtype==v.dtype==bool:
        not_u, not_v = ~u, ~v
        nff = (not_u & not_v).sum()
        nft = (not_u & v).sum()
        ntf = (u & not_v).sum()
        ntt = (u & v).sum()
    else:
        dtype = np.result_type(int, u.dtype, v.dtype)
        u = u.astype(dtype)
        v = v.astype(dtype)
        not_u = 1.0 - u
        not_v = 1.0 - v
        nff = (not_u * not_v).sum()
        nft = (not_u * v).sum()
        ntf = (u * not_v).sum()
        ntt = (u * v).sum()
    return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))


def Jaccard(u, v):
    nonzero = np.bitwise_or(u != 0, v != 0)
    unequal_nonzero = np.bitwise_and((u != v), nonzero)
    a = np.double(unequal_nonzero.sum())
    b = np.double(nonzero.sum())
    return (a / b) if b != 0 else 0


def similarity(a, b, metric='Rogerstanimoto'):
    if isinstance(a, str):
        a = smiles_to_fp(a)
    if isinstance(b, str):
        b = smiles_to_fp(b)
    if metric == 'Jaccard':
        return 1. - Jaccard(a, b)
    elif metric == 'Rogerstanimoto':
        return 1. - Rogerstanimoto(a, b)