import re
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondDir, BondStereo, BondType
import functools
from itertools import tee

def Property1(inp):
    name = '_{}'.format(inp.__name__)
    @functools.wraps(inp)
    def Property2(self):
        if not hasattr(self, name):
            setattr(self, name, inp(self))
        return getattr(self, name)
    return property(Property2)


def Property3(inp):
    a, b = tee(inp)
    next(b, None)
    return zip(a, b)

class Function1(object):
    BONDMAP = {'-': BondType.SINGLE, '=': BondType.DOUBLE,
            '#': BondType.TRIPLE, ':': BondType.AROMATIC}
    CHARGEMAP = {'+': 1, '0': 0, '-': -1}

    def __init__(self, name, smarts, bonds=(), charges=(), radicals=()):
        self.name = name
        self.survival_str = smarts
        self.bonds = [self.BONDMAP[b] for b in bonds]
        self.charges = [self.CHARGEMAP[b] for b in charges]

    @Property1
    def survival(self):
        return Chem.MolFromSmarts(self.survival_str)

    def __repr__(self):
        return 'Function1({!r}, {!r}, {!r}, {!r})'.format(self.name, self.survival_str, self.bonds, self.charges)

    def __str__(self):
        return self.name


class Function2(object):
    def __init__(self, name, smarts, score):
        self.name = name
        self.smarts_str = smarts
        self.score = score

    @Property1
    def smarts(self):
        return Chem.MolFromSmarts(self.smarts_str)

    def __repr__(self):
        return 'Function2({!r}, {!r}, {!r})'.format(self.name, self.smarts_str, self.score)

    def __str__(self):
        return self.name

Property4 = (
    Function1('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
    Function1('1,5 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[CH0]=[C]-[C]=[N]'),
    Function1('special imine r', '[CX4!H0]-[c]=[n]'),
    Function1('1,3 aromatic heteroatom H shift f', '[#7!H0]-[#6R1]=[#7X2]'),
    Function1('1,3 aromatic heteroatom H shift r', '[O,#7;!H0]-[#6R1]=[#7X2]'),
    Function1('1,3 heteroatom H shift',
                      '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7]'),
    Function1('1,5 aromatic heteroatom H shift f',
                      '[#7H;R]-[#6,nX2;R]=[#6,nX2;R]-[#6,#7X2;R]=[#7X2;!R]'),
    Function1('1,7 aromatic heteroatom H shift f',
                      '[#8,#16,Se,Te;!H0]-[#6,#7X2]1=[#6,#7X2]-[#6,#7X2]=[#6]-[#6,#7X2]=[#7X2,CX3]1'),
    Function1('ionic nitro/aci-nitro f', '[C!H0]-[N+;$([N][O-])]=[O]'),
    Function1('ionic nitro/aci-nitro r', '[O!H0]-[N+;$([N][O-])]=[C]'),
    Function1('oxim/nitroso f', '[O!H0]-[N]=[C]'),
    Function1('oxim/nitroso via phenol f', '[O!H0]-[N]=[C]-[C]=[C]-[C]=[OH0]'),
    Function1('oxim/nitroso via phenol r', '[O!H0]-[c]=[c]-[c]=[c]-[N]=[OH0]'),
    Function1('isocyanide f', '[C-0!H0]#[N+0]', bonds='#', charges='-+'),
    Function1('isocyanide r', '[N+!H0]#[C-]', bonds='#', charges='-+'),
)

Property5 = (
    Function2('benzoquinone', '[#6]1([#6]=[#6][#6]([#6]=[#6]1)=,:[N,S,O])=,:[N,S,O]', 25),
    Function2('oxim', '[#6]=[N][OH]', 4),
    Function2('-oxim', '[OH][#6]=[N][OH]', -5),
    Function2('C=O', '[#6]=,:[#8]', 2),
    Function2('N=O', '[#7]=,:[#8]', 2),
    Function2('P=O', '[#15]=,:[#8]', 2),
    Function2('C=hetero', '[#6]=[!#1;!#6]', 1),
    Function2('methyl', '[CX4H3]', 1),
    Function2('guanidine terminal=N', '[#7][#6](=[NR0])[#7H0]', 1),
    Function2('guanidine endocyclic=N', '[#7;R][#6;R]([N])=[#7;R]', 2),
    Function2('alpha-OH COOH', '*C([OH])C(=O)[O-,OH]', 2),
    Function2('special trimer-ring', 'O=C1CCOc2cc3c(ccn3)cc12', 5),
    Function2('6-element nH', '[nH;r6;!$(*c=O)]', -5000),
    Function2('6,5-element N=', 'N=[c,n;r6,r5]', -5000),
    Function2('aci-nitro', '[#6]=[N+]([O-])[OH]', -4),
    Function2('aci-nitro', '[n,N;H0,r5]=,:c[nH;r6]', -2),
    Function2('*#*-N=C-N', 'NC=N*#*', 1),
    Function2('a-N=C-N', 'NC=Na', -1),
    Function2('N=Ring', '[NH;!R]=[C;R]', -1),
    Function2('Ring=Ring', '[C;R]=[C;R]', 1),
    Function2('HO-C=N', '[OH]-[C,c]=,:[N,n]', -5000),
    Function2('O=a1aa[nH]aa1', 'O=a1aa[nH]aa1', -1),
    Function2('N=C-N = in a ring', 'N[C;R]=[N;R]', 0.5),
)

Property6 = 1000


class Function3(object):
    def __init__(self, transforms=Property4, scores=Property5, Property6=Property6):
        self.transforms = transforms
        self.scores = scores
        self.Property6 = Property6

    def __call__(self, mol):
        return self.canonicalize(mol)

    def canonicalize(self, mol):
        survivals = self._enumerate_survivals(mol)
        if len(survivals) == 1:
            return survivals[0]
        highest = None
        for t in survivals:
            smiles = Chem.MolToSmiles(t, isomericSmiles=True)
            score = 0
            ssr = Chem.GetSymmSSSR(t)
            for ring in ssr:
                btypes = {t.GetBondBetweenAtoms(*pair).GetBondType() for pair in Property3(ring)}
                elements = {t.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring}
                if btypes == {BondType.AROMATIC}:
                    score += 100
                    if elements == {6}:
                        score += 150
            for tscore in self.scores:
                for match in t.GetSubstructMatches(tscore.smarts):
                    score += tscore.score
            for atom in t.GetAtoms():
                if atom.GetAtomicNum() in {15, 16, 34, 52}:
                    hs = atom.GetTotalNumHs()
                    if hs:
                        score -= hs
            if not highest or highest['score'] < score or (highest['score'] == score and smiles < highest['smiles']):
                highest = {'smiles': smiles, 'survival': t, 'score': score}
        return highest['survival']
    @Property1
    def _enumerate_survivals(self):
        return Function4(self.transforms, self.Property6)


class Function4(object):

    def __init__(self, transforms=Property4, Property6=Property6):
        self.transforms = transforms
        self.Property6 = Property6

    def __call__(self, mol):
        return self.enumerate(mol)

    def enumerate(self, mol):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        survivals = {smiles: copy.deepcopy(mol)}
        kekulized = copy.deepcopy(mol)
        Chem.Kekulize(kekulized)
        kekulized = {smiles: kekulized}
        done = set()
        while len(survivals) < self.Property6:
            for tsmiles in sorted(survivals):
                if tsmiles in done:
                    continue
                for transform in self.transforms:
                    for match in kekulized[tsmiles].GetSubstructMatches(transform.survival):
                        product = copy.deepcopy(kekulized[tsmiles])
                        first = product.GetAtomWithIdx(match[0])
                        if first.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                            continue
                        has_stereo = False
                        for match_idx in range(1,len(match),2):
                            bond = product.GetBondBetweenAtoms(match[match_idx],match[match_idx+1])
                            if str(bond.GetStereo()) != 'STEREONONE':
                                has_stereo = True
                                break
                        if has_stereo: continue
                        last = product.GetAtomWithIdx(match[-1])
                        first.SetNumExplicitHs(max(0, first.GetTotalNumHs() - 1))
                        last.SetNumExplicitHs(last.GetTotalNumHs() + 1)
                        first.SetNoImplicit(True)
                        last.SetNoImplicit(True)
                        for bi, pair in enumerate(Property3(match)):
                            if transform.bonds:
                                product.GetBondBetweenAtoms(*pair).SetBondType(transform.bonds[bi])
                            else:
                                current_bond_type = product.GetBondBetweenAtoms(*pair).GetBondType()
                                product.GetBondBetweenAtoms(
                                    *pair).SetBondType(BondType.DOUBLE if current_bond_type == BondType.SINGLE else BondType.SINGLE)
                        if transform.charges:
                            for ci, idx in enumerate(match):
                                atom = product.GetAtomWithIdx(idx)
                                atom.SetFormalCharge(atom.GetFormalCharge() + transform.charges[ci])
                        try:
                            Chem.SanitizeMol(product)
                            smiles = Chem.MolToSmiles(product, isomericSmiles=True)
                            if smiles not in survivals:
                                kekulized_product = copy.deepcopy(product)
                                Chem.Kekulize(kekulized_product)
                                survivals[smiles] = product
                                kekulized[smiles] = kekulized_product
                            else:
                                pass
                        except:
                            pass
                done.add(tsmiles)
            if len(survivals) == len(done):
                break
        else:
            pass
        for survival in survivals.values():
            Chem.AssignStereochemistry(survival, force=True, cleanIt=True)
            for bond in survival.GetBonds():
                if bond.GetBondType() == BondType.DOUBLE and bond.GetStereo() > BondStereo.STEREOANY:
                    begin = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()
                    for othersurvival in survivals.values():
                        if not othersurvival.GetBondBetweenAtoms(begin, end).GetBondType() == BondType.DOUBLE:
                            neighbours = survival.GetAtomWithIdx(
                                begin).GetBonds() + survival.GetAtomWithIdx(end).GetBonds()
                            for otherbond in neighbours:
                                if otherbond.GetBondDir() in {BondDir.ENDUPRIGHT, BondDir.ENDDOWNRIGHT}:
                                    otherbond.SetBondDir(BondDir.NONE)
                            Chem.AssignStereochemistry(survival, force=True, cleanIt=True)
                            break
        return list(survivals.values())


def Function5(smi, sep='~', expression=False):
    smi = smi.replace(sep, '~')
    property10 = {
        'O=[Si]': '[O+]#[Si-]',
        '[C]=O': '[C-]#[O+]',
        '[Al+3]~[H-]~[H-]~[H-]~[H-]~[Li+]': '[AlH4-]~[Li+]',
        '[H][Al]([H])[H]~[H][Li]':'[AlH4-]~[Li+]',
        '[AlH3]~[LiH]':'[AlH4-]~[Li+]',
        '[C-]#N~[Na+]': 'N#C[Na]',
        '[C-]#N~[Li+]': '[Li]C#N',
        '[C-]#N~[K+]': 'N#C[K]',
        '[Cl-]~[Cl-]~[Pd+4]~c1ccc([P-](c2ccccc2)c2ccccc2)cc1~c1ccc([P-](c2ccccc2)c2ccccc2)cc1': '[Cl-]~[Cl-]~[Pd+2]~c1ccc(P(c2ccccc2)c2ccccc2)cc1~c1ccc(P(c2ccccc2)c2ccccc2)cc1',
        '[Pd+4]~c1ccc([P-](c2ccccc2)c2ccccc2)cc1~c1ccc([P-](c2ccccc2)c2ccccc2)cc1~c1ccc([P-](c2ccccc2)c2ccccc2)cc1~c1ccc([P-](c2ccccc2)c2ccccc2)cc1': '[Pd]~c1ccc(P(c2ccccc2)c2ccccc2)cc1~c1ccc(P(c2ccccc2)c2ccccc2)cc1~c1ccc(P(c2ccccc2)c2ccccc2)cc1~c1ccc(P(c2ccccc2)c2ccccc2)cc1',
        '[C-]~[Pd+]': '[C]~[Pd]',
        '[Cl-]~[Cl-]~[Fe+2]~[Pd+2]~c1ccc(P(c2ccccc2)[c-]2cccc2)cc1~c1ccc(P(c2ccccc2)[c-]2cccc2)cc1': '[CH]1[CH][CH][C](P(c2ccccc2)c2ccccc2)[CH]1~[CH]1[CH][CH][C](P(c2ccccc2)c2ccccc2)[CH]1~[Cl-]~[Cl-]~[Fe]~[Pd+2]',    
        'Cl~Cl~Cl[Pt](Cl)(Cl)Cl':'Cl[Pt-2](Cl)(Cl)(Cl)(Cl)Cl~[H+]~[H+]',
        'FB(F)F~F[Na]':'F[B-](F)(F)F~[Na+]',
        '[C][Pd]':'[C].[Pd]',
    }
    if smi in property10:
        smi = property10[smi]
    smis = []
    for s in smi.split('~'):
        smis.append(s)
    return sep.join(sorted(smis))


def Function6(rsmi, clean_isotops=True):
    if clean_isotops:
        rsmi = re.sub(r'(?<=\[)\d+([a-zA-Z]{1,3}[\+-]?\d*\])', lambda s:s.group() if s.group(1)=='Rn]' else s.group(1), rsmi)
    if 'Ac' in rsmi:
        rsmi = re.sub(r'\[Ac-(:?\d*)\]', 'CC(=O)[O-]', rsmi)
        rsmi = re.sub(r'\[AcH(:?\d*)\]', 'CC(=O)O', rsmi)
        rsmi = rsmi.replace('O=[Ac]O[Ac]=O', 'CC(=O)OC(=O)C')
        rsmi = rsmi.replace('Cl[Ac](Cl)Cl', 'Cl[Al](Cl)Cl')
        rsmi = rsmi.replace('O=[Ac]=O', 'CC(=O)OC(=O)C')
        rsmi = rsmi.replace('[CH]Cl.O=[Ac]', 'ClCCl.CC(=O)OC(=O)C')
        rsmi = rsmi.replace('O=[Ac]', 'CC(=O)OC(=O)C')
        rsmi = re.sub(r'(?:^|(?<=[\.\~\$]))\[Ac(:?\d*)\](?=[^=])', 'CC(=O)', rsmi)
        rsmi = re.sub(r'(?<=[^=])\[Ac(:?\d*)\](?:$|(?=[^=]))', 'C(=O)C', rsmi)
    return rsmi


class Property7(object):
    """An acid and its conjugate base, defined by SMARTS.

    A strength-ordered list of Property7s can be used to ensure the strongest acids in a molecule ionize first.
    """

    def __init__(self, name, acid, base):
        """Initialize an Property7 with the following parameters:

        :param string name: A name for this Property7.
        :param string acid: SMARTS pattern for the protonated acid.
        :param string base: SMARTS pattern for the conjugate ionized base.
        """
        self.name = name
        self.acid_str = acid
        self.base_str = base

    @Property1
    def acid(self):
        return Chem.MolFromSmarts(self.acid_str)

    @Property1
    def base(self):
        return Chem.MolFromSmarts(self.base_str)

    def __repr__(self):
        return 'Property7({!r}, {!r}, {!r})'.format(self.name, self.acid_str, self.base_str)

    def __str__(self):
        return self.name

Property8 = (
    Property7('HX', '[Cl,Br,I;H]', '[Cl,Br,I;-]'),
    Property7('-OSO3H', 'OS(=O)(=O)[OH]', 'OS(=O)(=O)[O-]'),
    Property7('-SO3H', '[!O]S(=O)(=O)[OH]', '[!O]S(=O)(=O)[O-]'),
    Property7('-OSO2H', 'O[SD3](=O)[OH]', 'O[SD3](=O)[O-]'),
    Property7('-SO2H', '[!O][SD3](=O)[OH]', '[!O][SD3](=O)[O-]'),
    Property7('-OPO3H2', 'OP(=O)([OH])[OH]', 'OP(=O)([OH])[O-]'),
    Property7('-PO3H2', '[!O]P(=O)([OH])[OH]', '[!O]P(=O)([OH])[O-]'),
    Property7('HF', '[FH]', '[F-]'),
    Property7('-SO3OOH', 'OS(=O)(=O)O[OH]', 'OS(=O)(=O)O[O-]'),
    Property7('-CO2H', 'C(=O)[OH]', 'C(=O)[O-]'),
    Property7('H2S', '[SH2]', '[SH-]'),
    Property7('thiophenol', 'c[SH]', 'c[S-]'),
    Property7('(-OPO3H)-', 'OP(=O)([O-])[OH]', 'OP(=O)([O-])[O-]'),
    Property7('(-PO3H)-', '[!O]P(=O)([O-])[OH]', '[!O]P(=O)([O-])[O-]'),
    Property7('H2O', '[OH2]', '[OH-]'),
    Property7('phthalimide', 'O=C2c1ccccc1C(=O)[NH]2', 'O=C2c1ccccc1C(=O)[N-]2'),
    Property7('CO3H (peracetyl)', 'C(=O)O[OH]', 'C(=O)O[O-]'),
    Property7('-BO2H2', '[!O]B([OH])[OH]', '[!O]B([OH])[O-]'),
    Property7('phenol', 'c[OH]', 'c[O-]'),
    Property7('SH (aliphatic)', 'C[SH]', 'C[S-]'),
    Property7('(-OBO2H)-', 'OB([O-])[OH]', 'OB([O-])[O-]'),
    Property7('(-BO2H)-', '[!O]B([O-])[OH]', '[!O]B([O-])[O-]'),
    Property7('-OH (aliphatic alcohol)', '[CX4][OH]', '[CX4][O-]'),
    Property7('-OBO2H2', 'OB([OH])[OH]', 'OB([OH])[O-]'),
    Property7('alpha-carbon-hydrogen-nitro group', 'O=N(O)[CH]', 'O=N(O)[C-]'),
    Property7('-SO2NH2', 'S(=O)(=O)[NH2]', 'S(=O)(=O)[NH-]'),
    Property7('-CONH2', 'C(=O)[NH2]', 'C(=O)[NH-]'),
    Property7('imidazole', 'c1cnc[nH]1', 'c1cnc[n-]1'),
)

class Function7(object):
    def __init__(self):
        self._pos_h = Chem.MolFromSmarts('[+!H0!$(*~[-])]')
        self._pos_h = Chem.MolFromSmarts('[N+!$(*~[-]);!H0]')
        self._pos_quat = Chem.MolFromSmarts('[+H0!$(*~[-])]')
        self._neg = Chem.MolFromSmarts('[O-,O-2,Cl-,Br-,I-;!$(*~[+H0])]')
        self._neg_acid = Chem.MolFromSmarts('[$([O-][C,P,S]=O),$([n-]1nnnc1),$(n1[n-]nnc1)]')
        self._nh4_o = Chem.MolFromSmarts('[N+!$(*~[-]);!H0].[$([O-][C,P,S]=O)]')
        self._nh4_x = Chem.MolFromSmarts('[N+!$(*~[-]);!H0].[O-,O-2,Cl-,Br-,I-;!$(*~[+H0])]')

    def __call__(self, mol):
        return self.universial(mol)

    def universial(self, mol):
        for smarts in [self._nh4_o,self._nh4_x]:
            pairs = mol.GetSubstructMatches(smarts)
            count = 20
            while pairs and count>0:
                i, j = pairs[0]
                count -= 1
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                atom_i.SetNumExplicitHs(max(0, atom_i.GetTotalNumHs() - 1))
                atom_i.SetNoImplicit(True)
                atom_i.SetFormalCharge(0)
                atom_j.SetFormalCharge(atom_j.GetFormalCharge() + 1)
                atom_j.SetNumExplicitHs(atom_j.GetTotalNumHs() + 1)
                atom_j.SetNoImplicit(True)
                Chem.SanitizeMol(mol)
                pairs = mol.GetSubstructMatches(smarts)
        return mol

    def universial_bak(self, mol):
        mol = copy.deepcopy(mol)
        p = [x[0] for x in mol.GetSubstructMatches(self._pos_h)]
        q = []
        n = []
        a = [x[0] for x in mol.GetSubstructMatches(self._neg_acid)]
        if q:
            neg_surplus = len(n) - len(q)
            if a and neg_surplus > 0:
                while neg_surplus > 0 and a:
                    atom = mol.GetAtomWithIdx(a.pop(0))
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                    neg_surplus -= 1
        else:
            for atom in [mol.GetAtomWithIdx(x) for x in n]:
                while atom.GetFormalCharge() < 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        for atom in [mol.GetAtomWithIdx(x) for x in p]:
            while atom.GetFormalCharge() > 0 and atom.GetNumExplicitHs() > 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
                atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        return mol


class Function8(object):
    def __init__(self, Property8=Property8, charge_corrections=()):
        self.Property8 = Property8
        self.charge_corrections = charge_corrections

    def __call__(self, mol):
        return self.audible(mol)

    def audible(self, mol):
        start_charge = Chem.GetFormalCharge(mol)
        for cc in self.charge_corrections:
            for match in mol.GetSubstructMatches(cc.smarts):
                atom = mol.GetAtomWithIdx(match[0])
                atom.SetFormalCharge(cc.charge)

        current_charge = Chem.GetFormalCharge(mol)
        charge_diff = Chem.GetFormalCharge(mol) - start_charge
        if not current_charge == 0:
            max_iteration = 50
            while charge_diff > 0 and max_iteration:
                ppos, poccur = self._strongest_protonated(mol)
                if ppos is None:
                    break
                patom = mol.GetAtomWithIdx(poccur[-1])
                patom.SetFormalCharge(patom.GetFormalCharge() - 1)
                if patom.GetNumExplicitHs() > 0:
                    patom.SetNumExplicitHs(patom.GetNumExplicitHs() - 1)
                patom.UpdatePropertyCache()
                charge_diff -= 1
                max_iteration -= 1

        already_moved = set()
        while True:
            ppos, poccur = self._strongest_protonated(mol)
            ipos, ioccur = self._weakest_ionized(mol)
            if ioccur and poccur and ppos < ipos:
                if poccur[-1] == ioccur[-1]:
                    break

                key = tuple(sorted([poccur[-1], ioccur[-1]]))
                if key in already_moved:
                    break
                already_moved.add(key)
                patom = mol.GetAtomWithIdx(poccur[-1])
                patom.SetFormalCharge(patom.GetFormalCharge() - 1)
                if patom.GetNumImplicitHs() == 0 and patom.GetNumExplicitHs() > 0:
                    patom.SetNumExplicitHs(patom.GetNumExplicitHs() - 1)
                patom.UpdatePropertyCache()
                iatom = mol.GetAtomWithIdx(ioccur[-1])
                iatom.SetFormalCharge(iatom.GetFormalCharge() + 1)
                if (iatom.GetNoImplicit() or
                        ((patom.GetAtomicNum() == 7 or patom.GetAtomicNum() == 15) and patom.GetIsAromatic()) or
                        iatom.GetTotalValence() not in list(Chem.GetPeriodicTable().GetValenceList(iatom.GetAtomicNum()))):
                    iatom.SetNumExplicitHs(iatom.GetNumExplicitHs() + 1)
                iatom.UpdatePropertyCache()
            else:
                break
        Chem.SanitizeMol(mol)
        return mol

    def _strongest_protonated(self, mol):
        for position, pair in enumerate(self.Property8):
            for occurrence in mol.GetSubstructMatches(pair.acid):
                return position, occurrence
        return None, None

    def _weakest_ionized(self, mol):
        for position, pair in enumerate(reversed(self.Property8)):
            for occurrence in mol.GetSubstructMatches(pair.base):
                return len(self.Property8) - position - 1, occurrence
        return None, None

class Function9(object):
    def __init__(self):
        self._metal_nof = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,'+
            'Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Sb]~[N,O,F,B,Si,P,As,Sb,S,'+
            'Se,Te,Cl,Br,I,At,Al]'
            )
        self._metal_non = Chem.MolFromSmarts(
                    '[Al,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,Hf,Ta,W,Re,'+
                    'Os,Ir,Pt,Au,Sb]~[B,C,Si,P,As,Sb,S,Se,Te,Cl,Br,I,At]')

        self._metal_nitril = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,'+
            'Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Sb]~[$(C#N)]')

    def __call__(self, mol):
        return self.disrelate(mol)

    def disrelate(self, mol):
        for smarts in [self._metal_nof, self._metal_non,self._metal_nitril]:
            pairs = mol.GetSubstructMatches(smarts)
            rwmol = Chem.RWMol(mol)
            orders = []
            for i, j in pairs:
                orders.append(int(mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()))
                rwmol.RemoveBond(i, j)
            mol = rwmol.GetMol()
            for n, (i, j) in enumerate(pairs):
                chg = orders[n]
                atom1 = mol.GetAtomWithIdx(i)
                atom1.SetFormalCharge(atom1.GetFormalCharge() + chg)
                atom2 = mol.GetAtomWithIdx(j)
                atom2.SetFormalCharge(atom2.GetFormalCharge() - chg)
        Chem.SanitizeMol(mol)
        return mol


class Function10(object):
    def __init__(self):
        self._metal_cp2 = Chem.MolFromSmarts(
            '[Fe,Co,Ni,Zr;!+0].[C,c;!+0]'
            )

        self._metal_nof = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,'+
            'Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Sb,'+
            'La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu;!+0].[N,O!$(*[N,I,Cl,Br;+,+2,+3]),F,C,c,Si,P,As,Sb,S,'+
            'Se,Te,Cl,Br,I,At;!+0]'
            )
        self._metal_non = Chem.MolFromSmarts(
                    '[Al,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,Hf,Ta,W,Re,'+
                    'Os,Ir,Pt,Au,Sb,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu;!+0].[B!v4,C,'+
                    'Si!v6,P,As,Sb,S,Se,Te,Cl,Br,I,At;!+0]')

        self._metal_h = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,'+
            'Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Sb,La,Ce,Pr,Nd,Pm,Sm,Eu,'+
            'Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu;!+0].[H-]'
            )
        self._metal_no3 = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,'+
            'Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Sb,La,Ce,Pr,Nd,Pm,Sm,Eu,'+
            'Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu;!+0].[O-$(*N(=O)[O-])]'
            )

        self._metal_xo3 = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,'+
            'Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Sb,La,Ce,Pr,Nd,Pm,Sm,Eu,'+
            'Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu;!+0].[O-$(*[I,Cl,Br;+][O-]),O-$(*[I,Cl,Br;+2]([O-])[O-]),O-$(*[I,Cl,Br;+3]([O-])([O-])[O-])]'
            )

        self._bond_type = {1:Chem.rdchem.BondType.SINGLE, 2:Chem.rdchem.BondType.DOUBLE, 
                            3:Chem.rdchem.BondType.TRIPLE, 4:Chem.rdchem.BondType.UNSPECIFIED}

    def __call__(self, mol):
        return self.relate(mol)

    def relate(self, mol):
        for smarts_id, smarts in enumerate([self._metal_cp2, self._metal_nof, self._metal_non, self._metal_h, 
                                            self._metal_no3, self._metal_xo3]):
            has_pairs = True
            while has_pairs:
                has_pairs = False
                pairs = mol.GetSubstructMatches(smarts)
                rwmol = Chem.RWMol(mol)
                for i, j in pairs:
                    charge_i, charge_j = mol.GetAtomWithIdx(i).GetFormalCharge(), mol.GetAtomWithIdx(j).GetFormalCharge()
                    if charge_i*charge_j >= 0 or charge_i<0:
                        continue
                    has_pairs = True
                    order = min([abs(charge_i), abs(charge_j)])
                    rwmol.AddBond(i, j, self._bond_type[order])
                    mol = rwmol.GetMol()
                    if charge_i<0:
                        mol.GetAtomWithIdx(i).SetFormalCharge(charge_i+order)
                        mol.GetAtomWithIdx(j).SetFormalCharge(charge_j-order)
                    else:
                        mol.GetAtomWithIdx(i).SetFormalCharge(charge_i-order)
                        mol.GetAtomWithIdx(j).SetFormalCharge(charge_j+order)
                    Chem.SanitizeMol(mol)
                    break
        return mol


class Function11(object):
    def __init__(self):
        self._metal_radical = Chem.MolFromSmarts(
            '[Be,Mg,Ca,Sr,Ba,Ra,Sn,Al,Zn,Cd,Hg,Pd,Pt].[C!X4,c!X3,#7;!r5]'
            )
        self._metal_h = Chem.MolFromSmarts(
            '[Be,Mg,Ca,Sr,Ba,Ra,Sn,Al,Zn,Cd,Hg].[H+0]'
            )
        self._li_radical = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs;+0].[C!X4,c!X3,#7]'
            )
        self._li_h = Chem.MolFromSmarts(
            '[Li,Na,K,Rb,Cs;+0].[H+0]'
            )
        self._cp_metals = ['Fe','Co','Ni','Zr']

    def __call__(self, mol):
        return self.relate(mol)

    def displayed(self, mol):
        Chem.SanitizeMol(mol)
        smarts = '[{}][P$(*([cr6])[cr6])]'
        for symbol in ['Pd','Rh','Ni']:
            metal_p = Chem.MolFromSmarts(smarts.format(symbol))
            hasPattern = mol.GetSubstructMatches(metal_p)
            while hasPattern:
                rwmol = Chem.RWMol(mol)
                rwmol.RemoveBond(*hasPattern[0])
                mol = rwmol.GetMol()
                Chem.SanitizeMol(mol)
                hasPattern = mol.GetSubstructMatches(metal_p)

        smarts = '[{}v0].[C]1~[C]~[C]~[C]~[C]~1.[C]1~[C]~[C]~[C]~[C]~1'
        for symbol in self._cp_metals:
            metal_cp = Chem.MolFromSmarts(smarts.format(symbol))
            hasPattern = mol.GetSubstructMatches(metal_cp)
            while hasPattern:
                matched_indices = hasPattern[0]
                rwmol = Chem.RWMol(mol)
                for i,j in [(1,2),(2,3),(3,4),(4,5),(5,1),(6,7),(7,8),(8,9),(9,10),(10,6)]:
                    rwmol.ReplaceBond(mol.GetBondBetweenAtoms(hasPattern[0][i],hasPattern[0][j]).GetIdx(), Chem.BondFromSmarts('-'))
                for i in matched_indices[1:]:
                    rwmol.AddBond(matched_indices[0], i, Function10()._bond_type[1])
                mol = rwmol.GetMol()
                Chem.SanitizeMol(mol)
                hasPattern = mol.GetSubstructMatches(metal_cp)
        def displayed2(mol):
            Chem.SanitizeMol(mol)
            for symbol in self._cp_metals:
                non_radical = Chem.MolFromSmarts('[C]1~[C]~[C]([{}D2][C]2(~[C]~[C]~[C]~[C]~2))~[C]~[C]~1'.format(symbol))
                hasPattern = mol.GetSubstructMatches(non_radical)
                while hasPattern:
                    matched_idices = hasPattern[0]
                    rwmol = Chem.RWMol(mol)
                    for pair in [(0,1),(1,2),(4,5),(5,6),(6,7),(7,8),(8,4),(9,10),(2,9),(0,10)]:
                        if mol.GetBondBetweenAtoms(matched_idices[pair[0]],matched_idices[pair[1]]).GetBondType() != Function10()._bond_type[1]:
                            rwmol.RemoveBond(matched_idices[pair[0]],matched_idices[pair[1]])
                            rwmol.AddBond(matched_idices[pair[0]],matched_idices[pair[1]], Function10()._bond_type[1])
                    for i in matched_idices[:3]+matched_idices[4:]:
                        if not mol.GetBondBetweenAtoms(matched_idices[3], i):
                            rwmol.AddBond(matched_idices[3], i, Function10()._bond_type[1])
                        if mol.GetAtomWithIdx(i).GetTotalNumHs()==2:
                            rwmol.ReplaceAtom(i,Chem.AtomFromSmiles('C'))
                    mol = rwmol.GetMol()
                    Chem.SanitizeMol(mol)
                    hasPattern = mol.GetSubstructMatches(non_radical)
                radical = Chem.MolFromSmarts('[C;r3!X4][{}]'.format(symbol))
                hasPattern = mol.GetSubstructMatches(radical)
                max_iteration = 50
                while hasPattern and max_iteration:
                    max_iteration -= 1
                    rwmol = Chem.RWMol(mol)
                    rwmol.ReplaceAtom(hasPattern[0][0],Chem.AtomFromSmiles('C'))
                    mol = rwmol.GetMol()
                    Chem.SanitizeMol(mol)
                    hasPattern = mol.GetSubstructMatches(radical)
            Chem.SanitizeMol(mol)
            self._metal_radical2 = Chem.MolFromSmarts(
                '[Be,Mg,Ca,Sr,Ba,Ra,Sn,Al,Zn,Cd,Hg,Pd,Pt,Li,Na,K,Rb,Cs].[C!X4,c!X3,#7]'
                )
            for smarts in [self._metal_radical2,]:
                pairs = mol.GetSubstructMatches(smarts)
                for i, j in pairs:
                    atom_i = mol.GetAtomWithIdx(i)
                    atom_j = mol.GetAtomWithIdx(j)
                    if not atom_i.GetNumRadicalElectrons():
                        continue
                    if not atom_j.GetNumRadicalElectrons():
                        continue
                    mol.GetAtomWithIdx(i).SetNumRadicalElectrons(0)
                    mol.GetAtomWithIdx(j).SetNumRadicalElectrons(0)
                    if not mol.GetBondBetweenAtoms(i, j):
                        rwmol = Chem.RWMol(mol)
                        rwmol.AddBond(i, j, Function10()._bond_type[1])
                        mol = rwmol.GetMol()
                    Chem.SanitizeMol(mol)
            return mol
        return displayed2(mol)

    def relate(self, mol):
        smarts = '[{}X1;!-]C=C'
        for symbol in ['O',]:
            occ = Chem.MolFromSmarts(smarts.format(symbol))
            hasPattern = mol.GetSubstructMatches(occ)
            while hasPattern:
                rwmol = Chem.RWMol(mol)
                rwmol.ReplaceAtom(hasPattern[0][0], Chem.AtomFromSmiles('O'))
                num_hs = mol.GetAtomWithIdx(hasPattern[0][2]).GetTotalNumHs()
                rwmol.ReplaceAtom(hasPattern[0][2], Chem.AtomFromSmarts('[CH%d]'%num_hs))
                rwmol.ReplaceBond(mol.GetBondBetweenAtoms(hasPattern[0][0],hasPattern[0][1]).GetIdx(), Chem.BondFromSmarts('='))
                rwmol.ReplaceBond(mol.GetBondBetweenAtoms(hasPattern[0][1],hasPattern[0][2]).GetIdx(), Chem.BondFromSmarts('-'))
                mol = rwmol.GetMol()
                Chem.SanitizeMol(mol)
                hasPattern = mol.GetSubstructMatches(occ)

        for smarts in [self._metal_radical,self._metal_h,self._li_radical,self._li_h]:
            pairs = mol.GetSubstructMatches(smarts)
            for i, j in pairs:
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                if not atom_i.GetNumRadicalElectrons():
                    if atom_i.GetSymbol() not in ['Pd','Pt']:
                        continue
                if not atom_j.GetNumRadicalElectrons():
                    continue
                mol.GetAtomWithIdx(i).SetNumRadicalElectrons(0)
                mol.GetAtomWithIdx(j).SetNumRadicalElectrons(0)
                if not mol.GetBondBetweenAtoms(i, j):
                    rwmol = Chem.RWMol(mol)
                    rwmol.AddBond(i, j, Function10()._bond_type[1])
                    mol = rwmol.GetMol()
                Chem.SanitizeMol(mol)
        sn_count = 0
        radical_count = 0
        sn_indices = {}
        for a in mol.GetAtoms():
            if a.GetSmarts() == '[Sn]':
                sn_count += 1
                ele_num = a.GetNumRadicalElectrons()
                radical_count += ele_num 
                sn_indices[a.GetIdx()] = ele_num
        if sn_count==2 and radical_count==2:
            rxn = AllChem.ReactionFromSmarts('([Snv4:1][*:2].[Snv2:3])>>[Sn:1][Sn:3][*:2]')
            a0_a1_indices = list(sn_indices.keys())
            if mol.GetBondBetweenAtoms(*a0_a1_indices):
                return mol
            rwmol = Chem.RWMol(mol)
            rwmol.AddBond(*a0_a1_indices, Function10()._bond_type[1])
            a0 = mol.GetAtomWithIdx(a0_a1_indices[0])
            a1 = mol.GetAtomWithIdx(a0_a1_indices[1])
            if sn_indices[a0_a1_indices[0]] == 1:
                pass
            elif sn_indices[a0_a1_indices[0]] == 0:
                rwmol.RemoveBond(a0_a1_indices[0], a0.GetNeighbors()[0].GetIdx())
                rwmol.AddBond(a0_a1_indices[1], a0.GetNeighbors()[0].GetIdx(), Function10()._bond_type[1])
            else:
                rwmol.RemoveBond(a0_a1_indices[1], a0.GetNeighbors()[1].GetIdx())
                rwmol.AddBond(a0_a1_indices[0], a0.GetNeighbors()[1].GetIdx(), Function10()._bond_type[1])
            mol = rwmol.GetMol()
        return mol

def Normalize(smi, sep='~', expression=False, dorelate=True, cp2relate=True, kekulize=False, clean_isotops=False):
    smi = smi.replace(sep, '.')
    smi = Function6(smi, clean_isotops)
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        smis = []
        for s in smi.split('.'):
            if '[Si-2]' in s or '[Ge-2]' in s or '[Sn-2]' in s or '[Pb-2]' in s:
                s = s.replace('[Si-2]','[Si+2]')
                s = s.replace('[Sn-2]','[Sn+2]')
                s = s.replace('[Ge-2]','[Ge+2]')
                s = s.replace('[Pb-2]','[Pb+2]')
                sif6 = Chem.MolFromSmiles(s)
                rxn = AllChem.ReactionFromSmarts('[F,Cl,Br,I:1][Si,Ge,Sn,Pb;+2:2][F,Cl,Br,I:3]>>[*+0:2].[*-:1].[*-:3]')
                s = '.'.join([Chem.MolToSmiles(m) for m in rxn.RunReactants((sif6,))[0]])
            smis.append(s)
        smi = '.'.join(smis)
        mol = Chem.MolFromSmiles(smi)
    tau_cano = Function3()
    cano_mol = tau_cano.canonicalize(mol)
    if dorelate:
        audibled = Function8().audible(cano_mol)
        grig_conn = Function11()
        conn_mol = grig_conn(audibled)
        met_conn = Function10()
        conn_mol = met_conn(conn_mol)
        if cp2relate:
            conn_mol = grig_conn.displayed(conn_mol)

        new_smi = Chem.MolToSmiles(conn_mol)
        new_smi = Chem.MolToSmiles(Chem.MolFromSmiles(new_smi))
        new_smi = sep.join(sorted(new_smi.split('.')))
        new_smi = Function5(new_smi, sep, expression).replace(sep,'.')
        audibled = Function7().universial(Chem.MolFromSmiles(new_smi))
        if kekulize:
            smis = sorted(Chem.MolToSmiles(audibled).split('.'))
            kekule_smis = []
            for smi in smis:
                mol = Chem.MolFromSmiles(smi)
                Chem.Kekulize(mol)
                kekule_smis.append(Chem.MolToSmiles(mol, kekuleSmiles=True))
            return sep.join(kekule_smis)
        else:
            return sep.join(sorted(Chem.MolToSmiles(audibled).split('.')))
    else:
        met_dis = Function9()
        dis_mol = met_dis(cano_mol)
        audibled = Function7().universial(dis_mol)
        audibled = Function8().audible(audibled)
    new_smi = Chem.MolToSmiles(audibled)
    new_smi = Chem.MolToSmiles(Chem.MolFromSmiles(new_smi))
    new_smi = sep.join(sorted(new_smi.split('.')))
    if kekulize:
        smis = sorted(Chem.MolToSmiles(audibled).split('.'))
        kekule_smis = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi.replace('~','.'))
            Chem.Kekulize(mol)
            kekule_smis.append(Chem.MolToSmiles(mol, kekuleSmiles=True).replace('.','~'))
        return sep.join(kekule_smis)
    else:
        return Function5(new_smi, sep, expression)

if __name__ == '__main__':
    print(Function4()(Chem.MolFromSmiles('CN=C(O)C(C)C(C)C(O)=NC')))
    original_smi = 'CC(=N)O.CC[Mg]Br.CCCn1cc[n+](C)c1.[Na+]~[OH-].[Na][H]'
    new_smi = '.'.join([Normalize(s) for s in original_smi.split('.')])
    print(new_smi) #'CC(N)=O.CC[Mg]Br.CCCn1cc[n+](C)c1.O[Na].[NaH]'
