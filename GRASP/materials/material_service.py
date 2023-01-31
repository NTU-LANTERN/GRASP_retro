import multiprocessing

from rdkit import Chem
from .mol_normalize import Normalize
from tqdm import tqdm
from loguru import logger
from rdkit import RDLogger

from GRASP.opt import mat_opt

def Parallel_(function, inputs, num_cores):
    pool = multiprocessing.Pool(num_cores)
    # pbar = tqdm(total=len(inputs))
    # pbar.set_description("START MULTIPROCESSING")
    all_results = []
    for result in pool.imap_unordered(function, inputs):
        all_results.append(result)
        # pbar.set_description(str(len(all_results)))
        # pbar.update()
    pool.close()
    pool.join()
    return all_results


class StartingMaterialsApi:
    def __init__(self, materials_path):
        with open(materials_path, "r") as f:
            self.materials = [x.replace("\n", "") for x in f.readlines()]
        logger.info("Init material base complete")

    def norm(self, smi):
        # The material files are processed to avoid unrecognizable inchified to canonicalzed materials, espically for MT single-step with reagents
        try:
            mol=Chem.MolFromSmiles(smi)
            Chem.RemoveStereochemistry(mol)
            smi = Chem.MolToSmiles(mol)
        except:
            pass
        smis = smi.split(".")
        flag = True
        for x in smis:
            if x not in self.materials:
                flag = False
                break
        if flag:
            return True
        try:
            flag = True
            for x in Normalize(smi).replace("~",".").split("."):
                if x not in self.materials:
                    flag = False
                    break
            return flag
        except:
            return False


    def find(self, smis:list, use_parallel:bool=False, num_cores=12):
        if use_parallel:
            return Parallel_(self.norm, smis, num_cores)
        else:
            results = []
            for smi in smis:
                results.append(self.norm(smi))
            return results


material_api = StartingMaterialsApi(mat_opt.mat_file)