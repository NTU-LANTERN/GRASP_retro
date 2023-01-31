

import pickle as pkl
import random 
import numpy as np
from loguru import logger

from GRASP.opt import env_opt
from GRASP.utils.misc import canonicalize, smiles_to_fp, batch_smiles_to_fp, similarity
from GRASP.single_step.tc_check import tc_api
from GRASP.materials.material_service import material_api


# tc_api.load_tc_cache(file='./tc_cache_cut_off_' + str(0.6) + '.pkl')

class MiniRetroEnv():
    
    def __init__(self, opt):
        self.opt = opt
        self._load_start_mol()
        self.cur_state = (None, None)

    def step(self, action):
        self.cur_state = (action['_ns_mol'], action['action'])
        if not [x for x in action['reactants'] if not any(material_api.find([x]))]:
            return self.cur_state, 1, True, self.get_dummy_action_space(self.cur_state)
        else:
            _next_states = [x for x in action['reactants'] if not any(material_api.find([x]))]
            # if for what? why single?
            if all([tc_api.single_exp_check(x) for x in _next_states]):
                return self.cur_state, 0, False, self.get_valid_actions(self.cur_state)
            else:
                return self.cur_state, 0, True, self.get_dummy_action_space(self.cur_state)

    def reset(self):
        _mol = random.choice(self.start_mol)
        self.cur_state = (_mol, smiles_to_fp(_mol))
        return self.cur_state

    def _load_start_mol(self):
        with open(self.opt.train_file, 'rb') as f:
            logger.info("Loading target molecules for training")
            self.start_mol = [canonicalize(x) for x in set(pkl.load(f)) if not any(material_api.find([canonicalize(x)]))]
            logger.info("Env loaded %d target molecules for training"%len(self.start_mol))
    
    def get_valid_actions(self, state):
        valid_reactions = tc_api.single_exp_check(state[0])
        if not valid_reactions:
            return []
        parsed_valid_reactions = []
        for r in valid_reactions:
            _actions = [x for x in r['reactants'] if not any(material_api.find([x]))]
            if not _actions:
                r['_ns_mol'] = r['reactants'][np.argmax(similarity(x, state[1], metric='Jaccard') for x in r['reactants'])]
                _actions_emb = smiles_to_fp(r['_ns_mol']).astype(float)
            else:
                # _actions_emb = np.average(batch_smiles_to_fp(_actions))
                _actions_emb = np.logical_or.reduce(batch_smiles_to_fp(_actions)).astype(float)
                r['_ns_mol'] = _actions[np.argmax(similarity(x, _actions_emb, 'Jaccard') for x in _actions)]
            r['action'] = _actions_emb
            parsed_valid_reactions.append(r)
        return parsed_valid_reactions

    def get_dummy_action_space(self, state):
        # dummy placeholder for teminate state future action_space, will not count towards training
        return [{
            '_ns_mol' : state[0],
            'action' : state[1]
        }]

mini_retro_env = MiniRetroEnv(env_opt)