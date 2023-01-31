from GRASP.search.search_tree import MCTS_inference
from GRASP.models.train import train, test
from GRASP.models.base_agent import GRASP_Agent
from GRASP.utils.misc import canonicalize, remove_atom_mapping
from GRASP.opt import rl_opt

from GRASP.single_step.tc_check import tc_api
from GRASP.materials.material_service import material_api
from GRASP.models.env import mini_retro_env

if __name__ == '__main__':
    
    agent = GRASP_Agent(rl_opt)
    agent.cuda_convert()
    train(mini_retro_env, agent, 1000, './model_ckpt')
