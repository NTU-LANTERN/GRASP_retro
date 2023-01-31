
from GRASP.models.base_agent import GRASP_Agent
from GRASP.opt import rl_opt
from GRASP.models.utils import to_tensor, to_numpy


MODEL_CKPT = './model_ckpt/'

class Agent_api():

    def __init__(self, opt, debug=True):
        self.debug = debug
        self.agent = GRASP_Agent(opt)
        self.agent.load_weights(MODEL_CKPT+'actor.pt', MODEL_CKPT+'critic.pt')
        self.agent.cuda_convert()
        self.agent.is_training = False
        self.agent.eval()

    def get_Q(self, s, a, g):
        s = to_tensor(s, device=self.agent.device)
        a = to_tensor(a, device=self.agent.device)
        g = to_tensor(g, device=self.agent.device)
        Q = to_numpy(self.agent.critic([s,a,g])).flatten()[0]
        return Q

    def get_proto_a(self, s, g):
        s = to_tensor(s, device=self.agent.device)
        g = to_tensor(g, device=self.agent.device)
        a = to_numpy(self.agent.actor([s,g])).flatten()
        return a

agent_api = Agent_api(rl_opt, debug=True)