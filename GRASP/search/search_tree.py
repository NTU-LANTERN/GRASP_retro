# encoding=utf8

import itertools
from collections import defaultdict
from queue import Queue

import numpy as np


from GRASP.search.node import MolNode
from GRASP.search.node import ReactionNode


class MCTS_inference(object):
    def __init__(self, target, tc_api, mat_api, agent_api=None, depth=6, c_puct=1, n_playout=1000, output="out.txt", head=5, n_expand=50):
        self.root = MolNode(target, None, None, 0.0, {target}, is_solved=mat_api.find(target), is_material=mat_api.find(target))
        self.tc_api = tc_api
        self.mat_api = mat_api
        self.agent_api = agent_api
        self.L_max = depth
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.node_num = 1
        self.temp_node_num = 0
        self.step_loss = 0.05
        self.output = output
        self.top_n = head
        self.n_expand = n_expand
        self.visited = set(target)
        self.n_play = 0
    
    def _playout(self):
        node = self.root
        # Selection
        while node and not node.is_leaf:
            node = self.select(node)
            
        if node:
            # Expansion 
            if node.depth <= self.L_max:
                leaf_value, add_nodes = self.expand(node, self.L_max)
                self.node_num += add_nodes
            else:
                leaf_value = node.get_value()
            
            # Update value and visit count of nodes in this traversal.
            node.update_recursive(leaf_value)
        else:
            print("Finished searching all valid reactions in search space...")
            return True

        return False
    
    @staticmethod
    def select(node):
        """
            Recursive select to a leaf MolNode
        """
        curr_node = node
        return curr_node.select()
    
    def expand(self, node, max_depth):
        """
            MolNode expansion
        """
        value, add_nodes = -1, 0
        if node.depth <= self.L_max:
            if node.target not in self.visited:
                self.visited.add(node.target)
                self.n_play += 1
            reactions = node.available(self.n_expand, self.tc_api)
            value, add_nodes = node.expand_and_eval(reactions, self.mat_api, self.agent_api, max_depth)
        return value, add_nodes
    
    def run(self, log_interval=10):
        """
            Run playout until max iteration
        """
        result = None
        finish_flag = False
        while self.n_play < self.n_playout:
            if self.root.is_dead or finish_flag:
                print("Finished searching all valid reactions in search space...")
                break
            finish_flag = self._playout()
            if self.n_play % log_interval == log_interval - 1:
                print("====================================")
                print('Playout iteration: {:d}'.format(self.n_play + 1))
                self.n_play += 1
                # print('Total nodes: {:d}'.format(self.node_num))
                if self.root.get_solved():
                    print('Successfully find the solution!')
                    result = self.find_solution(self.root, write_file=True)
        print("====================================")
        if self.n_play >= self.n_playout:
            print("Reached maximum iteration...")
        if self.root.get_solved():
            print('Successfully find the solution!')
            result = self.find_solution(self.root, write_file=True)
        else:
            print("Failed to find solution!")
        return result
    
    def find_solution(self, node, write_file=True):
        """
        Find all solution and output file
        """
        final_solution = {}
        res, score, layer_dicts = self.find_solution_iterate(node)
        for i, (r, s, d) in enumerate(zip(res.split("$$"), score.split("$$"), layer_dicts)):
            final_solution[i] = (r, s, d)
        final_solution = sorted(final_solution.values(), key=lambda x: x[1], reverse=True)
        if 0 < self.top_n <= len(final_solution):
            if write_file:
                with open("{}".format(self.output), "w") as f:
                    for i, p in enumerate(final_solution[:self.top_n]):
                        print("The %d route : %s with score %s" % (i + 1, self.export_to_file(p[2]), p[1]))
                        f.write(self.export_to_file(p[2]) + "\n")
            else:
                for i, p in enumerate(final_solution[:self.top_n]):
                    print("The %d route : %s with score %s" % (i + 1, self.export_to_file(p[2]), p[1]))
        else:
            if write_file:
                with open("{}".format(self.output), "w") as f:
                    for i, p in enumerate(final_solution):
                        print("The %d route : %s with score %s" % (i + 1, self.export_to_file(p[2]), p[1]))
                        f.write(self.export_to_file(p[2]) + "\n")
            else:
                for i, p in enumerate(final_solution):
                    print("The %d route : %s with score %s" % (i + 1, self.export_to_file(p[2]), p[1]))
        return final_solution
    
    def find_solution_iterate(self, node):
        """
        Iteratively search for all success routes
        """
        if not node.children:
            return None, None, None
        elif isinstance(node, ReactionNode):
            all_route = {}
            for i, m in enumerate(node.children):
                (route, score, layer_dict) = self.find_solution_iterate(m)
                if route is not None:
                    all_route[i] = (route, score, layer_dict)
            if all_route:
                all_possible_route = list(itertools.product(*[route[0].split("$$") for route in all_route.values()]))
                all_score = list(itertools.product(*[route[1].split("$$") for route in all_route.values()]))
                all_dict = list(itertools.product(*[route[2] for route in all_route.values()]))
                temp_dict = defaultdict(list)
                temp_dict[node.depth - 1].append((node.reaction, str(node.prior)))
                return ([str(node.reaction + "-->" + "||".join(r)) for r in all_possible_route],
                        [str(np.prod(np.float64(score))) for score in all_score],
                        [self.combine_dict(temp_dict.copy(), d) for d in all_dict])
            else:
                temp_dict = defaultdict(list)
                temp_dict[node.depth - 1].append((node.reaction, str(node.prior)))
                return [str(node.reaction)], [str(node.prior)], [temp_dict.copy()]
        elif isinstance(node, MolNode):
            all_route = []
            all_score = []
            all_dict = []
            for r in node.children:
                if r.is_solved:
                    (route, score, layer_dict) = self.find_solution_iterate(r)
                    all_route.extend(route)
                    all_score.extend(score)
                    all_dict.extend(layer_dict)
            return "$$".join(all_route), "$$".join(all_score), all_dict
    
    @staticmethod
    def export_to_file(d):
        res = []
        for k in sorted(d.keys()):
            res.append("$".join([item[0] for item in d[k]]))
        return "->".join(res)
    
    @staticmethod
    def combine_dict(target_dict, dict_list):
        for d in dict_list:
            if d is not None:
                for k in d.keys():
                    target_dict[k].extend(d[k])
        return target_dict
    
    def __str__(self):
        return "MCTS_inference"
