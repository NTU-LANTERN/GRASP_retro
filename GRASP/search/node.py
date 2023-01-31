# encoding=utf8

import random
import numpy as np

from GRASP.utils.misc import smiles_to_fp


class ReactionNode(object):
    def __init__(self, parent, reaction, prior_p, mt_score, c_puct=3):
        self.parent = parent
        self.children = []
        self.target = self.parent.target
        self.reaction = reaction
        self.depth = self.parent.depth + 1
        self.mol_solved_reaction = {}
        self.path = set()
        self.mt_score = mt_score
        self.n_visits = 0
        self.Q = 0
        self.prior = prior_p
        self.is_leaf = False
        self.is_dead = False
        self.is_solved = False
        self.is_material = False
        self.c_puct = c_puct
    
    def select(self):
        """
            Select, if all solved then select a non-material Mol
        """
        candidate = [m for m in self.children if not m.is_solved]
        if candidate:
            mol = random.choice(candidate)
            # todo check is it a better strategy
            # mol = max(candidate, key=lambda c: c.get_value())
        else:
            candidate = [m for m in self.children if not m.is_material]
            mol = random.choice(candidate)
        return mol
    
    def update(self, value):
        """
            Update with Q value
        """
        self.n_visits += 1
        self.is_solved = all([m.get_solved() for m in self.children])
        self.is_dead = any([m.get_dead() for m in self.children])
        self.is_material = all([m.get_material() for m in self.children])
        self.Q += 1.0 * (value - self.Q) / self.n_visits
        return
    
    def update_recursive(self, value):
        self.update(value)
        self.parent.update_recursive(value)
        return
    
    def get_value(self):
        u = (self.c_puct * self.prior * np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + u
    
    def get_min_qs(self):
        return self.children
    
    def get_solved(self):
        return self.is_solved
    
    def get_dead(self):
        return self.is_dead
    
    def is_root(self):
        return self.parent is None
    
    def __repr__(self):
        return self.reaction


class MolNode(object):
    def __init__(self, target, parent, prior_p, q, ancestor, is_solved=False, is_material=False, is_dead=False,
                 c_puct=3):
        self.target = target
        self.parent = parent
        if parent:
            self.reaction = parent.reaction
        self.children = []
        self.children_template = {}
        self.n_visits = 0
        self.Q = q
        self.u = 0
        self.prior = prior_p
        if parent:
            self.depth = self.parent.depth
        else:
            self.depth = 1
        self.is_leaf = True
        self.is_dead = is_dead
        self.is_solved = is_solved
        self.is_material = is_material
        self.c_puct = c_puct
        self.ancestor = ancestor
        self.path = set()
    
    def expand_and_eval(self, reactions, mat_api, agent_api=None, goal=None, c_puct=3, max_depth=3):
        """
        Expand and evaluate
        """
        add_nodes = 0 
        if reactions:
            # Regular Expansion
            for reaction in reactions:
                reactants = reaction["reactants"]
                reaction_str = reaction['reaction']
                reaction_score = reaction['mt_score']
                reaction_prop = reaction_score
                if self.ancestor.intersection(set(reactants)):
                    continue
                if reaction_str not in self.children:
                    reaction_node = ReactionNode(self, reaction_str, reaction_prop, reaction_score, c_puct=c_puct)
                    for mol in reactants:
                        current_ancestor = self.ancestor.union({mol})
                        if any(mat_api.find([mol])):
                            reaction_node.children.append(
                                MolNode(mol, reaction_node, reaction_node.prior, 1, current_ancestor, is_solved=True,
                                        is_material=True,
                                        c_puct=c_puct))
                        else:
                            if self.depth < max_depth:
                                if agent_api:
                                    _Q = agent_api.get_Q(smiles_to_fp(self.target), smiles_to_fp(mol), goal)
                                else:
                                    _Q = 0
                                reaction_node.children.append(
                                    MolNode(mol, reaction_node, reaction_node.prior, _Q, current_ancestor,
                                            c_puct=c_puct))
                            else:
                                reaction_node.children.append(
                                    MolNode(mol, reaction_node, reaction_node.prior, 0, current_ancestor,
                                            c_puct=c_puct, is_dead=True))
                    # Update ReactionNode Q with children for expand
                    reaction_node.update(np.min([m.get_Q() for m in reaction_node.children]))
                    self.children.append(reaction_node)
                    add_nodes += len(reaction_node.children)
        
        if self.children:
            value = max([r.get_value() for r in self.children])
        else:
            # No Expansion
            self.is_dead = True
            value = -1
        
        self.is_leaf = False
        return value, add_nodes
    
    def select(self):
        """
            Select a non-dead, non-solved reaction.
        """
        candidate = [r for r in self.children if r.is_dead is False and r.is_material is False and r.is_solved is False]
        if not candidate:
            candidate = [r for r in self.children if r.is_dead is False and r.is_material is False]
            if not candidate:
                return None
        re = max(candidate, key=lambda x: x.get_value())
        return re
    
    def update_recursive(self, value):
        """
            Update and-or with Q recursively
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (value - self.Q) / self.n_visits
        if self.children:
            self.is_dead = all([r.is_dead for r in self.children])
            self.is_solved = any([r.is_solved for r in self.children])
            self.is_material = all([r.is_material for r in self.children if not r.is_dead])
        if self.is_solved and self.children:
            temp_path = set()
            for r in self.children:
                if r.path:
                    temp_path = temp_path.union(r.path)
            self.path = temp_path
        if self.parent:
            self.parent.update_recursive(value)
        else:
            return
    
    def available(self, topk, tc_api):
        try:
            reactions = tc_api.single_exp_check(target=self.target)
        except Exception as e:
            print('Error in predicting %s' % self.target, e)
            reactions = []
        return reactions[:topk]
    
    def get_value(self):
        if self.parent:
            self.u = (self.c_puct * self.prior *
                      np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        else:
            self.u = (self.c_puct * self.prior *
                      np.sqrt(self.n_visits) / (1 + self.n_visits))
        return self.Q + self.u
    
    def get_Q(self):
        return self.Q
    
    def get_leaf(self):
        return self.is_leaf
    
    def get_solved(self):
        return self.is_solved
    
    def get_dead(self):
        return self.is_dead
    
    def get_material(self):
        return self.is_material
    
    def __repr__(self):
        return self.target
