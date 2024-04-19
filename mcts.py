'''
Implementation of Monte Carlo Tree Search (MCTS) for self-play
'''
from typing import Any
import threading
import queue
import torch
from collections import defaultdict
import copy
from abc import abstractmethod, ABCMeta

class AbstractNode():
    @abstractmethod
    def is_over(self) -> bool:
        return

    @abstractmethod
    def select_rollout_action(self, c: float) -> torch.Tensor:
        return
    
    @abstractmethod
    def execute_action(self, action) -> "AbstractNode":
        return
    
    @abstractmethod
    def evaluate(self, agent) -> float:
        return
    
    @abstractmethod
    def backup(self, v, root) -> None:
        return
    
    @abstractmethod
    def get_search_vector(self, temp) -> torch.Tensor:
        return

class MCTS():
    def __init__(self, agent: Any, c_puct: float, temp: float, n_rollouts: int) -> None:
        self.agent = agent
        self.c_puct = c_puct
        self.temperature = temp
        self.n_rollouts = n_rollouts

    def run_tree_search(self, root: AbstractNode, game_state_counts: defaultdict) -> torch.Tensor:
        '''Performs Monte Carlo Tree Search and returns search vector'''
        # TODO: multithreaded search
        
        # At current node, build vector \pi
        for _ in range(self.n_rollouts):
            curr_inner = root
            gsc = copy.deepcopy(game_state_counts)
            while not curr_inner.is_over():
                # Choose action via PUCT algorithm
                a = curr_inner.select_rollout_action(self.c_puct)

                # Expand and evaluate
                child = curr_inner.execute_action(a, gsc)
                v = child.evaluate(self.agent)
                
                # Backup
                child.backup(v, root)
                curr_inner = child

        # Return search vector \pi
        return root.get_search_vector(self.temperature)
