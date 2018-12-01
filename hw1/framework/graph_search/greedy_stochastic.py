from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np


class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        if self.close.has_state(successor_node.state):
            return
        if not self.open.has_state(successor_node.state):
            self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        h = self.heuristic_function.estimate
        return h(search_node.state)

    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        """
        # Pop the N-most prioritised nodes from self.open
        nodes_options = []
        popped_nodes = 0
        while popped_nodes < self.N:
            if self.open.is_empty():
                break
            node = self.open.pop_next_node()
            popped_nodes += 1
            nodes_options.append(node)
        if not nodes_options:
            return
        # Choose one randomly and put the un-chosen nodes back in self.open
        chosen_node = self._choose_node_randomly(nodes_options)
        for n in nodes_options:
            if n == chosen_node:
                continue
            self.open.push_node(n)
        # Scale T value for next time
        self.T *= self.T_scale_factor
        # We're done
        return chosen_node

    def _choose_node_randomly(self, nodes_options):
        # A priority of zero will have probability of 1, so just choose it now
        zero_priority_nodes = [ n for n in nodes_options if n.expanding_priority == 0 ]
        if (zero_priority_nodes):
            return zero_priority_nodes[0]
        # Normalize priorities, to prevent numbers exploding to inf or zero
        priorities = np.array([ n.expanding_priority for n in nodes_options ])
        alpha = np.min(priorities) 
        normalized_priorities = priorities / alpha
        # Calculate probabilies vector
        priorities_exp = np.power(normalized_priorities, -1/self.T)
        priorities_sum = priorities_exp.sum()
        P = priorities_exp / priorities_sum
        # Choose a node randomly based on calculated priorities
        chosen_node = np.random.choice(nodes_options, p=P)
        return chosen_node
