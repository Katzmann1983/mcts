"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(
            int
        )  # Number of wins of False, especially self.N if false wins, 0 if true wins
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # node -> children of the node
        self.exploration_weight = exploration_weight
        self.terminal = (
            {}
        )  # If terminal node = Q/N, else min/max depending on active player

    def score(self, n):
        try:
            return self.terminal[n]
        except KeyError:
            if self.N[n] == 0:
                # avoid unseen moves
                return float("inf") if n.turn else float("-inf")
            return self.Q[n] / self.N[n]

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        if node.turn:
            return min(self.children[node], key=self.score)
        else:
            return max(self.children[node], key=self.score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path, dead_end = self._select(node)
        leaf = path[-1]
        if not dead_end:
            self._expand(leaf)
            reward = self._simulate(leaf)
        else:
            rewards = [self.score(child) for child in self.children[leaf]]
            if leaf.turn:
                reward = min(rewards)
            else:
                reward = max(rewards)
            assert reward in (0, 0.5, 1)
            best_choice = self.choose(leaf)
            assert self.score(best_choice) == reward
            self.terminal[leaf] = reward
            if leaf.tup == (
                True,
                False,
                False,
                True,
                True,
                None,
                False,
                True,
                False,
            ):
                print("0")
            # assert self.Q[best_choice] / self.N[best_choice] == reward
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path, False
            unexplored = [node for node in self.children[node] if self.N[node] == 0]
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path, False
            # TODO Fix that the two different terminals are the same, e.g. when node = board.terminal = False
            interesting = [
                n
                for n in self.children[node]
                if (n not in self.terminal) and (not n.is_terminal()) and self.N[n] > 0
            ]
            if len(interesting) == 0:
                return path, True
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded

        self.children[node] = node.find_children()

        # Check if any of the cild nodes represent previously visited states:
        if len(self.children[node]) > 0:
            if all([child.is_terminal() for child in self.children[node]]):
                if all([self.Q[child] > 0 for child in self.children[node]]):
                    # Remove child node from the list of children?
                    self.terminal[node] = max(
                        [self.Q[child] / self.N[child] for child in self.children[node]]
                    )

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                self.terminal[node] = 1 - reward
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        # Only non-terminals, not visited:
        interesting = [
            n
            for n in self.children[node]
            if (n not in self.terminal) and (not n.is_terminal()) and self.N[n] > 0
        ]

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(interesting, key=uct)

    def __str__(self) -> str:
        rval = ""
        rval += f"total reward of each node: {str(self.Q)}\n"
        rval += f"total visit count for each node {str(self.N)}\n"
        rval += f"children of each node: {str(self.children)}\n"
        rval += f"exploration_weight: {str(self.exploration_weight)}\n"
        return rval


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
