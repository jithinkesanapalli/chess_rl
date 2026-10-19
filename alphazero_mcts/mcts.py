"""PUCT Monte Carlo Tree Search using the neural policy and value heads."""
import math
import chess
import numpy as np
import torch
from chess_utils import encode_board, legal_priors_from_logits

class Edge:
    def __init__(self, move, prior):
        self.move = move
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.child = None

    @property
    def q(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

class Node:
    def __init__(self, board):
        self.board = board
        self.edges = []
        self.visit_count = 0
        self.expanded = False

class MCTS:
    def __init__(self, model, device="cpu", simulations=64, c_puct=1.5):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct

    @torch.no_grad()
    def _evaluate(self, board):
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        return logits[0], float(value[0].item())

    def _expand(self, node):
        logits, value = self._evaluate(node.board)
        moves, priors = legal_priors_from_logits(logits, node.board)
        node.edges = [Edge(m, p) for m, p in zip(moves, priors)]
        node.expanded = True
        return value

    def _terminal_value(self, board):
        if not board.is_game_over(claim_draw=True):
            return None
        result = board.result(claim_draw=True)
        if result == "1/2-1/2":
            return 0.0
        winner = chess.WHITE if result == "1-0" else chess.BLACK
        return 1.0 if board.turn == winner else -1.0

    def _simulate(self, node):
        terminal = self._terminal_value(node.board)
        if terminal is not None:
            return terminal
        if not node.expanded:
            return self._expand(node)

        sqrt_n = math.sqrt(max(1, node.visit_count))
        best_edge, best_score = None, -float("inf")
        for edge in node.edges:
            u = self.c_puct * edge.prior * sqrt_n / (1 + edge.visit_count)
            score = edge.q + u
            if score > best_score:
                best_edge, best_score = edge, score

        child_board = node.board.copy()
        child_board.push(best_edge.move)
        if best_edge.child is None:
            best_edge.child = Node(child_board)
        value = -self._simulate(best_edge.child)
        best_edge.visit_count += 1
        best_edge.value_sum += value
        node.visit_count += 1
        return value

    def search(self, board, add_root_noise=False):
        root = Node(board.copy())
        if self._terminal_value(root.board) is not None:
            return root
        self._expand(root)
        if add_root_noise and root.edges:
            noise = np.random.dirichlet([0.3] * len(root.edges))
            for edge, n in zip(root.edges, noise):
                edge.prior = 0.75 * edge.prior + 0.25 * float(n)
        for _ in range(self.simulations):
            self._simulate(root)
        return root

    @staticmethod
    def visit_policy(root, temperature=1.0):
        target = np.zeros(4096, dtype=np.float32)
        edges = root.edges
        if not edges:
            return target
        counts = np.array([e.visit_count for e in edges], dtype=np.float64)
        if temperature <= 1e-6:
            probs = np.zeros_like(counts)
            probs[int(np.argmax(counts))] = 1.0
        elif counts.sum() == 0:
            priors = np.array([e.prior for e in edges])
            probs = priors / priors.sum()
        else:
            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()
        for edge, p in zip(edges, probs):
            target[edge.move.from_square * 64 + edge.move.to_square] += float(p)
        return target

    @staticmethod
    def select_move(root, temperature=0.0):
        if not root.edges:
            return None
        edges = root.edges
        counts = np.array([e.visit_count for e in edges], dtype=np.float64)
        if temperature <= 1e-6:
            return edges[int(np.argmax(counts))].move
        weights = counts ** (1.0 / temperature)
        if weights.sum() <= 0:
            weights = np.array([e.prior for e in edges], dtype=np.float64)
        weights /= weights.sum()
        return edges[int(np.random.choice(len(edges), p=weights))].move
