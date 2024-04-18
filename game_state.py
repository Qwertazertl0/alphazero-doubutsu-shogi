import torch

TOTAL_NODE_COUNT = 0
TIME_HISTORY = 1

class ShogiNode():
    '''
    Represents game state using binary/count planes as described in original AlphaZero paper.
    Doubutsu Shogi uses a 4 x 3 board.
    '''
    def __init__(self, state=None, actions=None) -> None:
        self.children = []
        self.edges = torch.zeros(size=(4, 12, 4, 3)) # (N,W,Q,P) tuples for action space

        # Feature    | # Planes
        # ---------------------
        # P1 pieces  | 5
        # P2 pieces  | 5
        # Repetition | 2
        # P1 capture | 3
        # P2 capture | 3
        # ---------------------
        # Color      | 1
        # Move count | 1
        self.state = torch.zeros(size=(18 * TIME_HISTORY + 2, 4, 3))

    