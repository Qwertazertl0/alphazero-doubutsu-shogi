import torch
from typing import Optional, Sequence, Union
from collections import defaultdict

TOTAL_NODE_COUNT = 0

TIME_STEPS = 1
ROWS = 4
COLS = 3
STATE_PLANES = 18
META_PLANES = 2
class ShogiState():
    '''
    Represents game state using binary/count planes as described in original AlphaZero paper.
    Doubutsu Shogi uses a 4 x 3 board.
    '''
    def __init__(self) -> None:
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
        #
        # Order: K, B, R, P, C, k, b, r, p, c
        #        r1, r2
        #        B, R, P, b, r, p
        #        c, m
        self.state = torch.zeros(size=(STATE_PLANES * TIME_STEPS + META_PLANES, ROWS, COLS), dtype=int)

    def set_to_start(self) -> None:
        '''Set the board to the game's starting position'''
        self.state[:,:,:] = 0
        self.state[0,3,1] = 1 # K
        self.state[1,3,0] = 1 # B
        self.state[2,3,2] = 1 # R
        self.state[3,2,1] = 1 # P
        self.state[5,0,1] = 1 # k
        self.state[6,0,2] = 1 # b
        self.state[7,0,0] = 1 # r
        self.state[8,1,1] = 1 # p

    def __repr__(self) -> str:
        '''Returns current board state in format: xxx/xxx/xxx/xxx|B#R#P#|b#r#p#|#|#|##'''
        curr = self.state[:10, :, :]
        ch_set = 'KBRPCkbrpc'
        out = []
        for i in range(ROWS):
            for j in range(COLS):
                n = torch.argmax(curr[:,i,j])
                out.append(ch_set[n] if curr[:,i,j][n] else '-')
            out.append('/')
        out[-1] = '|'
        out.append(f'B{self.state[12,0,0]}R{self.state[13,0,0]}P{self.state[14,0,0]}|')
        out.append(f'b{self.state[15,0,0]}r{self.state[16,0,0]}p{self.state[17,0,0]}|')
        rep = self.state[10,0,0] + self.state[11,0,0]
        out.append(f'{self.state[-2,0,0]}|{rep}|{self.state[-1,0,0]}')
        return ''.join(out)

    def pretty_print(self) -> None:
            '''Prints board in 2D format'''
            pass
            # TODO

def repr_to_state(repr: str) -> torch.Tensor:
    pass # TODO
    
def get_actions(state: Union[ShogiState, torch.Tensor]) -> torch.Tensor:
    '''Given assumed valid ShogiState, return legal move mask'''
    if type(state) == ShogiState:
        state = state.state
    
    pieces = state[:5].sum(dim=0)
    empty = 1-pieces
    krpc = pieces - state[1]
    kbc = pieces - state[2] - state[3]
    kb = kbc - state[4]
    krc = krpc - state[3]
    action_planes = torch.zeros(12, ROWS, COLS, dtype=int)

    # regular movements
    for move_group, move_pieces in zip([[0], [1, 7], [3, 5], [2, 4, 6]], [krpc, kbc, kb, krc]):
        for m in move_group:
            i = 1 if m in [1, 2, 3] else (0 if m in [0, 4] else -1)
            j = 1 if m in [3, 4, 5] else (0 if m in [2, 6] else -1)
            bounds = torch.zeros(4,3,dtype=bool)
            if i == 1:
                bounds[:,-1] = True
            elif i == -1:
                bounds[:,0] = True
            if j == 1:
                bounds[-1,:] = True
            elif j == -1:
                bounds[0,:] = True
            moveable = (move_pieces.roll(shifts=(j,i), dims=(0,1)) & empty).roll(shifts=(-j,-i), dims=(0,1))
            action_planes[m] = move_pieces & ~bounds & moveable

    # promotions
    action_planes[8] = empty & state[3]
    action_planes[0] ^= state[3] & torch.tensor([[0,0,0],[1,1,1],[0,0,0],[0,0,0]], dtype=bool)

    # drops
    empty_drop = 1-state[:10].sum(dim=0)
    if state[12][0,0] > 0:
        action_planes[9] = empty_drop
    if state[13][0,0] > 0:
        action_planes[10] = empty_drop
    if state[14][0,0] > 0:
        action_planes[11] = empty_drop

    return action_planes

def check_end(state: Union[ShogiState, torch.Tensor], actions=None) -> Optional[int]:
    '''Check assumed valid ShogiState and return -1/0/1 for current player loss/draw/win or None if game is not over'''
    if type(state) == ShogiState:
        state = state.state

    # check for captured king
    if state[0].sum() == 0:
        return -1
    
    # check try rule
    if not actions:
        actions = get_actions(state)
    if (state[5] & torch.tensor([[0,0,0],[0,0,0],[0,0,0],[1,1,1]], dtype=bool)).sum():
        if state[5,3,0] and (actions[4,2,0] + actions[5,2,1] + actions[6,3,1]) == 0:
            return -1
        elif state[5,3,1] and (actions[2,3,0] + actions[3,2,0] + actions[4,2,1] + actions[5,2,2] + actions[6,3,2]) == 0:
            return -1
        elif state[5,3,2] and (actions[2,3,1] + actions[3,2,1] + actions[4,2,2]) == 0:
            return -1

    # check repetition count
    if state[11,0,0]:
        return 0
    
    return actions

def execute_action(state: ShogiState, state_counts: defaultdict, action: int) -> None:
    '''Given state-action pair (s, a), return resulting state'''
    plane = action // 12
    r = action % 12 // COLS
    c = action % COLS

    # update board state and captures if needed
    if plane < 8:
        piece = state.state[:5,r,c]
        dc = 1 if plane in [1,2,3] else (0 if plane in [0,4] else -1)
        dr = 1 if plane in [3,4,5] else (0 if plane in [2,6] else -1)
        if state.state[5:10,r+dr,c+dc].sum():
            p = torch.argmax(state.state[5:10,r+dr,c+dc])
            if p > 0:
                state.state[[None,12,13,14,14][p]] += 1
            state.state[5:10,r+dr,c+dc] = 0
        state.state[:5,r+dr,c+dc] = piece
        state.state[:5,r,c] = 0
    elif plane == 8:
        assert(r == 1)
        state.state[3,r,c] = 0
        state.state[4,r-1,c] = 1
    else:
        state.state[plane-8,r,c] = 1
        state.state[plane+3] -= 1

    # flip board and captures and color and update move count
    state.state[:10,:,:] = torch.flip(state.state[:10,:,:], dims=[1])
    state.state[:5,:,:], state.state[5:10,:,:] = state.state[5:10,:,:].clone(), state.state[:5,:,:].clone()
    state.state[12:15,:,:], state.state[15:18,:,:] = state.state[15:18,:,:].clone(), state.state[12:15,:,:].clone()
    state.state[-2] = 1 - state.state[-2]
    state.state[-1] += 1

    # update repetitions and state counts
    state_repr = state.__repr__()[:31]
    state_counts[state_repr] += 1
    if state_counts[state_repr] == 1:
        state.state[10:12] = 0
    elif state_counts[state_repr] == 2:
        state.state[10] = 1
        state.state[11] = 0
    elif state_counts[state_repr] == 3:
        state.state[10:12] = 1

    # update state history (for TIME_STEPS > 1)
    # TODO

class ShogiNode():
    '''
    Node wrapping state and action for MCTS
    '''
    def __init__(self, state=None, actions=None) -> None:
        self.children = []

        # Actions (12): Order
        # 1 unit in each cardinal direction and diagonals (8): N, NE, E, SE, S, SW, W, NW
        # Pawn promotion (1)
        # Drop (3): B, R, P
        self.edges = torch.zeros(size=(4, 12, ROWS, COLS)) # (N,W,Q,P) tuples for action space
        self._state = ShogiState()
        if state:
            self._state.state = state
        self.state = self._state.state

    def __repr__(self) -> str:
        return self._state.__repr__()