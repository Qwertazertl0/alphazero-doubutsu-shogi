import torch
from typing import Optional, Sequence, Union
from collections import defaultdict
from mcts import AbstractNode

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
    def __init__(self, state=None) -> None:
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
        if state is None:
            self.state = torch.zeros(size=(STATE_PLANES * TIME_STEPS + META_PLANES, ROWS, COLS), dtype=int)
        else:
            self.state = state

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
        return state_to_repr(self.state)

    def pretty_print(self) -> None:
            '''Prints board in 2D format'''
            pass
            # TODO

def state_to_repr(state: torch.Tensor) -> str:
    '''Returns current board state in format: xxx/xxx/xxx/xxx|B#R#P#|b#r#p#|#|#|##'''
    curr = state[:10, :, :]
    ch_set = 'KBRPCkbrpc'
    out = []
    for i in range(ROWS):
        for j in range(COLS):
            n = torch.argmax(curr[:,i,j])
            out.append(ch_set[n] if curr[:,i,j][n] else '-')
        out.append('/')
    out[-1] = '|'
    out.append(f'B{state[12,0,0]}R{state[13,0,0]}P{state[14,0,0]}|')
    out.append(f'b{state[15,0,0]}r{state[16,0,0]}p{state[17,0,0]}|')
    rep = state[10,0,0] + state[11,0,0]
    out.append(f'{state[-2,0,0]}|{rep}|{state[-1,0,0]}')
    return ''.join(out)

def repr_to_state(repr: str) -> torch.Tensor:
    state = torch.zeros(size=(STATE_PLANES * TIME_STEPS + META_PLANES, ROWS, COLS), dtype=int)
    # TODO: time steps > 1

    board, pc1, pc2, color, rep, move_cnt = repr.split('|')
    state[-1], state[-2] = int(move_cnt), int(color)
    state[10], state[11] = int(rep in '12'), int(rep == '2')
    state[12], state[13], state[14] = map(int, [pc1[1], pc1[3], pc1[5]])
    state[15], state[16], state[17] = map(int, [pc2[1], pc2[3], pc2[5]])
    ix_map = {ch:ix for ch, ix in zip('KBRPCkbrpc-', [*range(10),-1])}
    for r, row in enumerate(board.split('/')):
        for c, ch in enumerate(row):
            plane = ix_map[ch]
            if plane >= 0:
                state[plane, r, c] = 1
    return state
    
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

def get_action_index(action: int) -> tuple[int, int, int]:
    plane = action // 12
    r = action % 12 // COLS
    c = action % COLS
    return plane, r, c

def get_action_str(action: Union[int, tuple]) -> None:
    p, r, c = get_action_index(action) if isinstance(action, int) else action
    verb = 'Move' if p < 8 else ('Promote' if p == 8 else 'Drop')
    if p >= 8:
        piece = 'PBRP'[p-8]
        start = ''
        end = f'at [{r},{c}]'
    else:
        piece = 'piece'
        start = f' at [{r},{c}]'
        i = 1 if p in [1, 2, 3] else (0 if p in [0, 4] else -1)
        j = 1 if p in [3, 4, 5] else (0 if p in [2, 6] else -1)
        end = f'to [{r+j},{c+i}]'
    return f'[{action}] {verb} {piece}{start} {end}'

def flip_board(state: torch.Tensor) -> torch.Tensor:
    state[:10,:,:] = torch.flip(state[:10,:,:], dims=[1,2])
    state[:5,:,:], state[5:10,:,:] = state[5:10,:,:].clone(), state[:5,:,:].clone()
    state[12:15,:,:], state[15:18,:,:] = state[15:18,:,:].clone(), state[12:15,:,:].clone()
    return state

def check_end(state: Union[ShogiState, torch.Tensor], actions=None) -> Optional[Union[int, torch.Tensor]]:
    '''Check assumed valid ShogiState and return -1/0/1 for sente player loss/draw/win or None if game is not over.'''
    if type(state) == ShogiState:
        state = state.state

    if actions is None:
        actions = get_actions(state)

    loss_val = -1 if state[-2,0,0] == 0 else 1
    # check for captured king
    if state[0].sum() == 0:
        return loss_val
    elif state[5].sum() == 0:
        return -loss_val
    
    # check try rule
    if (state[5] & torch.tensor([[0,0,0],[0,0,0],[0,0,0],[1,1,1]], dtype=bool)).sum() > 0:
        if state[5,3,0] and (actions[4,2,0] + actions[5,2,1] + actions[6,3,1]) == 0:
            return loss_val
        elif state[5,3,1] and (actions[2,3,0] + actions[3,2,0] + actions[4,2,1] + actions[5,2,2] + actions[6,3,2]) == 0:
            return loss_val
        elif state[5,3,2] and (actions[2,3,1] + actions[3,2,1] + actions[4,2,2]) == 0:
            return loss_val
    elif (state[0] & torch.tensor([[1,1,1],[0,0,0],[0,0,0],[0,0,0]], dtype=bool)).sum() > 0:
        opp_state = flip_board(state.clone())
        opp_actions = get_actions(opp_state)
        if opp_state[5,3,0] and (opp_actions[4,2,0] + opp_actions[5,2,1] + opp_actions[6,3,1]) == 0:
            return -loss_val
        elif opp_state[5,3,1] and (opp_actions[2,3,0] + opp_actions[3,2,0] + opp_actions[4,2,1] + opp_actions[5,2,2] + opp_actions[6,3,2]) == 0:
            return -loss_val
        elif opp_state[5,3,2] and (opp_actions[2,3,1] + opp_actions[3,2,1] + opp_actions[4,2,2]) == 0:
            return -loss_val
        
    # check repetition count
    if state[11,0,0]:
        return 0

def execute_action(state: Union[ShogiState, torch.Tensor], state_counts: defaultdict, action: int) -> None:
    '''Given state-action pair (s, a), modify input state to produce resulting state'''
    if type(state) == ShogiState:
        state = state.state
    plane, r, c = get_action_index(action)

    # update board state and captures if needed
    if plane < 8:
        piece = state[:5,r,c]
        dc = 1 if plane in [1,2,3] else (0 if plane in [0,4] else -1)
        dr = 1 if plane in [3,4,5] else (0 if plane in [2,6] else -1)
        if state[5:10,r+dr,c+dc].sum():
            p = torch.argmax(state[5:10,r+dr,c+dc])
            if p > 0:
                state[[None,12,13,14,14][p]] += 1
            state[5:10,r+dr,c+dc] = 0
        state[:5,r+dr,c+dc] = piece
        state[:5,r,c] = 0
    elif plane == 8:
        assert(r == 1)
        state[3,r,c] = 0
        state[4,r-1,c] = 1
    else:
        state[plane-8,r,c] = 1
        state[plane+3] -= 1

    # flip board and captures and color and update move count
    state = flip_board(state)
    state[-2] = 1 - state[-2]
    state[-1] += 1

    # update repetitions and state counts
    state_repr = state_to_repr(state)[:31]
    state_counts[state_repr] += 1
    if state_counts[state_repr] == 1:
        state[10:12] = 0
    elif state_counts[state_repr] == 2:
        state[10] = 1
        state[11] = 0
    elif state_counts[state_repr] == 3:
        state[10:12] = 1

    # update state history (for TIME_STEPS > 1)
    # TODO
    
class ShogiNode(AbstractNode):
    '''
    Node wrapping state and action for MCTS
    '''
    def __init__(self, in_action: int, parent=None, state=None) -> None:
        self.parent = parent
        self.in_action = in_action
        self.children = {}

        # Actions (12): Order
        # 1 unit in each cardinal direction and diagonals (8): N, NE, E, SE, S, SW, W, NW
        # Pawn promotion (1)
        # Drop (3): B, R, P
        self.edges = torch.zeros(size=(3, 12, ROWS, COLS)) # (N,W,Q) tuples for action space
        self._state = ShogiState(state)
        self.state = self._state.state
        self.actions = get_actions(self.state)
        self.prior = None

        self.value = check_end(self.state, self.actions)
        self.evaluated = False

    def __repr__(self) -> str:
        return self._state.__repr__()
    
    def destruct(self) -> None:
        self.parent = None
        self.prior, self.value = None, None
        self.edges, self._state, self.state = None, None, None
        for _, node in self.children.items():
            node.destruct()
        self.children = None
    
    def set_to_start(self) -> None:
        self.parent = None
        self.in_action = None
        self.children = {}
        self.edges = torch.zeros(size=(4, 12, ROWS, COLS))

        self._state.set_to_start()
        self.actions = get_actions(self.state)
        self.value = check_end(self.state, self.actions)
        self.evaluated = False
    
    def is_over(self) -> bool:
        return isinstance(self.value, int)
    
    def select_rollout_action(self, c_puct: float) -> torch.Tensor:
        n_total = self.edges[0].sum()
        c_n = torch.sqrt(n_total) / (1 + self.edges[0])
        u = c_puct * self.prior * (1 if n_total == 0 else c_n)
        # assert((self.actions == get_actions(self.state)).all()), 'self.actions is stale'
        if self.state[-2,0,0] == 0:
            a = torch.argmax((self.edges[2]+1)*self.actions + u).item()
        else:
            a = torch.argmin((self.edges[2]-1)*self.actions - u).item()
        # assert(self.actions.flatten()[a] == 1), f'Illegal move attempted: {a} on {self}'
        return a

    def execute_action(self, action: int, state_counts) -> "AbstractNode":
        child_state = self.state.clone()
        execute_action(child_state, state_counts, action)

        child = ShogiNode(in_action=action, parent=self, state=child_state)
        self.children[action] = child
        return child
    
    def evaluate(self, agent) -> float:
        if self.value is not None:
            return self.value
        
        input_ = self.state[:TIME_STEPS*STATE_PLANES].to(torch.float)[None,:,:,:]
        p, v = agent(input_)
        p_exp = torch.exp(p-p[self.actions].max())
        p = p_exp / (p_exp*self.actions).sum()
        p[self.actions==0] = 0
        # assert(all(self.actions[p>0])), 'Positive prior assigned to illegal move'
        self.prior = p

        self.evaluated = True
        self.value = v
        return v.item()
    
    def backup(self, value: float, root: "ShogiNode") -> None:
        self._backup(value, root.__repr__, self.in_action)

    def _backup(self, value: float, root_repr: str, in_action: int) -> None:
        if self.__repr__ == root_repr:
            return
        
        p, r, c = get_action_index(in_action)
        # assert(self.parent.actions[p, r, c]==1), 'Illegal in-action'
        self.parent.edges[0, p, r, c] += 1
        self.parent.edges[1, p, r, c] += value
        self.parent.edges[2, p, r, c] = self.parent.edges[1, p, r, c] / self.parent.edges[0, p, r, c]
        self.parent._backup(value, root_repr, self.parent.in_action)
    
    def get_search_vector(self, temp: float) -> torch.Tensor:
        if temp == 0:
            amax = self.edges[0].argmax()
            vec = torch.zeros_like(self.edges[0])
            vec[get_action_index(amax)] = 1
            return vec
        with torch.no_grad():
            n = torch.pow(self.edges[0], 1/temp)
            sv = n / n.sum()
        # if sv.isnan().any():
        #     import pdb
        #     breakpoint()

        # assert((self.actions == get_actions(self.state)).all()), 'self.actions is stale'
        # print(torch.arange(144)[self.edges[0].flatten()>0])
        # print(self.edges[0].flatten()[self.edges[0].flatten()>0])
        # if not (self.actions - (sv>0).int()).min() >= 0:
        #     print(self.actions[self.edges[0]>0])
        #     print(self.edges[0][self.edges[0]>0])
        #     assert(False)
        # print(torch.arange(144)[sv.flatten()>0])
        return sv