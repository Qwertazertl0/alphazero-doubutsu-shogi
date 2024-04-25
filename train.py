import torch
from torch.optim import AdamW
import argparse
from collections import defaultdict
from torch.distributions.categorical import Categorical

from agent.network import AgentNetwork
from mcts import MCTS
from game_state import *

def parse_args():
    parser = argparse.ArgumentParser(prog='AlphaZero-DoubutsuShogi-Trainer',
                                     description='Training script for Doubutsu Shogi agent via AlphaZero algorithm')
    
    parser.add_argument('model_id', type=str, help='Unique model identifier for this training run')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of self-play games to train on')
    parser.add_argument('--num_blocks', type=int, default=9, help='Number of conv blocks to use in model')
    parser.add_argument('--num_rollouts', type=int, default=100, help='Number of rollouts to use in MCTS')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--ckpt_freq', type=int, default=0, help='Number of steps between each checkpoint')
    parser.add_argument('--load_ckpt', type=str, default='', help='Path to model checkpoint to be loaded')
    parser.add_argument('--c_puct', type=float, default=1, help='Value of constant used in PUCT algorithm')
    parser.add_argument('--temperature', type=float, default=0.4, help='Sampling temperature when choosing from search')

    args = parser.parse_args()
    return args

def train_step(agent: AgentNetwork, optim: torch.optim.Optimizer, configs: argparse.Namespace) -> None:
    '''Main training; plays one game of doubutsu shogi against itself, and updates parameters'''
    start_node = ShogiNode(-1, None, None)
    start_node.set_to_start()
    game_state_counts = defaultdict(int)
    mcts = MCTS(agent, c_puct=configs.c_puct, temp=configs.temperature, n_rollouts=configs.num_rollouts)
    start_value = start_node.evaluate(agent)

    # Play out game
    curr = start_node
    while not curr.is_over():
        print(curr, f'{curr.value.item():0.4f}')
        if curr.state[-1,0,0] > 100:
            # TODO
            return
        search = mcts.run_tree_search(curr, game_state_counts)
        dist = Categorical(search.flatten())
        a = int(dist.sample())

        p, r, c = get_action_index(a)
        print(f'\t{get_action_str([p,r,c]):40s} (Value: {curr.edges[2,p,r,c]:0.4f}; Visit Count: {curr.edges[0,p,r,c]:0.0f})')

        for action in list(curr.children.keys()):
            if a != action:
                curr.children[action].destruct()
                curr.children.pop(action)
        curr = curr.children[a]
        game_state_counts[state_to_repr(curr.state)[:31]] += 1
    move_cnt = curr.state[-1,0,0]

    # Compute loss
    z = check_end(curr.state)
    print(f'Game result: {z}')
    policy_loss = 0
    score_loss = 0
    l2_w_reg = 0
    for p in agent.parameters():
        if p.requires_grad:
            l2_w_reg = l2_w_reg + (p**2).sum()
    l2_w_reg = torch.sqrt(l2_w_reg)
    c_wr = 1e-4
    c_score = 1e-2

    curr = curr.parent
    while curr.parent is not None:
        flip = -1 if curr.state[-2,0,0] == 1 else 1
        score_loss = score_loss + (flip*z - curr.value) ** 2
        policy_loss = policy_loss + torch.nansum(-curr.get_search_vector(temp=configs.temperature) * torch.log(curr.prior))
        curr = curr.parent
    score_loss = c_score * score_loss / move_cnt
    policy_loss = policy_loss / move_cnt
    regularizer_loss = c_wr * l2_w_reg
    total_loss = score_loss + policy_loss + regularizer_loss
    total_loss.backward()
    optim.step()
    optim.zero_grad()

    print(f'Avg. Loss (S/P/R/T): {score_loss.item():0.4f} / {policy_loss.item():0.4f}' \
          f' / {regularizer_loss.item():0.4f} / {total_loss.item():0.4f}')
    print(f'Start Value: {start_value}')
    curr.destruct()


if __name__ == '__main__':
    import sys
    print('Command:', *sys.argv)
    args = parse_args()

    if args.load_ckpt != '':
        agent = torch.load(args.load_ckpt)
    else:
        agent = AgentNetwork(num_blocks=args.num_blocks)
    print('Model Parameters:', sum(p.numel() for p in agent.parameters() if p.requires_grad))
    optim = AdamW(agent.parameters(), lr=args.lr)

    for s in range(args.num_steps):
        print('Step:', s)
        train_step(agent, optim, args)
        if s > 0 and args.ckpt_freq > 0 and s % args.ckpt_freq == 0:
            filename = f'ckpts/{args.model_id}_step_{s}.pt' # TODO: decide on format, add datetime?
            torch.save(agent, filename)

    filename = f'ckpts/{args.model_id}_step_{s}_final.pt' # TODO: decide on format, add datetime?
    torch.save(agent, filename)