# -*- coding: utf-8 -*-
"""
IsaacGym version test script
Parameters must match those used during training in main_isaacgym.py (confirmed from backup arguments.py)
"""
# Must import isaacgym before torch
from isaacgym import gymapi, gymtorch

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment/physics0"))

from agent import Agent
from arguments import get_args


def test_isaacgym(args, model_path, num_episodes=100, num_envs=16, headless=True, verbose=True):
    """
    IsaacGym environment test function
    """
    # ==================== 1. Configure parameters (must match training) ====================
    args.num_processes = num_envs
    args.globalView = True

    # [CRITICAL] These parameters must match those used during training (confirm from logs/.../arguments.py)
    args.scale = [5, 5, 5]  # value used during training
    args.bin_dimension = np.array([0.32, 0.32, 0.30])  # value used during training
    args.ZRotNum = 2  # value used during training
    # args.dataSample = 'fixed'  # use fixed sequence

    # Device configuration
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda:{}'.format(args.device if isinstance(args.device, int) else 0))
    else:
        args.device = torch.device('cpu')

    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(args.device)

    # ==================== 2. Create environment ====================
    if verbose:
        print("=" * 60)
        print("   IR-BPP IsaacGym Test")
        print("=" * 60)
        print(f"\n[1] Creating IsaacGym environment...")
        print(f"    Parameter configuration (matching training):")
        print(f"    - scale: {args.scale}")
        print(f"    - bin_dimension: {args.bin_dimension}")
        print(f"    - ZRotNum: {args.ZRotNum}")
        print(f"    - dataSample: {args.dataSample}")

    from environment.physics0.binPhy_isaacgym import PackingGame

    env = PackingGame(
        args=args,
        num_envs=num_envs,
        sim_device=args.device.index if args.device.type == 'cuda' else 0,
        headless=headless
    )

    if verbose:
        print(f"    - Number of environments: {env.num_envs}")
        print(f"    - Device: {env.device}")
        print(f"    - Observation dimension: {env.obs_len}")
        print(f"    - Action space: {env.action_space.n}")
        print(f"    - bin_volume: {env.bin_volume.item():.6f} m³")

    # ==================== 3. Load model ====================
    if verbose:
        print(f"\n[2] Loading model: {model_path}")

    args.action_space = env.action_space.n
    args.level = 'location'
    args.model = model_path

    dqn = Agent(args)
    dqn.online_net.eval()

    # ==================== 4. Test loop ====================
    if verbose:
        print(f"\n[3] Starting test ({num_episodes} episodes)...")

    result_rewards = torch.zeros(num_episodes, device=env.device)
    result_lengths = torch.zeros(num_episodes, device=env.device, dtype=torch.long)
    result_ratios = torch.zeros(num_episodes, device=env.device)

    completed_episodes = 0
    step = 0
    max_steps = num_episodes * 200

    state = env.reset()
    episode_rewards = torch.zeros(num_envs, device=env.device)
    episode_lengths = torch.zeros(num_envs, device=env.device, dtype=torch.long)

    pbar = tqdm(total=num_episodes, desc="Testing") if verbose else None

    def get_mask_from_candidates(candidates, num_envs, selected_action):
        if candidates is None:
            return None
        validity = candidates[:, :, 4]
        mask = (validity > 0).float()
        return mask

    while completed_episodes < num_episodes and step < max_steps:
        step += 1

        if env.candidates is not None:
            mask = get_mask_from_candidates(env.candidates, num_envs, args.selectedAction)
        else:
            mask = None

        with torch.no_grad():
            action = dqn.act(state, mask)

        next_state, reward, done, info = env.step(action)

        episode_rewards += reward
        episode_lengths += 1

        if done.any():
            done_mask = done
            num_done = done_mask.sum().item()

            remaining = num_episodes - completed_episodes
            record_count = min(num_done, remaining)

            if record_count > 0:
                done_indices = torch.where(done_mask)[0][:record_count]

                store_start = completed_episodes
                store_end = completed_episodes + record_count
                result_rewards[store_start:store_end] = episode_rewards[done_indices]
                result_lengths[store_start:store_end] = episode_lengths[done_indices]
                result_ratios[store_start:store_end] = info['ratio'][done_indices]

                completed_episodes += record_count
                if pbar:
                    pbar.update(record_count)

            episode_rewards[done_mask] = 0
            episode_lengths[done_mask] = 0

            done_envs = torch.where(done_mask)[0]
            reset_obs = env.reset(done_envs)
            next_state[done_envs] = reset_obs

        state = next_state

    if pbar:
        pbar.close()

    # ==================== 5. Compute statistics ====================
    all_rewards = result_rewards[:completed_episodes].cpu().numpy()
    all_lengths = result_lengths[:completed_episodes].cpu().numpy()
    all_ratios = result_ratios[:completed_episodes].cpu().numpy()

    if completed_episodes > 0:
        max_ratio_idx = np.argmax(all_ratios)

        results = {
            'num_episodes': completed_episodes,
            'avg_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'avg_length': float(np.mean(all_lengths)),
            'std_length': float(np.std(all_lengths)),
            'avg_ratio': float(np.mean(all_ratios)),
            'std_ratio': float(np.std(all_ratios)),
            'min_ratio': float(np.min(all_ratios)),
            'max_ratio': float(np.max(all_ratios)),
            'max_ratio_length': int(all_lengths[max_ratio_idx]),
        }
    else:
        results = {'num_episodes': 0, 'avg_reward': 0.0, 'avg_length': 0.0, 'avg_ratio': 0.0}

    if verbose:
        print("\n" + "=" * 60)
        print("   Test Results")
        print("=" * 60)
        print(f"  Completed episodes: {results['num_episodes']}")
        print(f"\n  Reward statistics:")
        print(f"    - Mean: {results['avg_reward']:.4f}")
        print(f"    - Std:  {results.get('std_reward', 0):.4f}")
        print(f"\n  Episode length statistics:")
        print(f"    - Mean: {results['avg_length']:.2f}")
        print(f"    - Std:  {results.get('std_length', 0):.2f}")
        print(f"\n  Space utilization (Ratio) statistics:")
        print(f"    - Mean: {results['avg_ratio']:.4f} ({results['avg_ratio']*100:.2f}%)")
        print(f"    - Std:  {results.get('std_ratio', 0):.4f}")
        print(f"    - Min:  {results.get('min_ratio', 0):.4f} ({results.get('min_ratio', 0)*100:.2f}%)")
        print(f"    - Max:  {results.get('max_ratio', 0):.4f} ({results.get('max_ratio', 0)*100:.2f}%)")
        if 'max_ratio_length' in results:
            print(f"    - Episode length at max ratio: {results['max_ratio_length']}")
        print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='IsaacGym BPP test script', add_help=False)
    parser.add_argument('--model', type=str, default=None, help='model path')
    parser.add_argument('--test-episodes', type=int, default=100, help='number of test episodes')
    parser.add_argument('--test-envs', type=int, default=16, help='number of parallel environments')
    parser.add_argument('--test-render', action='store_true', help='render mode')

    test_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = get_args()

    if test_args.model is None:
        print("Error: please specify a model path with --model")
        print("Example: python test_isaacgym.py --model ./logs/experiment/xxx/checkpoint.pt")
        return

    results = test_isaacgym(
        args=args,
        model_path=test_args.model,
        num_episodes=test_args.test_episodes,
        num_envs=test_args.test_envs,
        headless=not test_args.test_render,
        verbose=True
    )


if __name__ == '__main__':
    main()
