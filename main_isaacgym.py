# -*- coding: utf-8 -*-
"""
IsaacGym version main training entry point
Uses PackingGame (IsaacGym) for GPU vectorized training
"""
# Must import isaacgym before torch
from isaacgym import gymapi, gymtorch

import copy
import os
import numpy as np
import torch
from agent import Agent
from tensorboardX import SummaryWriter
import time
from tools import backup, load_shape_dict, shotInfoPre, shapeProcessing
from trainer_isaacgym import TrainerIsaacGym, VectorizedReplayMemory
from arguments import get_args

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment/physics0"))


def main(args):
    # ==================== 1. Experiment Configuration ====================
    if args.custom is None:
        args.custom = input('Please input the experiment name\n')
    timeStr = args.custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    # Device configuration
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda:{}'.format(args.device))
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(args.device)

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # ==================== IsaacGym Specific Configuration ====================
    # Number of parallel environments (IsaacGym supports large-scale parallelism)
    if args.num_processes == 2:  # only override the argument default, not an explicit CLI value
        args.num_processes = 512

    # Scale factor
    args.scale = [5, 5, 5]

    # Bin size: 0.32 / 0.01 = 32x32 height map (required by CNN)
    args.bin_dimension = np.array([0.32, 0.32, 0.30])

    # Number of rotations: 2 means only two rotations (cubes only need 2)
    args.ZRotNum = 2 # 2

    # Replay memory size
    args.memory_capacity = 200000

    # Learning rate (moderately increased to speed up learning)
    args.learning_rate = 1e-4  # default 6.25e-5

    # 1-step return: the vectorised buffer writes all 512 envs to CONSECUTIVE positions,
    # so positions [p, p+1, p+2] come from 3 different environments — multi-step lookback
    # across those positions produces rewards from unrelated episodes, which is incorrect.
    # Proper n-step requires a per-env ring buffer; until then, use n=1 (standard DQN).
    args.multi_step = 1

    # Fixed item sequence (for debugging)
    if args.fixed_sequence:
        args.dataSample = 'fixed'
        print("Using fixed item sequence mode")

    # ==================== 2. Backup and Logging ====================
    backup(timeStr, args)
    log_writer_path = './logs/runs/{}'.format('IR-IsaacGym-' + timeStr)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)

    # ==================== 3. Create IsaacGym Environment ====================
    print("=" * 60)
    print("   IR-BPP IsaacGym Training")
    print("=" * 60)
    print(f"\n[1] Creating IsaacGym environment...")

    from environment.physics0.binPhy_isaacgym import PackingGame

    env = PackingGame(
        args=args,
        num_envs=args.num_processes,
        sim_device=args.device.index if args.device.type == 'cuda' else 0,
        headless=True  # Disable rendering to speed up training
    )

    print(f"    - Number of environments: {env.num_envs}")
    print(f"    - Device: {env.device}")
    print(f"    - Number of item types: {env.num_item_types}")
    print(f"    - Observation dimension: {env.obs_len}")
    print(f"    - Action space: {env.action_space.n}")

    # Update args
    args.action_space = env.action_space.n
    obs_len = env.obs_len

    # ==================== 4. Create Agent and Memory ====================
    print(f"\n[2] Creating Agent and Memory...")

    # Non-hierarchical policy: use location network only
    args.level = 'location'
    args.model = args.locmodel
    dqn = Agent(args)

    # Use vectorized experience replay
    mem = VectorizedReplayMemory(
        args=args,
        capacity=args.memory_capacity,
        obs_len=obs_len,
        num_envs=args.num_processes
    )

    trainTool = TrainerIsaacGym(writer, timeStr, dqn, mem)
    print(f"    - Policy type: Online Packing")
    print(f"    - Replay memory capacity: {args.memory_capacity} (shared)")
    print(f"    - Model: {args.model}")

    # ==================== 5. Start Training ====================
    if args.evaluate:
        print("\n[3] Evaluation mode...")
        print("    Warning: test function for IsaacGym version is not yet implemented")
    else:
        print(f"\n[3] Starting training...")
        print(f"    - Total steps: {args.T_max}")
        print(f"    - Learning start step: {args.learn_start}")
        print(f"    - Batch size: {args.batch_size}")
        print(f"    - Target network update interval: {args.target_update}")
        print("-" * 60)

        trainTool.train_q_value(env, args)

    print("\nTraining complete!")


if __name__ == '__main__':
    # Default to cube_discrete for isaacgym training; override with --dataset if needed
    if '--dataset' not in sys.argv:
        sys.argv += ['--dataset', 'cube_discrete']
    args = get_args()
    main(args)
