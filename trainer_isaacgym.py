"""
IsaacGym version trainer
Adapted to the vectorized environment interface of PackingGame (IsaacGym)
Fully vectorized implementation, avoiding explicit for loops over environments
"""
import os
import numpy as np
import torch
from tqdm import trange
from collections import deque
from tensorboardX import SummaryWriter
import time


def get_mask_from_candidates(candidates, num_envs, num_actions, device):
    """
    Extract action mask from the candidates tensor

    Args:
        candidates: [num_envs, selectedAction, 5] candidate action tensor
        num_envs: number of environments
        num_actions: action space size (selectedAction)
        device: device

    Returns:
        mask: [num_envs, num_actions] valid action mask (0/1 format)
    """
    if candidates is None:
        return None
    # candidates[:, :, 4] is the validity flag; convert to 0/1 format
    validity = candidates[:, :, 4]  # [num_envs, selectedAction]
    mask = (validity > 0).float()  # convert to 0/1 format
    return mask


class VectorizedReplayMemory:
    """
    Vectorized experience replay buffer (GPU version)
    Supports batch addition of experiences from multiple environments
    All data stored on GPU to avoid CPU-GPU transfers
    Compatible with the learn method interface in agent.py
    """
    def __init__(self, args, capacity, obs_len, num_envs):
        self.device = args.device
        self.capacity = capacity
        self.obs_len = obs_len
        self.discount = args.discount
        self.n = args.multi_step
        self.num_envs = num_envs
        self.priority_weight = args.priority_weight
        self.priority_exponent = args.priority_exponent

        # Per-environment timestep counter (GPU)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Storage index
        self.index = 0
        self.full = False
        self.size = capacity

        # Data storage - all on GPU
        self.timesteps = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.env_ids = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.states = torch.zeros(capacity, obs_len, dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros(capacity, obs_len, dtype=torch.float32, device=self.device)  # store next_state directly
        self.actions = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=self.device)
        self.nonterminals = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.priorities = torch.ones(capacity, dtype=torch.float32, device=self.device)

        self.max_priority = 1.0
        self.n_step_scaling = torch.tensor(
            [self.discount ** i for i in range(self.n)],
            dtype=torch.float32,
            device=self.device
        )

    def __len__(self):
        """Compatible with len(memory) call in agent.py"""
        return 1  # used as a single memory

    def __iter__(self):
        """Compatible with for mem in memory call in agent.py"""
        return iter([self])

    def __getitem__(self, idx):
        """Compatible with memory[i] call in agent.py"""
        if idx == 0:
            return self
        raise IndexError(f"VectorizedReplayMemory only has 1 element, got index {idx}")

    def append_batch(self, states, actions, rewards, dones, next_states, valid_mask=None):
        """
        Batch-add experiences (vectorized, fully on GPU)

        Args:
            states: [num_envs, obs_len] state tensor (GPU)
            actions: [num_envs] action tensor (GPU)
            rewards: [num_envs] reward tensor (GPU)
            dones: [num_envs] done flag tensor (GPU)
            next_states: [num_envs, obs_len] next state tensor (GPU)
            valid_mask: [num_envs] valid sample mask; None means all valid
        """
        num_envs = states.shape[0]

        if valid_mask is None:
            valid_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        # Get valid sample indices (GPU)
        valid_indices = torch.where(valid_mask)[0]
        num_valid = len(valid_indices)

        if num_valid == 0:
            return

        # Compute storage positions (GPU)
        store_indices = (torch.arange(num_valid, device=self.device) + self.index) % self.capacity

        # Batch store (all on GPU)
        self.timesteps[store_indices] = self.t[valid_indices]
        self.env_ids[store_indices] = valid_indices
        self.states[store_indices] = states[valid_indices]
        self.next_states[store_indices] = next_states[valid_indices]  # store next_state
        self.actions[store_indices] = actions[valid_indices]
        rewards_squeezed = rewards.squeeze()
        if rewards_squeezed.dim() == 0:
            rewards_squeezed = rewards_squeezed.unsqueeze(0)
        self.rewards[store_indices] = rewards_squeezed[valid_indices]
        self.nonterminals[store_indices] = ~dones[valid_indices]
        self.priorities[store_indices] = self.max_priority

        # Update index
        self.index = (self.index + num_valid) % self.capacity
        self.full = self.full or (self.index < num_valid)

        # Update timesteps - reset done environments to 0 (GPU)
        self.t = torch.where(dones, torch.zeros_like(self.t), self.t + 1)

    def sample(self, batch_size):
        """
        Priority sampling (vectorized implementation, fully on GPU)

        Returns:
            tree_idxs, states, actions, returns, next_states, nonterminals, weights
        """
        # Current number of valid samples
        current_size = self.capacity if self.full else self.index

        if current_size < batch_size + self.n:
            # Not enough samples; return empty tensors
            empty_states = torch.zeros(batch_size, self.obs_len, device=self.device)
            empty_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            empty_returns = torch.zeros(batch_size, device=self.device)
            empty_nonterminals = torch.zeros(batch_size, device=self.device)
            empty_weights = torch.ones(batch_size, device=self.device)
            return list(range(batch_size)), empty_states, empty_actions, empty_returns, empty_states, empty_nonterminals, empty_weights

        # Compute sampling probabilities (GPU)
        valid_priorities = self.priorities[:current_size].clamp(min=1e-8)  # guard against zero/stale entries
        probs = valid_priorities ** self.priority_exponent
        probs = probs / probs.sum()

        # Sample indices (GPU)
        idxs = torch.multinomial(probs, batch_size, replacement=False)

        # Retrieve samples (GPU)
        states = self.states[idxs]
        actions = self.actions[idxs]

        # Vectorized n-step return computation (GPU)
        # Build index matrix [batch_size, n]
        step_offsets = torch.arange(self.n, device=self.device).unsqueeze(0)  # [1, n]
        all_idxs = (idxs.unsqueeze(1) + step_offsets) % self.capacity  # [batch_size, n]

        # Retrieve rewards and done flags
        step_rewards = self.rewards[all_idxs]  # [batch_size, n]
        step_nonterminals = self.nonterminals[all_idxs]  # [batch_size, n]

        # Compute cumulative mask (once done, all subsequent steps are invalid)
        cumulative_nonterminal = torch.cumprod(
            torch.cat([torch.ones(batch_size, 1, device=self.device), step_nonterminals[:, :-1].float()], dim=1),
            dim=1
        )  # [batch_size, n]

        # Discount factors (GPU)
        discounts = self.n_step_scaling  # already on GPU

        # Compute n-step discounted returns
        returns = (step_rewards * cumulative_nonterminal * discounts).sum(dim=1)  # [batch_size]

        # Use stored next_states directly (solves multi-env interleaved storage issue)
        next_states = self.next_states[idxs]

        # Use stored nonterminal
        nonterminals = self.nonterminals[idxs].float()  # [batch_size]

        # Compute importance sampling weights (GPU)
        sampled_probs = probs[idxs]
        weights = (current_size * sampled_probs) ** (-self.priority_weight)
        weights = weights / weights.max()

        # Return GPU tensors directly to avoid CPU-GPU transfers
        return idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        """Update priorities (GPU); stores raw |TD error| — exponent is applied in sample()."""
        if not isinstance(priorities, torch.Tensor):
            priorities = torch.tensor(priorities, dtype=torch.float32, device=self.device)
        else:
            priorities = priorities.to(self.device)
        # Store raw positive priorities; clamp away zero/NaN so they stay samplable
        priorities = priorities.abs().clamp(min=1e-8)

        if not isinstance(idxs, torch.Tensor):
            idxs = torch.tensor(idxs, dtype=torch.long, device=self.device)
        self.priorities[idxs] = priorities
        self.max_priority = torch.max(self.max_priority * torch.ones(1, device=self.device), priorities.max()).item()


class TrainerIsaacGym:
    """DQN trainer for IsaacGym environment (vectorized version)"""

    def __init__(self, writer, timeStr, dqn, mem):
        """
        Args:
            writer: TensorBoard writer
            timeStr: experiment timestamp string
            dqn: DQN Agent
            mem: experience replay buffer (VectorizedReplayMemory or list)
        """
        self.writer = writer
        self.timeStr = timeStr
        self.dqn = dqn
        self.mem = mem

    def train_q_value(self, env, args):
        """
        Main training loop (fully vectorized)

        Args:
            env: PackingGame (IsaacGym) environment instance
            args: training parameters
        """
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)

        # Statistics queues
        episode_rewards = deque(maxlen=100)
        episode_ratio = deque(maxlen=100)
        episode_counter = deque(maxlen=100)

        num_envs = args.num_processes

        # Per-environment cumulative reward and step count (GPU tensor)
        env_rewards = torch.zeros(num_envs, device=env.device)
        env_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)

        # Instant reward statistics (for more frequent TensorBoard logging)
        step_rewards = deque(maxlen=1000)  # recent 1000 steps
        log_interval = 100  # log every 100 steps

        # Reset environment
        state = env.reset()  # [num_envs, obs_len] PyTorch tensor

        reward_clip = args.reward_clip
        loss = 0

        # Training mode
        self.dqn.train()

        print(f"Starting training: T_max={args.T_max}, num_envs={num_envs}")
        print(f"Observation shape: {state.shape}, action space: {args.action_space}")

        for T in trange(1, args.T_max + 1):
            # Reset noise network
            if T % args.replay_frequency == 0:
                self.dqn.reset_noise()

            # Get action mask (vectorized)
            if args.selectedAction:
                mask = get_mask_from_candidates(
                    env.candidates, num_envs, args.action_space, env.device
                )
            else:
                naiveMask = env.stored_naiveMask
                # Convert to 0/1 format: >0 is valid -> 1, otherwise -> 0
                mask = (naiveMask > 0).float().view(num_envs, -1)

            # Select action (vectorized)
            action = self.dqn.act(state, mask)  # [num_envs] tensor

            # ============ Debug: check Q-function ============
            if T % 2000 == 0:
                with torch.no_grad():
                    # Get Q-value distribution
                    q_dist = self.dqn.online_net(state)  # [num_envs, action_space, atoms]
                    q_values = (q_dist * self.dqn.support).sum(2)  # [num_envs, action_space]

                    # Q-values after applying mask
                    q_masked = q_values.clone()
                    q_masked[~mask.bool()] = float('-inf')

                    # Valid Q-value statistics
                    valid_q = q_masked[q_masked > float('-inf')]
                    if len(valid_q) > 0:
                        print(f"\n[T={T}] === Q-function debug ===")
                        print(f"  Q values: mean={valid_q.mean().item():.4f}, std={valid_q.std().item():.4f}, "
                              f"min={valid_q.min().item():.4f}, max={valid_q.max().item():.4f}")

                        # Range between max and min Q values (larger = better action discrimination)
                        q_range = valid_q.max().item() - valid_q.min().item()
                        print(f"  Q range: {q_range:.4f} (larger = better action discrimination)")

                    # Action distribution
                    unique_actions, counts = torch.unique(action, return_counts=True)
                    print(f"  Distinct actions selected: {len(unique_actions)}/{num_envs} (fewer = more concentrated)")
                    top_k = min(5, len(counts))
                    top_counts = counts.topk(top_k)
                    print(f"  Top{top_k} actions: {list(zip(unique_actions[top_counts.indices].tolist(), top_counts.values.tolist()))}")

                    # Check whether network parameters are changing
                    param_sum = sum(p.abs().sum().item() for p in self.dqn.online_net.parameters())
                    print(f"  Sum of absolute network parameters: {param_sum:.2f}")
            # ============ End debug ============

            # Execute action (vectorized)
            next_state, reward, done, info = env.step(action)

            # Update cumulative reward and step count (vectorized, fully GPU)
            env_rewards = env_rewards + reward.squeeze()
            env_steps = env_steps + 1

            # Periodically log to TensorBoard (reduce CPU-GPU synchronization)
            if T % log_interval == 0:
                step_reward = reward.mean().item()
                step_rewards.append(step_reward)
                # deque does not support slicing; use the full deque average
                self.writer.add_scalar('Reward/Step mean (recent)', np.mean(step_rewards) if step_rewards else 0, T)
                self.writer.add_scalar('Reward/Step instant', step_reward, T)

            # Reward clipping (vectorized)
            if reward_clip > 0:
                reward = torch.clamp(reward, -reward_clip, reward_clip)

            # Get valid sample mask (vectorized)
            valid_mask = info.get('valid', torch.ones(num_envs, dtype=torch.bool, device=env.device))
            if not isinstance(valid_mask, torch.Tensor):
                valid_mask = torch.ones(num_envs, dtype=torch.bool, device=env.device)

            # Batch store experiences (vectorized)
            if isinstance(self.mem, VectorizedReplayMemory):
                self.mem.append_batch(state, action, reward, done, next_state, valid_mask)
            else:
                # Compatible with original list format - use vectorized indexing
                state_cpu = state.cpu()
                action_cpu = action.cpu()
                reward_cpu = reward.cpu()
                done_cpu = done.cpu()
                valid_cpu = valid_mask.cpu()

                # Batch process using tensor operations
                valid_indices = torch.where(valid_cpu)[0]
                for idx in valid_indices:
                    i = idx.item()
                    self.mem[i].append(state_cpu[i], action_cpu[i], reward_cpu[i], done_cpu[i].item())

            # Handle episode endings (vectorized)
            if done.any():
                done_mask = done

                # Batch record statistics (vectorized)
                done_rewards = env_rewards[done_mask]
                done_steps = env_steps[done_mask]

                # Add to statistics queues
                episode_rewards.extend(done_rewards.cpu().tolist())
                episode_counter.extend(done_steps.cpu().tolist())

                if 'ratio' in info:
                    ratio = info['ratio']
                    if isinstance(ratio, torch.Tensor):
                        done_ratios = ratio[done_mask]
                        episode_ratio.extend(done_ratios.cpu().tolist())

                # Reset statistics for done environments (vectorized)
                env_rewards = torch.where(done_mask, torch.zeros_like(env_rewards), env_rewards)
                env_steps = torch.where(done_mask, torch.zeros_like(env_steps), env_steps)

                # Reset done environments
                done_env_ids = torch.where(done_mask)[0]
                reset_obs = env.reset(done_env_ids)
                # Update next_state for these environments
                next_state[done_env_ids] = reset_obs

            # Training update
            if T >= args.learn_start:
                # Update priority weight (vectorized)
                if isinstance(self.mem, VectorizedReplayMemory):
                    self.mem.priority_weight = min(
                        self.mem.priority_weight + priority_weight_increase, 1
                    )
                else:
                    # Batch update
                    new_weight = min(self.mem[0].priority_weight + priority_weight_increase, 1)
                    for m in self.mem:
                        m.priority_weight = new_weight

                # Number of learning iterations - fixed small value to maintain speed
                # With many environments, each step collects a lot of data; few iterations suffice
                num_learning_iters = 8  # increased from 4 to better utilise collected data

                if T % args.replay_frequency == 0:
                    for _ in range(num_learning_iters):
                        loss = self.dqn.learn(self.mem)

                    # Debug: print loss info
                    if T % 2000 == 0:
                        mem_size = self.mem.capacity if self.mem.full else self.mem.index
                        if isinstance(loss, torch.Tensor):
                            print(f"  Loss: {loss.mean().item():.6f}, Memory size: {mem_size}, Learning iterations: {num_learning_iters}")
                        else:
                            print(f"  Loss: {loss}, Memory size: {mem_size}, Learning iterations: {num_learning_iters}")

                # Update target network
                if T % args.target_update == 0:
                    self.dqn.update_target_net()
                    print(f"\n[T={T}] >>> Target network updated <<<")

                # Save checkpoint
                if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                    if T % args.checkpoint_interval == 0:
                        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                    self.dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

                # Log training loss
                if T % args.print_log_interval == 0:
                    if isinstance(loss, torch.Tensor):
                        self.writer.add_scalar("Training/Value loss", loss.mean().item(), T)
                    elif loss != 0:
                        self.writer.add_scalar("Training/Value loss", loss, T)

            # Update state
            state = next_state

            # Render (if enabled)
            if hasattr(env, 'render'):
                env.render()

            # Log metrics
            if len(episode_rewards) != 0:
                self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
            if len(episode_ratio) != 0:
                self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
            if len(episode_counter) != 0:
                self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)


class TrainerIsaacGymHierarchical:
    """Hierarchical DQN trainer for IsaacGym environment (vectorized version)"""

    def __init__(self, writer, timeStr, DQNs, MEMs):
        """
        Args:
            writer: TensorBoard writer
            timeStr: experiment timestamp string
            DQNs: [orderDQN, locDQN] two Agent instances
            MEMs: [orderMem, locMem] two experience replay buffers
        """
        self.writer = writer
        self.timeStr = timeStr
        self.orderDQN, self.locDQN = DQNs
        self.orderMem, self.locMem = MEMs

    def train_q_value(self, env, args):
        """
        Hierarchical policy main training loop (fully vectorized)

        Args:
            env: PackingGame (IsaacGym) environment instance
            args: training parameters
        """
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        actionNum = args.action_space
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)

        # Statistics queues
        episode_rewards = deque(maxlen=100)
        episode_ratio = deque(maxlen=100)
        episode_counter = deque(maxlen=100)

        num_envs = args.num_processes

        # Per-environment cumulative reward and step count (GPU tensor)
        env_rewards = torch.zeros(num_envs, device=env.device)
        env_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)

        # Instant reward statistics (for more frequent TensorBoard logging)
        step_rewards = deque(maxlen=1000)
        log_interval = 100

        # Reset environment to get order state
        orderState = env.reset()

        reward_clip = args.reward_clip
        orderLoss, locLoss = 0, 0

        # Training mode
        self.orderDQN.train()
        self.locDQN.train()

        print(f"Starting hierarchical training: T_max={args.T_max}, num_envs={num_envs}")

        for T in trange(1, args.T_max + 1):
            # Reset noise
            if T % args.replay_frequency == 0:
                self.orderDQN.reset_noise()
                self.locDQN.reset_noise()

            # 1. Order network selects item (vectorized)
            orderAction = self.orderDQN.act(orderState, None)

            # 2. Get location candidate state (vectorized)
            locState = env.get_action_candidates(orderAction)

            # 3. Get location mask (vectorized)
            if args.selectedAction:
                locMask_raw = locState[:, 0:args.selectedAction * 5].reshape(
                    num_envs, args.selectedAction, 5
                )[:, :, -1]
                locMask = (locMask_raw > 0).float()  # convert to 0/1 format
            else:
                locMask_raw = locState[:, 0:actionNum].reshape(-1, actionNum)
                locMask = (locMask_raw > 0).float()  # convert to 0/1 format

            # 4. Location network selects position (vectorized)
            locAction = self.locDQN.act(locState, locMask)

            # 5. Execute action (vectorized)
            next_order_state, reward, done, info = env.step(locAction)

            # Update cumulative reward and step count (vectorized, fully GPU)
            env_rewards = env_rewards + reward.squeeze()
            env_steps = env_steps + 1

            # Periodically log to TensorBoard (reduce CPU-GPU synchronization)
            if T % log_interval == 0:
                step_reward = reward.mean().item()
                step_rewards.append(step_reward)
                # deque does not support slicing; use the full deque average
                self.writer.add_scalar('Reward/Step mean (recent)', np.mean(step_rewards) if step_rewards else 0, T)
                self.writer.add_scalar('Reward/Step instant', step_reward, T)

            # Reward clipping (vectorized)
            if reward_clip > 0:
                reward = torch.clamp(reward, -reward_clip, reward_clip)

            # Get valid sample mask (vectorized)
            valid_mask = info.get('valid', torch.ones(num_envs, dtype=torch.bool, device=env.device))
            if not isinstance(valid_mask, torch.Tensor):
                valid_mask = torch.ones(num_envs, dtype=torch.bool, device=env.device)

            # Batch store experiences (vectorized)
            if isinstance(self.orderMem, VectorizedReplayMemory):
                self.orderMem.append_batch(orderState, orderAction, reward, done, next_order_state, valid_mask)
                self.locMem.append_batch(locState, locAction, reward, done, locState, valid_mask)  # locState used temporarily
            else:
                # Compatible with original list format - use vectorized indexing
                orderState_cpu = orderState.cpu()
                locState_cpu = locState.cpu()
                orderAction_cpu = orderAction.cpu()
                locAction_cpu = locAction.cpu()
                reward_cpu = reward.cpu()
                done_cpu = done.cpu()
                valid_cpu = valid_mask.cpu()

                valid_indices = torch.where(valid_cpu)[0]
                for idx in valid_indices:
                    i = idx.item()
                    self.orderMem[i].append(orderState_cpu[i], orderAction_cpu[i], reward_cpu[i], done_cpu[i].item())
                    self.locMem[i].append(locState_cpu[i], locAction_cpu[i], reward_cpu[i], done_cpu[i].item())

            # Handle episode endings (vectorized)
            if done.any():
                done_mask = done

                # Batch record statistics
                done_rewards = env_rewards[done_mask]
                done_steps = env_steps[done_mask]

                episode_rewards.extend(done_rewards.cpu().tolist())
                episode_counter.extend(done_steps.cpu().tolist())

                if 'ratio' in info:
                    ratio = info['ratio']
                    if isinstance(ratio, torch.Tensor):
                        done_ratios = ratio[done_mask]
                        episode_ratio.extend(done_ratios.cpu().tolist())

                # Reset statistics for done environments (vectorized)
                env_rewards = torch.where(done_mask, torch.zeros_like(env_rewards), env_rewards)
                env_steps = torch.where(done_mask, torch.zeros_like(env_steps), env_steps)

                # Reset done environments
                done_env_ids = torch.where(done_mask)[0]
                reset_obs = env.reset(done_env_ids)
                # Update next_order_state for these environments
                next_order_state[done_env_ids] = reset_obs

            # Training update
            if T >= args.learn_start:
                # Batch update priority weight
                if isinstance(self.orderMem, VectorizedReplayMemory):
                    self.orderMem.priority_weight = min(
                        self.orderMem.priority_weight + priority_weight_increase, 1
                    )
                    self.locMem.priority_weight = min(
                        self.locMem.priority_weight + priority_weight_increase, 1
                    )
                else:
                    new_weight = min(self.orderMem[0].priority_weight + priority_weight_increase, 1)
                    for m in self.orderMem:
                        m.priority_weight = new_weight
                    for m in self.locMem:
                        m.priority_weight = new_weight

                # Number of learning iterations - fixed small value to maintain speed
                num_learning_iters = 4

                if T % args.replay_frequency == 0:
                    for _ in range(num_learning_iters):
                        orderLoss = self.orderDQN.learn(self.orderMem)
                        locLoss = self.locDQN.learn(self.locMem)

                if T % args.target_update == 0:
                    self.orderDQN.update_target_net()
                    self.locDQN.update_target_net()
                    print(f"[T={T}] >>> Target network updated <<<")

                if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                    if T % args.checkpoint_interval == 0:
                        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                    self.orderDQN.save(model_save_path, 'orderCheckpoint_{}.pt'.format(sub_time_str))
                    self.locDQN.save(model_save_path, 'locCheckpoint_{}.pt'.format(sub_time_str))

                if T % args.print_log_interval == 0:
                    if isinstance(locLoss, torch.Tensor):
                        self.writer.add_scalar("Training/Value loss", locLoss.mean().item(), T)
                    if isinstance(orderLoss, torch.Tensor):
                        self.writer.add_scalar("Training/Order value loss", orderLoss.mean().item(), T)

            orderState = next_order_state

            # Render (if enabled)
            if hasattr(env, 'render'):
                env.render()

            if len(episode_rewards) != 0:
                self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
            if len(episode_ratio) != 0:
                self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
            if len(episode_counter) != 0:
                self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)
