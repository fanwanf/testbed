import numpy as np
import torch


class ItemCreatorIsaacGym:
    """
    Item creator for Isaac Gym (supports multi-environment parallel)

    Uses PyTorch tensors to store item queues; each environment has an independent queue.
    Fully vectorized implementation without for loops over environments.
    """

    def __init__(self, num_envs, device, max_queue_length=100):
        """
        Args:
            num_envs: number of environments
            device: PyTorch device
            max_queue_length: maximum queue length
        """
        self.num_envs = num_envs
        self.device = device if isinstance(device, torch.device) else torch.device(f'cuda:{device}' if isinstance(device, int) else device)
        self.max_queue_length = max_queue_length

        # Item queue: [num_envs, max_queue_length]
        # -1 indicates an empty slot
        self.item_queue = torch.full((num_envs, max_queue_length), -1,
                                      dtype=torch.int32, device=self.device)

        # Current valid queue length per environment: [num_envs]
        self.queue_lengths = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

    def _to_tensor(self, env_indices):
        """Convert environment indices to tensor"""
        if env_indices is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_indices, int):
            return torch.tensor([env_indices], device=self.device, dtype=torch.long)
        elif not isinstance(env_indices, torch.Tensor):
            return torch.tensor(env_indices, device=self.device, dtype=torch.long)
        return env_indices.to(self.device).long()

    def reset(self, env_indices=None):
        """
        Reset item queue for specified environments (vectorized)

        Args:
            env_indices: environment indices to reset; None resets all environments
        """
        env_indices = self._to_tensor(env_indices)

        # Clear item queue for specified environments
        self.item_queue[env_indices] = -1
        self.queue_lengths[env_indices] = 0

    def generate_item(self, env_indices=None):
        """
        Generate new items for specified environments (subclasses implement the logic)

        Args:
            env_indices: environment indices to generate items for; None means all environments

        Returns:
            new_items: [N] generated item IDs
        """
        pass

    def preview(self, length=1, env_indices=None):
        """
        Preview the first `length` items in the queue for specified environments (vectorized)

        Args:
            length: number of items to preview
            env_indices: environment indices to preview; None means all environments

        Returns:
            items: [N, length] item ID tensor
        """
        env_indices = self._to_tensor(env_indices)

        # Ensure the queue has enough items
        need_generate = env_indices[self.queue_lengths[env_indices] < length]
        while len(need_generate) > 0:
            self.generate_item(need_generate)
            need_generate = env_indices[self.queue_lengths[env_indices] < length]

        # Return the first `length` items for each environment
        return self.item_queue[env_indices, :length].clone()

    def update_item_queue(self, indices, env_indices=None):
        """
        Remove items at specified positions from the queue for given environments (vectorized)

        Implements deletion by shifting subsequent items forward.

        Args:
            indices: [N] position index to remove per environment, or scalar (same position for all)
            env_indices: [N] environment indices to update; None means all environments
        """
        env_indices = self._to_tensor(env_indices)
        N = len(env_indices)

        if isinstance(indices, int):
            indices = torch.full((N,), indices, device=self.device, dtype=torch.long)
        elif not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device, dtype=torch.long)

        # Ensure indices length matches env_indices length
        if len(indices) == 1 and N > 1:
            indices = indices.expand(N)

        # Get current queue lengths
        queue_lens = self.queue_lengths[env_indices]  # [N]

        # Valid deletion mask: queue non-empty and index valid
        valid_mask = (queue_lens > 0) & (indices >= 0) & (indices < queue_lens)

        if not valid_mask.any():
            return

        # Filter valid environments
        valid_envs = env_indices[valid_mask]
        valid_indices = indices[valid_mask]
        valid_lens = queue_lens[valid_mask].long()  # ensure index is long type

        # Vectorized shift: move items after idx one position forward
        for shift in range(self.max_queue_length - 1):
            # Find environments that need shifting at current position
            need_shift = (valid_indices <= shift) & (shift < valid_lens - 1)
            if need_shift.any():
                shift_envs = valid_envs[need_shift]
                self.item_queue[shift_envs, shift] = self.item_queue[shift_envs, shift + 1].clone()

        # Set last valid position to -1
        self.item_queue[valid_envs, valid_lens - 1] = -1

        # Decrease queue length by 1
        self.queue_lengths[valid_envs] -= 1

    def pop_first(self, env_indices=None):
        """
        Pop the first item from the queue for specified environments (vectorized)

        Args:
            env_indices: environment indices to operate on; None means all environments

        Returns:
            items: [N] popped item IDs
        """
        env_indices = self._to_tensor(env_indices)

        # Get the first item
        items = self.item_queue[env_indices, 0].clone()

        # Remove the first item
        self.update_item_queue(0, env_indices)

        return items


class RandomItemCreatorIsaacGym(ItemCreatorIsaacGym):
    """
    Fully random item generation (Isaac Gym version, vectorized)
    """

    def __init__(self, item_set, num_envs, device, max_queue_length=100):
        """
        Args:
            item_set: set of available item IDs (numpy array or list)
            num_envs: number of environments
            device: PyTorch device
            max_queue_length: maximum queue length
        """
        super().__init__(num_envs, device, max_queue_length)
        self.item_set = torch.tensor(item_set, dtype=torch.int32, device=self.device)
        self.num_items = len(item_set)
        print(f"RandomItemCreatorIsaacGym: {self.num_items} items available")

    def generate_item(self, env_indices=None):
        """
        Randomly generate one item for specified environments (vectorized)

        Args:
            env_indices: environment indices to generate items for; None means all environments

        Returns:
            new_items: [N] generated item IDs
        """
        env_indices = self._to_tensor(env_indices)
        N = len(env_indices)

        # Randomly select item indices
        random_indices = torch.randint(0, self.num_items, (N,), device=self.device)
        new_items = self.item_set[random_indices]

        # Get current queue lengths
        queue_lens = self.queue_lengths[env_indices]  # [N]

        # Filter environments that can accept new items (queue not full)
        can_add = queue_lens < self.max_queue_length

        if can_add.any():
            add_envs = env_indices[can_add]
            add_items = new_items[can_add]
            add_positions = queue_lens[can_add].long()  # ensure index is long type

            # Vectorized add item to queue
            self.item_queue[add_envs, add_positions] = add_items
            self.queue_lengths[add_envs] += 1

        return new_items


class RandomInstanceCreatorIsaacGym(ItemCreatorIsaacGym):
    """
    Two-level random item generation based on instances (Isaac Gym version, vectorized)
    First randomly select a shape type, then randomly select an instance from that type.
    """

    def __init__(self, item_set, dicPath, num_envs, device, max_queue_length=100):
        """
        Args:
            item_set: set of available item IDs
            dicPath: dictionary mapping ID to shape name
            num_envs: number of environments
            device: PyTorch device
            max_queue_length: maximum queue length
        """
        super().__init__(num_envs, device, max_queue_length)

        # Build reverse dictionary: shape type -> list of item IDs
        self.inverse_dict = {}
        for k in dicPath.keys():
            shape_type = dicPath[k][0:-6]  # remove suffix to get shape type
            if shape_type not in self.inverse_dict:
                self.inverse_dict[shape_type] = []
            self.inverse_dict[shape_type].append(k)

        # Convert dictionary to tensor form for GPU operations
        self.shape_types = list(self.inverse_dict.keys())
        self.num_types = len(self.shape_types)

        # Precompute item lists per type (padded to equal length)
        max_items_per_type = max(len(items) for items in self.inverse_dict.values())
        self.type_items = torch.full((self.num_types, max_items_per_type), -1,
                                      dtype=torch.int32, device=self.device)
        self.type_item_counts = torch.zeros(self.num_types, dtype=torch.int32, device=self.device)

        for i, shape_type in enumerate(self.shape_types):
            items = self.inverse_dict[shape_type]
            self.type_items[i, :len(items)] = torch.tensor(items, dtype=torch.int32, device=self.device)
            self.type_item_counts[i] = len(items)

        print(f"RandomInstanceCreatorIsaacGym: {self.num_types} shape types")

    def generate_item(self, env_indices=None):
        """
        Generate items for specified environments (two-level random, vectorized)

        Args:
            env_indices: environment indices to generate items for; None means all environments

        Returns:
            new_items: [N] generated item IDs
        """
        env_indices = self._to_tensor(env_indices)
        N = len(env_indices)

        # Level 1 random: select shape type
        type_indices = torch.randint(0, self.num_types, (N,), device=self.device)

        # Get item count per selected type
        item_counts = self.type_item_counts[type_indices]  # [N]

        # Level 2 random: select specific item from chosen type (vectorized)
        # Generate random numbers in [0, 1), multiply by item count, and take integer part
        random_offsets = (torch.rand(N, device=self.device) * item_counts.float()).long()
        # Use torch.minimum instead of clamp (clamp doesn't support mixed scalar/tensor)
        random_offsets = torch.minimum(random_offsets, item_counts - 1)
        random_offsets = torch.maximum(random_offsets, torch.zeros_like(random_offsets))

        # Get item ID
        new_items = self.type_items[type_indices, random_offsets]

        # Get current queue lengths
        queue_lens = self.queue_lengths[env_indices]

        # Filter environments that can accept new items
        can_add = queue_lens < self.max_queue_length

        if can_add.any():
            add_envs = env_indices[can_add]
            add_items = new_items[can_add]
            add_positions = queue_lens[can_add].long()  # ensure index is long type

            self.item_queue[add_envs, add_positions] = add_items
            self.queue_lengths[add_envs] += 1

        return new_items


class RandomCateCreatorIsaacGym(ItemCreatorIsaacGym):
    """
    Two-level random item generation by category (Isaac Gym version, vectorized)
    First randomly select a category, then randomly select an item from that category.
    """

    def __init__(self, item_set, dicPath, num_envs, device, max_queue_length=100):
        """
        Args:
            item_set: set of available item IDs
            dicPath: dictionary mapping ID to shape name (format: category/item_name)
            num_envs: number of environments
            device: PyTorch device
            max_queue_length: maximum queue length
        """
        super().__init__(num_envs, device, max_queue_length)

        # Categories and their weights
        self.categories = {'objects': 0.34, 'concave': 0.33, 'board': 0.33}

        # Build category dictionary: category -> list of item IDs
        self.obj_cates = {key: [] for key in self.categories.keys()}

        for k, item in zip(dicPath.keys(), dicPath.values()):
            parts = item.split('/')
            if len(parts) >= 2:
                cate = parts[0]
                if cate in self.obj_cates:
                    self.obj_cates[cate].append(k)

        # Convert dictionary to tensor form
        self.category_names = list(self.categories.keys())
        self.num_categories = len(self.category_names)

        # Precompute item lists per category
        max_items_per_cate = max(len(items) for items in self.obj_cates.values()) if self.obj_cates else 1
        if max_items_per_cate == 0:
            max_items_per_cate = 1

        self.cate_items = torch.full((self.num_categories, max_items_per_cate), -1,
                                      dtype=torch.int32, device=self.device)
        self.cate_item_counts = torch.zeros(self.num_categories, dtype=torch.int32, device=self.device)

        for i, cate in enumerate(self.category_names):
            items = self.obj_cates[cate]
            if len(items) > 0:
                self.cate_items[i, :len(items)] = torch.tensor(items, dtype=torch.int32, device=self.device)
                self.cate_item_counts[i] = len(items)

        # Valid categories (categories with items)
        self.valid_cates = torch.where(self.cate_item_counts > 0)[0]
        self.num_valid_cates = len(self.valid_cates)

        print(f"RandomCateCreatorIsaacGym: {self.num_valid_cates} categories with items")

    def generate_item(self, env_indices=None):
        """
        Generate items for specified environments (category-based random, vectorized)

        Args:
            env_indices: environment indices to generate items for; None means all environments

        Returns:
            new_items: [N] generated item IDs
        """
        env_indices = self._to_tensor(env_indices)
        N = len(env_indices)

        if self.num_valid_cates == 0:
            return torch.full((N,), -1, dtype=torch.int32, device=self.device)

        # Level 1 random: select valid category
        cate_indices = self.valid_cates[torch.randint(0, self.num_valid_cates, (N,), device=self.device)]

        # Get item count per selected category
        item_counts = self.cate_item_counts[cate_indices]  # [N]

        # Level 2 random: select specific item from chosen category (vectorized)
        random_offsets = (torch.rand(N, device=self.device) * item_counts.float()).long()
        # Use torch.minimum/maximum instead of clamp
        max_offset = torch.clamp(item_counts - 1, min=0)
        random_offsets = torch.minimum(random_offsets, max_offset)
        random_offsets = torch.maximum(random_offsets, torch.zeros_like(random_offsets))

        # Get item ID
        new_items = self.cate_items[cate_indices, random_offsets]

        # Get current queue lengths
        queue_lens = self.queue_lengths[env_indices]

        # Filter environments that can accept new items
        can_add = queue_lens < self.max_queue_length

        if can_add.any():
            add_envs = env_indices[can_add]
            add_items = new_items[can_add]
            add_positions = queue_lens[can_add].long()  # ensure index is long type

            self.item_queue[add_envs, add_positions] = add_items
            self.queue_lengths[add_envs] += 1

        return new_items


class FixedSequenceCreatorIsaacGym(ItemCreatorIsaacGym):
    """
    Fixed-sequence item generation (Isaac Gym version, vectorized)
    All environments use the same fixed item sequence for debugging and training validation.
    """

    def __init__(self, item_set, num_envs, device, max_queue_length=100, sequence_length=50):
        """
        Args:
            item_set: set of available item IDs
            num_envs: number of environments
            device: PyTorch device
            max_queue_length: maximum queue length
            sequence_length: fixed sequence length
        """
        super().__init__(num_envs, device, max_queue_length)
        self.item_set = torch.tensor(item_set, dtype=torch.int32, device=self.device)
        self.num_items = len(item_set)
        self.sequence_length = sequence_length

        # Generate fixed sequence (on CPU to ensure reproducibility)
        # Save current random state
        cpu_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state()

        # Set fixed seed
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Generate indices on CPU, then move to GPU (ensures reproducibility)
        # Use shuffle instead of randint to ensure each type appears once
        # This allows using item_type_ids + 5 directly as actor index
        if sequence_length <= self.num_items:
            # Randomly shuffle all items and take the first sequence_length
            perm = torch.randperm(self.num_items)[:sequence_length]
            self.fixed_sequence = self.item_set[perm.to(self.device)]
        else:
            # If sequence length exceeds number of items, repeat (may cause actor conflicts)
            print(f"Warning: sequence_length ({sequence_length}) > num_items ({self.num_items}), may cause actor conflicts")
            indices = torch.randint(0, self.num_items, (sequence_length,))
            self.fixed_sequence = self.item_set[indices.to(self.device)]

        # Restore random state
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state)

        # Current position in sequence per environment
        self.sequence_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        print(f"FixedSequenceCreatorIsaacGym: fixed sequence length {sequence_length}, {self.num_items} item types")
        print(f"  Full fixed sequence: {self.fixed_sequence.tolist()}")

    def reset(self, env_indices=None, sync_all=False):
        """
        Reset specified environments (vectorized)

        Args:
            env_indices: environments to reset
            sync_all: if True, reset sequence position for all environments (for full sync)
        """
        env_indices = self._to_tensor(env_indices)
        super().reset(env_indices)
        # Reset sequence position
        if sync_all:
            # Sync mode: reset all environments' sequence positions
            self.sequence_indices[:] = 0
        else:
            # Independent mode: only reset specified environments' sequence positions
            self.sequence_indices[env_indices] = 0

    def generate_item(self, env_indices=None):
        """
        Generate items from fixed sequence for specified environments (vectorized)
        """
        env_indices = self._to_tensor(env_indices)
        N = len(env_indices)

        # Get current sequence position per environment
        seq_idxs = self.sequence_indices[env_indices]

        # Cycle through sequence
        seq_idxs = seq_idxs % self.sequence_length

        # Get item IDs
        new_items = self.fixed_sequence[seq_idxs]

        # Get current queue lengths
        queue_lens = self.queue_lengths[env_indices]

        # Filter environments that can accept new items
        can_add = queue_lens < self.max_queue_length

        if can_add.any():
            add_envs = env_indices[can_add]
            add_items = new_items[can_add]
            add_positions = queue_lens[can_add].long()  # ensure index is long type

            self.item_queue[add_envs, add_positions] = add_items
            self.queue_lengths[add_envs] += 1

        # Advance sequence position
        self.sequence_indices[env_indices] += 1

        return new_items


class LoadItemCreatorIsaacGym(ItemCreatorIsaacGym):
    """
    Load predefined item trajectories (Isaac Gym version, vectorized)
    Each environment can independently load different trajectories.
    """

    def __init__(self, data_name, num_envs, device, max_queue_length=100):
        """
        Args:
            data_name: trajectory data file path
            num_envs: number of environments
            device: PyTorch device
            max_queue_length: maximum queue length
        """
        super().__init__(num_envs, device, max_queue_length)

        self.data_name = data_name
        print(f"Load dataset: {data_name}")

        # Load trajectory data
        self.item_trajs = torch.load(self.data_name, weights_only=False)
        self.traj_nums = len(self.item_trajs)

        # Current trajectory index and item index per environment
        self.traj_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.item_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Preprocess trajectory data: convert to tensors
        max_traj_len = max(len(traj) for traj in self.item_trajs) if self.item_trajs else 1
        self.traj_data = torch.full((self.traj_nums, max_traj_len), -1,
                                     dtype=torch.int32, device=self.device)
        self.traj_lengths = torch.zeros(self.traj_nums, dtype=torch.long, device=self.device)

        for i, traj in enumerate(self.item_trajs):
            traj_len = len(traj)
            # Handle None values in trajectory
            traj_tensor = []
            for item in traj:
                if item is None:
                    traj_tensor.append(-1)
                else:
                    traj_tensor.append(item)
            self.traj_data[i, :traj_len] = torch.tensor(traj_tensor, dtype=torch.int32, device=self.device)
            self.traj_lengths[i] = traj_len

        print(f"LoadItemCreatorIsaacGym: {self.traj_nums} trajectories loaded")

    def reset(self, env_indices=None, traj_indices=None):
        """
        Reset item queue for specified environments and set new trajectories (vectorized)

        Args:
            env_indices: environment indices to reset
            traj_indices: trajectory index for each environment; None means increment
        """
        env_indices = self._to_tensor(env_indices)

        # Call parent class to reset queue
        super().reset(env_indices)

        # Update trajectory indices
        if traj_indices is None:
            # Increment trajectory index
            self.traj_indices[env_indices] = (self.traj_indices[env_indices] + 1) % self.traj_nums
        else:
            if isinstance(traj_indices, int):
                traj_indices = torch.full((len(env_indices),), traj_indices,
                                          dtype=torch.long, device=self.device)
            elif not isinstance(traj_indices, torch.Tensor):
                traj_indices = torch.tensor(traj_indices, dtype=torch.long, device=self.device)
            self.traj_indices[env_indices] = traj_indices

        # Reset item indices
        self.item_indices[env_indices] = 0

    def generate_item(self, env_indices=None):
        """
        Load the next item from the trajectory for specified environments (vectorized)

        Args:
            env_indices: environment indices to generate items for; None means all environments

        Returns:
            new_items: [N] generated item IDs
        """
        env_indices = self._to_tensor(env_indices)
        N = len(env_indices)

        # Get trajectory index and item index per environment
        traj_idxs = self.traj_indices[env_indices]  # [N]
        item_idxs = self.item_indices[env_indices]  # [N]
        traj_lens = self.traj_lengths[traj_idxs]     # [N]
        queue_lens = self.queue_lengths[env_indices] # [N]

        # Check if item index is within trajectory range
        in_range = item_idxs < traj_lens

        # Get item IDs (vectorized indexing)
        # Use clamp to prevent index out of bounds
        safe_item_idxs = torch.clamp(item_idxs, 0, self.traj_data.shape[1] - 1)
        new_items = self.traj_data[traj_idxs, safe_item_idxs]

        # Set items beyond trajectory range to -1
        new_items = torch.where(in_range, new_items, torch.tensor(-1, dtype=torch.int32, device=self.device))

        # Filter environments that can accept new items
        can_add = queue_lens < self.max_queue_length

        if can_add.any():
            add_envs = env_indices[can_add]
            add_items = new_items[can_add]
            add_positions = queue_lens[can_add].long()  # ensure index is long type

            self.item_queue[add_envs, add_positions] = add_items
            self.queue_lengths[add_envs] += 1

        # Advance item index
        self.item_indices[env_indices] += 1

        return new_items
