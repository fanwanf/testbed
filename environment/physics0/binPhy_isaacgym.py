"""
Isaac Gym version of the 3D bin packing problem environment.
Fully vectorized implementation with support for multi-environment parallel simulation.
"""
import numpy as np
import os

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import to_torch, quaternion_to_matrix, tensor_clamp, quat_diff_rad

import torch
import time
import gym
import numpy as np
from torch import load

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from tools import getRotationMatrix

import transforms3d
from .Interface_isaacgym import IsaacGymInterface
from .Interface import Interface
from .IRcreator import RandomItemCreator, LoadItemCreator, RandomInstanceCreator, RandomCateCreator
from .IRcreator_isaacgym import (
    RandomItemCreatorIsaacGym,
    LoadItemCreatorIsaacGym,
    RandomInstanceCreatorIsaacGym,
    RandomCateCreatorIsaacGym,
    FixedSequenceCreatorIsaacGym
)
from .space_isaacgym import SpaceIsaacGym
from .cvTools_isaacgym import getConvexHullActions
from arguments import get_args


class PackingGame:
    """
    Multi-environment parallel 3D bin packing reinforcement learning environment.

    Features:
    - Fully vectorized operations, no for-loop environment iteration
    - Supports Isaac Gym physics simulation
    - Logic consistent with the original PyBullet version
    """

    def __init__(self, args, num_envs=4, sim_device=0, headless=False):
        """
        Initialize the PackingGame environment.

        Args:
            args: Argument object containing all configuration parameters
            num_envs: Number of parallel environments
            sim_device: Simulation device (GPU ID or 'cpu')
            headless: Whether to run in headless mode
        """
        # ==================== 1. Parse Arguments ====================
        self.args = args
        args_dict = vars(args) if hasattr(args, '__dict__') else args

        # Basic configuration parameters
        self.resolutionAct = args_dict['resolutionA']      # Action resolution
        self.resolutionH = args_dict['resolutionH']        # Height map resolution
        self.bin_dimension = args_dict['bin_dimension']    # Container dimensions [x, y, z]
        self.scale = args_dict['scale']                    # Scale factor
        self.objPath = args_dict['objPath']                # Object model path
        self.meshScale = args_dict['meshScale']            # Mesh scale
        self.shapeDict = args_dict['shapeDict']            # Shape dictionary
        self.infoDict = args_dict['infoDict']              # Object info dictionary
        self.dicPath = load(args_dict['dicPath'])          # id2shape mapping
        self.ZRotNum = args_dict['ZRotNum']                # Number of Z-axis rotations
        self.heightMapPre = args_dict['heightMap']         # Whether to use height map
        self.globalView = not args_dict.get('only_simulate_current', False)  # Default True (global view)
        self.selectedAction = args_dict['selectedAction']  # Number of candidate actions to select
        self.bufferSize = args_dict['bufferSize']          # Item buffer size
        self.chooseItem = self.bufferSize > 1              # Whether to choose items
        self.simulation = args_dict.get('simulation', True)  # Default True (accurate simulation)
        self.evaluate = args_dict['evaluate']              # Whether in evaluation mode
        self.maxBatch = args_dict['maxBatch']              # Maximum batch size
        self.heightResolution = args_dict['resolutionZ']   # Height resolution
        self.dataSample = args_dict['dataSample']          # Data sampling method
        self.dataname = args_dict['test_name']             # Test data name
        self.visual = args_dict['visual']                  # Whether to visualize
        self.non_blocking = args_dict['non_blocking']      # Non-blocking simulation
        self.time_limit = args_dict['time_limit']          # Time limit

        # Environment parameters
        self.num_envs = num_envs
        self.device = f'cuda:{sim_device}' if isinstance(sim_device, int) else sim_device
        self.sim_device = sim_device
        self.headless = headless

        # ==================== 2. Initialize Isaac Gym Interface ====================
        self.interface = IsaacGymInterface(
            sim_device=sim_device,
            graphics_device_id=0 if not headless else -1,
            headless=headless,
            bin=self.bin_dimension,
            scale=self.scale,
            meshScale=self.meshScale,  # Object scale factor
            foldername=self.objPath,
            num_envs=num_envs,
            maxBatch=self.maxBatch
        )

        # ==================== 3. Initialize Space Manager ====================
        # Compute action space grid dimensions
        bin_dim_tensor = torch.tensor(self.bin_dimension, device=self.device, dtype=torch.float32)
        self.rangeX_A = int(torch.ceil(bin_dim_tensor[0] / self.resolutionAct).item())
        self.rangeY_A = int(torch.ceil(bin_dim_tensor[1] / self.resolutionAct).item())

        self.space = SpaceIsaacGym(
            bin_dimension=self.bin_dimension,
            resolutionAct=self.resolutionAct,
            resolutionH=self.resolutionH,
            boxPack=False,
            ZRotNum=self.ZRotNum,
            shotInfo=args_dict.get('shotInfo', None),
            scale=self.scale,
            device=self.device,
            num_envs=self.num_envs,
            args=self.args
        )

        # ==================== 4. Initialize Item Creator ====================
        self.num_item_types = len(self.shapeDict.keys())
        item_set = torch.arange(self.num_item_types, device=self.device)

        if self.evaluate and self.dataname is not None:
            self.item_creator = LoadItemCreatorIsaacGym(
                data_name=self.dataname,
                num_envs=self.num_envs,
                device=self.device
            )
        elif self.dataSample == 'fixed':
            # Fixed sequence mode - used for debugging and training validation
            # Sequence length cannot exceed item count, otherwise actor conflicts occur (same item placed repeatedly)
            max_seq_len = min(50, len(item_set))
            self.item_creator = FixedSequenceCreatorIsaacGym(
                item_set=item_set,
                num_envs=self.num_envs,
                device=self.device,
                sequence_length=max_seq_len
            )
        elif self.dataSample == 'category':
            self.item_creator = RandomCateCreatorIsaacGym(
                item_set=item_set,
                dicPath=self.dicPath,
                num_envs=self.num_envs,
                device=self.device
            )
        elif self.dataSample == 'instance':
            self.item_creator = RandomInstanceCreatorIsaacGym(
                item_set=item_set,
                dicPath=self.dicPath,
                num_envs=self.num_envs,
                device=self.device
            )
        else:
            # Default: pose sampling
            self.item_creator = RandomItemCreatorIsaacGym(
                item_set=item_set,
                num_envs=self.num_envs,
                device=self.device
            )

        # ==================== 5. Initialize State Tensors ====================
        # Item info tensor [num_envs, max_items, 9]
        # 9 dims: [item_id, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, placed_flag]
        self.max_items = 1000
        self.item_vec = torch.zeros((self.num_envs, self.max_items, 9), device=self.device)

        # Next item info vector [num_envs, 9]
        self.next_item_vec = torch.zeros((self.num_envs, 9), device=self.device)

        # Item counter [num_envs]
        self.item_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Placed items mask [num_envs, num_item_types]
        self.placed_items_mask = torch.zeros(
            self.num_envs, self.num_item_types, device=self.device, dtype=torch.bool
        )

        # ==================== 6. Initialize Rotation Transforms ====================
        self.rotNum = self.ZRotNum
        self._init_transformations()

        # ==================== 7. Initialize Item Info ====================
        self._init_items_volume()
        self._init_item_extents()

        # ==================== 8. Initialize Action and Observation Spaces ====================
        self.act_len = self.selectedAction if not self.chooseItem else self.bufferSize
        self._init_observation_space()

        # ==================== 9. Store Candidate Actions and Masks ====================
        self.stored_naiveMask = torch.zeros(
            (self.num_envs, self.rotNum, self.rangeX_A, self.rangeY_A),
            device=self.device, dtype=torch.int32
        )
        self.candidates = None

        # ==================== 10. Other State Variables ====================
        # Placement tolerance
        # Original tolerance = 0
        # Use small tolerance for direct placement to avoid edge-grazing issues during free fall
        # PhysX depenetration will automatically resolve minor penetrations
        # Increase tolerance to avoid edge collisions:
        # - heightmap is updated with the planned position, but after physics simulation the object may have minor offset
        # - the first object may not lie completely flat, actual height may be higher than what heightmap records
        # - sufficient margin is needed for the object to "jump over" these minor differences
        self.tolerance = 0.01  # 10mm tolerance

        # Episode counter
        self.episodeCounter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.updatePeriod = 500

        # Total packed volume per environment (used for terminal packing efficiency reward)
        self.total_packed_volume = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Hierarchical action related
        self.orderAction = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.hierachical = False

        # Precomputed corner offsets (used for AABB computation)
        self._init_corner_offsets()

        # ID to name mapping (used for object loading)
        self.id2name = {idx: name[:-4] for idx, name in self.dicPath.items()}  # Strip .obj suffix

        print(f"PackingGame initialized:")
        print(f"  Number of environments: {self.num_envs}")
        print(f"  Number of item types: {self.num_item_types}")
        print(f"  Action space size: {self.act_len}")
        print(f"  Height map dimensions: [{self.space.rangeX_C}, {self.space.rangeY_C}]")
        print(f"  Action grid dimensions: [{self.rangeX_A}, {self.rangeY_A}]")

    def _init_items_volume(self):
        """Initialize item volume tensor (vectorized)"""
        # Compute number of actors
        num_actors_per_env = len(self.interface.assets) + 5

        # Initialize volume tensor
        self.items_volume = torch.zeros(num_actors_per_env, device=self.device, dtype=torch.float32)

        # Batch-retrieve volumes (actor_idx = 5 + item_type)
        volumes = torch.tensor(
            [self.infoDict[i][0]['volume'] for i in range(self.num_item_types)],
            device=self.device, dtype=torch.float32
        )
        self.items_volume[5:5 + self.num_item_types] = volumes

        # Container volume
        self.bin_volume = torch.prod(
            torch.tensor(self.bin_dimension, device=self.device, dtype=torch.float32)
                )

    def _init_transformations(self):
        """Initialize rotation transform matrices and quaternions (vectorized)"""
        # Use getRotationMatrix to obtain rotation matrices
        DownFaceList, ZRotList = getRotationMatrix(1, self.ZRotNum)

        # Batch-compute quaternions
        transformation_list = []
        for d in DownFaceList:
            for z in ZRotList:
                rot_matrix = np.dot(z, d)[0:3, 0:3]
                quat = transforms3d.quaternions.mat2quat(rot_matrix)
                transformation_list.append([quat[1], quat[2], quat[3], quat[0]])  # xyzw

        self.transformation = torch.tensor(transformation_list, device=self.device, dtype=torch.float32)

        # Vectorized computation of rotation matrices
        self.rotation_matrices = self._quaternion_to_rotation_matrix(self.transformation)

        # Store rotation angle mapping (used for quaternion computation in test code)
        # getRotationMatrix returns rotations in order [0, 90, 180, 270, 45, 135, 225, 315]
        self.rot_angle_map = torch.tensor(
            [0, 90, 180, 270, 45, 135, 225, 315][:self.ZRotNum],
            device=self.device, dtype=torch.float32
        )

    def _quaternion_to_rotation_matrix(self, quats):
        """
        Batch-compute rotation matrices from quaternions (fully vectorized).

        Args:
            quats: [N, 4] quaternions (x, y, z, w)

        Returns:
            rotation_matrices: [N, 3, 3] rotation matrices
        """
        x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        N = len(quats)
        rotation_matrices = torch.zeros((N, 3, 3), device=self.device, dtype=torch.float32)
        rotation_matrices[:, 0, 0] = 1 - 2 * (yy + zz)
        rotation_matrices[:, 0, 1] = 2 * (xy - wz)
        rotation_matrices[:, 0, 2] = 2 * (xz + wy)
        rotation_matrices[:, 1, 0] = 2 * (xy + wz)
        rotation_matrices[:, 1, 1] = 1 - 2 * (xx + zz)
        rotation_matrices[:, 1, 2] = 2 * (yz - wx)
        rotation_matrices[:, 2, 0] = 2 * (xz - wy)
        rotation_matrices[:, 2, 1] = 2 * (yz + wx)
        rotation_matrices[:, 2, 2] = 1 - 2 * (xx + yy)

        return rotation_matrices

    def _init_item_extents(self):
        """Precompute extents for all items under all rotations (fully vectorized)"""
        # Get mesh bounds: [num_actors, 2, 4]
        mesh_bounds = self.interface.mesh_bounds
        num_actors = mesh_bounds.shape[0]

        # Only process item actors (index >= 5)
        num_items = min(num_actors - 5, self.num_item_types)
        if num_items <= 0:
            self.item_extents = torch.zeros((1, self.rotNum, 3), device=self.device)
            return

        # Compute original extents: [num_items, 3]
        original_extents = mesh_bounds[5:5+num_items, 1, 0:3] - mesh_bounds[5:5+num_items, 0, 0:3]

        # Get absolute values of rotation matrices: [rotNum, 3, 3]
        abs_rot_matrices = torch.abs(self.rotation_matrices)

        # Batch-compute rotated extents
        # [1, rotNum, 3, 3] @ [num_items, 1, 3, 1] -> [num_items, rotNum, 3, 1]
        abs_rot_expanded = abs_rot_matrices.unsqueeze(0)  # [1, rotNum, 3, 3]
        extents_expanded = original_extents.unsqueeze(1).unsqueeze(-1)  # [num_items, 1, 3, 1]

        rotated_extents = torch.matmul(abs_rot_expanded, extents_expanded)
        self.item_extents = rotated_extents.squeeze(-1)  # [num_items, rotNum, 3]

    def _init_corner_offsets(self):
        """Precompute corner offset matrix for AABB computation"""
        # 8 corner combinations: (0,0,0), (0,0,1), (0,1,0), ... (1,1,1)
        self.corner_signs = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], device=self.device, dtype=torch.float32)

    def _init_observation_space(self):
        """Initialize observation space"""
        # Compute observation length
        if not self.chooseItem:
            # Item info (9) + candidate actions/mask
            self.obs_len = 9  # next_item_vec
            if self.selectedAction:
                self.obs_len += self.selectedAction * 5  # candidate points [ROT, X, Y, H, V]
            else:
                self.obs_len += self.act_len
        else:
            self.obs_len = self.bufferSize

        # Height map
        if self.heightMapPre:
            heightmap_size = self.space.rangeX_C * self.space.rangeY_C
            self.obs_len += heightmap_size

        # Create gym spaces
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=float(self.bin_dimension[2]),
            shape=(self.obs_len,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.act_len)

    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def close(self):
        """Close the environment"""
        self.interface.close()

    def reset(self, envs_index=None, data_index=None):
        """
        Reset specified environments (fully vectorized).

        Args:
            envs_index: Environment indices to reset
                       - None: reset all environments
                       - int: reset a single environment
                       - list/tensor: reset multiple specified environments
            data_index: Data index (used by LoadItemCreator to load specific sequences)
                       - None: generate randomly
                       - int/tensor: specify data index

        Returns:
            observation: [num_reset_envs, obs_len] observation after reset
        """
        # ==================== 1. Handle Environment Indices ====================
        if envs_index is None:
            envs_index = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(envs_index, int):
            envs_index = torch.tensor([envs_index], device=self.device, dtype=torch.long)
        elif isinstance(envs_index, (list, np.ndarray)):
            envs_index = torch.tensor(envs_index, device=self.device, dtype=torch.long)
        else:
            envs_index = envs_index.to(self.device).long()

        num_reset = len(envs_index)

        # ==================== 2. Update Episode Counter ====================
        self.episodeCounter[envs_index] = (self.episodeCounter[envs_index] + 1) % self.updatePeriod

        # ==================== 3. Reset Space Manager ====================
        self.space.reset(envs_index)

        # ==================== 4. Reset Physics Interface ====================
        self.interface.reset(envs_index)

        # Important: run a few simulation steps to ensure objects move to initial positions
        for _ in range(3):
            self.interface.step_simulation()

        # ==================== 5. Reset Item Creator ====================
        # LoadItemCreatorIsaacGym supports traj_indices parameter; other creators do not
        if hasattr(self.item_creator, 'traj_indices'):
            self.item_creator.reset(envs_index, data_index)
        else:
            self.item_creator.reset(envs_index)

        # ==================== 6. Vectorized Reset of State Tensors ====================
        # Item info
        self.item_vec[envs_index] = 0
        self.next_item_vec[envs_index] = 0

        # Item count
        self.item_idx[envs_index] = 0

        # Placed items mask
        self.placed_items_mask[envs_index] = False

        # Candidate action mask
        self.stored_naiveMask[envs_index] = 0

        # Hierarchical action
        self.orderAction[envs_index] = 0

        # Packed volume accumulator (also reset in step() when done, but clear here for safety)
        self.total_packed_volume[envs_index] = 0.0

        # ==================== 7. Generate Initial Observation ====================
        # Call cur_observation directly, passing environment indices
        observation = self.cur_observation(genItem=True, envs_index=envs_index)

        return observation

    def get_ratio(self):
        """Compute space utilization ratio (vectorized, fully on GPU)"""
        # Use fixed capacity to avoid .item() calls from dynamic sizing
        max_capacity = self.item_vec.shape[1]

        # Create fixed-size indices (pre-allocated, avoids re-creation each time)
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, max_capacity)
        item_indices = torch.arange(max_capacity, device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        # Validity mask (item_indices < item_idx means the item has been placed)
        valid_mask = item_indices < self.item_idx.unsqueeze(1)

        # Get item type IDs (item_vec[..., 0] stores item type IDs 0-22)
        item_ids = torch.where(
            valid_mask,
            self.item_vec[batch_indices, item_indices, 0].long(),
            torch.zeros_like(item_indices)
        )

        # Compute volume (actor_idx = item_id + 5)
        volumes = self.items_volume[item_ids + 5]
        masked_volumes = torch.where(valid_mask, volumes, torch.zeros_like(volumes))
        total_volume = torch.sum(masked_volumes, dim=1)
        bin_volume = torch.prod(torch.tensor(self.bin_dimension, device=self.device))

        return total_volume / bin_volume

    def get_item_ratio(self, next_item_id):
        """Compute the ratio of item volume to container volume"""
        # next_item_id is the item type ID (0-22), actor_idx = 5 + item_type
        return self.items_volume[next_item_id + 5] / torch.prod(
            torch.tensor(self.bin_dimension, device=self.device)
        )

    def cur_observation(self, genItem=True, draw=False, envs_index=None):
        """
        Get current observation (fully vectorized).

        Args:
            genItem: Whether to generate a new item ID
            draw: Whether to draw (kept for compatibility with original version)
            envs_index: Environment indices to get observations for
                       - None: all environments
                       - tensor/list: specified environments

        Returns:
            observation: [N, obs_len] observation tensor, N is the number of environments
        """
        # Handle environment indices
        if envs_index is None:
            envs_index = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(envs_index, torch.Tensor):
            envs_index = torch.tensor(envs_index, device=self.device, dtype=torch.long)

        num_envs_selected = len(envs_index)

        # Get positions and orientations for all environments (used to update item_vec)
        positions, orientations, valid_mask = self.interface.getAllPositionAndOrientation(inner=False)

        # Vectorized update of item_vec
        if positions is not None:
            self._update_item_vec_vectorized(envs_index, positions, orientations)

        if not self.chooseItem:
            # Generate next item
            if genItem:
                next_item_ids_all = self.item_creator.preview(length=1).squeeze(-1)  # [num_envs]
                next_item_ids = next_item_ids_all[envs_index]  # [N]
                # Filter already-placed item types to avoid re-using the same actor
                next_item_ids = self._handle_placed_items(envs_index, next_item_ids)
            else:
                next_item_ids = self.next_item_vec[envs_index, 0].long()

            self.next_item_vec[envs_index, 0] = next_item_ids.float()

            # Populate fields 1-7 with next item's geometric properties (normalized by bin dims):
            #   [1-3]: bounding box (w, d, h) for rotation 0  / bin_dimension
            #   [4-6]: bounding box (w, d, h) for rotation 1  / bin_dimension  (zeros if rotNum < 2)
            #   [7]:   volume ratio = item_volume / bin_volume
            valid_geom = next_item_ids >= 0
            if valid_geom.any():
                valid_envs_g = envs_index[valid_geom]
                valid_ids_g  = next_item_ids[valid_geom].long()
                bin_dim = torch.tensor(self.bin_dimension, device=self.device, dtype=torch.float32)
                self.next_item_vec[valid_envs_g, 1:4] = self.space.boundingSize[valid_ids_g, 0, :] / bin_dim
                if self.rotNum >= 2:
                    self.next_item_vec[valid_envs_g, 4:7] = self.space.boundingSize[valid_ids_g, 1, :] / bin_dim
                self.next_item_vec[valid_envs_g, 7] = self.items_volume[valid_ids_g + 5] / self.bin_volume
            if (~valid_geom).any():
                self.next_item_vec[envs_index[~valid_geom], 1:] = 0.0

            # Compute possible placement positions (needs to be computed for all environments)
            next_item_ids_all = self.next_item_vec[:, 0].long()
            naiveMask = self.space.get_possible_position(next_item_ids_all, device=self.device)
            self.stored_naiveMask = naiveMask

            # Extract data for specified environments
            naiveMask_selected = naiveMask[envs_index]
            next_item_vec_selected = self.next_item_vec[envs_index]
            heightmapC_selected = self.space.heightmapC[envs_index]
            posZvalid_selected = self.space.posZvalid[envs_index]

            # Build observation
            result = next_item_vec_selected.view(num_envs_selected, -1)

            if not self.selectedAction:
                result = torch.cat([
                    next_item_vec_selected.view(num_envs_selected, -1),
                    naiveMask_selected.view(num_envs_selected, -1)
                ], dim=1)

            if self.heightMapPre:
                result = torch.cat([
                    result,
                    heightmapC_selected.view(num_envs_selected, -1)
                ], dim=1)

            if self.selectedAction:
                # Compute candidate actions
                candidates = getConvexHullActions(
                    posZvalid_selected,
                    naiveMask_selected,
                    self.heightResolution,
                    self.selectedAction,
                    self.bin_dimension[-1],
                    self.device
                )

                # Update global candidates
                if self.candidates is None:
                    self.candidates = torch.zeros(
                        self.num_envs, self.selectedAction, 5,
                        device=self.device, dtype=torch.float32
                    )
                self.candidates[envs_index] = candidates

                result = torch.cat([
                    candidates.view(num_envs_selected, -1),
                    result
                ], dim=1)

            return result

        else:
            # chooseItem mode
            next_k_items = self.item_creator.preview(length=self.bufferSize)
            next_k_items_selected = next_k_items[envs_index]
            heightmapC_selected = self.space.heightmapC[envs_index]

            result = torch.cat([
                next_k_items_selected.float(),
                heightmapC_selected.view(num_envs_selected, -1)
            ], dim=1)

            return result

    def _handle_placed_items(self, envs_index, next_item_ids, return_replaced=False):
        """
        Handle already-placed items and return unplaced item IDs.

        Args:
            envs_index: [N] environment indices
            next_item_ids: [N] candidate item IDs
            return_replaced: Whether to return the replacement mask

        Returns:
            valid_item_ids: [N] valid item IDs (not yet placed)
            replaced: [N] whether items were replaced (only returned when return_replaced=True)
        """
        num_envs = len(envs_index)
        replaced = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        # Check if items are already placed
        safe_ids = torch.clamp(next_item_ids.long(), 0, self.num_item_types - 1)
        already_placed = self.placed_items_mask[envs_index, safe_ids]

        if not already_placed.any():
            if return_replaced:
                return next_item_ids, replaced
            return next_item_ids

        # Find unplaced items
        unplaced_mask = ~self.placed_items_mask[envs_index]  # [N, num_item_types]
        unplaced_counts = unplaced_mask.sum(dim=1)  # [N]

        # Environments that need regeneration
        need_regenerate = already_placed & (unplaced_counts > 0)
        all_placed = already_placed & (unplaced_counts == 0)

        # Mark replaced environments
        replaced = already_placed.clone()

        # Mark environments where all items have been placed
        if all_placed.any():
            next_item_ids[all_placed] = -1

        # Sample unplaced items for environments that need regeneration
        if need_regenerate.any():
            regen_indices = torch.where(need_regenerate)[0]
            sampling_weights = unplaced_mask[regen_indices].float()
            sampled = torch.multinomial(sampling_weights, num_samples=1).squeeze(-1)
            next_item_ids[regen_indices] = sampled.to(next_item_ids.dtype)

        if return_replaced:
            return next_item_ids, replaced
        return next_item_ids

    def _update_item_vec_vectorized(self, env_indices, positions, orientations):
        """Vectorized update of item_vec (fully on GPU, no CPU sync)"""
        update_mask = self.item_idx > 0

        if not update_mask.any():
            return

        update_envs = env_indices[update_mask]
        num_actors = positions.shape[1]
        max_items_in_actors = num_actors - 5
        max_capacity = self.item_vec.shape[1]
        max_item_idx = min(max_capacity, max_items_in_actors)

        if max_item_idx <= 0:
            return

        batch_indices = update_envs.unsqueeze(1).expand(-1, max_item_idx)
        item_indices = torch.arange(max_item_idx, device=self.device).unsqueeze(0).expand(len(update_envs), -1)

        # Use tensor comparison to avoid .item()
        valid_mask = item_indices < self.item_idx[update_envs].unsqueeze(1)

        # Use item type ID as actor index (consistent with step() which places at type_id+5)
        # item_vec[:,:,0] stores the type ID of each placed item
        actor_indices = self.item_vec[
            update_envs.unsqueeze(1).expand(-1, max_item_idx),
            item_indices, 0
        ].long() + 5  # [num_update_envs, max_item_idx]
        actor_indices = torch.clamp(actor_indices, 5, num_actors - 1)

        env_positions = positions[update_envs.unsqueeze(1).expand(-1, max_item_idx), actor_indices]
        env_orientations = orientations[update_envs.unsqueeze(1).expand(-1, max_item_idx), actor_indices]

        self.item_vec[batch_indices, item_indices, 1:4] = torch.where(
            valid_mask.unsqueeze(-1), env_positions, self.item_vec[batch_indices, item_indices, 1:4]
        )
        self.item_vec[batch_indices, item_indices, 4:8] = torch.where(
            valid_mask.unsqueeze(-1), env_orientations, self.item_vec[batch_indices, item_indices, 4:8]
        )

    def _check_all_placed_items_bounds(self, envs_index):
        """
        After physics settles, check if any previously placed item's centroid has
        left the bin boundaries (detected by actor root state position).

        Args:
            envs_index: [N] tensor of env indices to check

        Returns:
            [N] bool tensor — True = all items still in bounds, False = at least one flew out
        """
        self.interface.refresh_tensor()

        scaled_bin = (self.interface.bin * self.interface.defaultScale).to(torch.float32)
        N = len(envs_index)
        item_counts = self.item_idx[envs_index]  # [N]

        all_ok = torch.ones(N, device=self.device, dtype=torch.bool)
        if not (item_counts > 0).any():
            return all_ok

        max_items = self.item_vec.shape[1]

        # Slot validity mask: slot i is occupied if i < item_count for that env
        slot_range = torch.arange(max_items, device=self.device).unsqueeze(0)  # [1, max_items]
        placed_mask = slot_range < item_counts.unsqueeze(1)  # [N, max_items]

        # Actor index = item type ID + 5 (walls occupy 0-4)
        item_type_ids = self.item_vec[envs_index, :, 0].long()  # [N, max_items]
        actor_indices = (item_type_ids + 5).clamp(
            5, self.interface.actor_root_state_tensor.shape[1] - 1
        )  # [N, max_items]

        # Gather centroid positions [N, max_items, 3]
        env_expand = envs_index.unsqueeze(1).expand(-1, max_items)  # [N, max_items]
        pos = self.interface.actor_root_state_tensor[env_expand, actor_indices, :3]

        # Same convention as simulateToQuasistatic: compare against [0, scaled_bin]
        oob = (
            (pos[:, :, 0] <= 0) | (pos[:, :, 0] >= scaled_bin[0]) |
            (pos[:, :, 1] <= 0) | (pos[:, :, 1] >= scaled_bin[1]) |
            (pos[:, :, 2] <= 0) | (pos[:, :, 2] >= scaled_bin[2])
        )  # [N, max_items]

        # Fail any env where at least one placed slot is out of bounds
        all_ok[( oob & placed_mask).any(dim=1)] = False
        return all_ok

    def action_to_position(self, action, envs_index=None):
        """
        Convert action index to placement position (fully vectorized).

        Args:
            action: Action index
                   - int: all environments use the same action
                   - [N] tensor: action index per environment
            envs_index: Environment indices
                       - None: all environments
                       - [N] tensor: specified environments

        Returns:
            rotIdx: [N] rotation index
            targetFLB: [N, 3] target position (Front-Left-Bottom, unscaled coordinates)
            coordinate: [N, 2] grid coordinates (lx, ly)
        """
        # Handle environment indices
        if envs_index is None:
            envs_index = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(envs_index, torch.Tensor):
            envs_index = torch.tensor(envs_index, device=self.device, dtype=torch.long)

        num_envs = len(envs_index)

        # Handle action indices
        if isinstance(action, int):
            action = torch.full((num_envs,), action, device=self.device, dtype=torch.long)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.long)

        # Ensure actions are within valid range
        action = torch.clamp(action, 0, self.selectedAction - 1)

        # Get selected candidate from candidates
        # candidates: [num_envs, selectedAction, 5] -> [ROT, X, Y, H, V]
        selected_candidates = self.candidates[envs_index, action]  # [N, 5]

        # Extract each component
        rotIdx = selected_candidates[:, 0].long()      # [N] rotation index
        lx = selected_candidates[:, 1].long()          # [N] X grid coordinate
        ly = selected_candidates[:, 2].long()          # [N] Y grid coordinate
        posZ = selected_candidates[:, 3]               # [N] placement height

        # Compute target position (unscaled bin local coordinates)
        # Consistent with original: use grid coordinates directly, no XY offset
        targetFLB = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        targetFLB[:, 0] = lx.float() * self.resolutionAct  # X coordinate
        targetFLB[:, 1] = ly.float() * self.resolutionAct  # Y coordinate
        targetFLB[:, 2] = posZ + self.tolerance            # Z coordinate (with tolerance)

        # Precision handling (6 decimal places)
        targetFLB = torch.round(targetFLB * 1e6) / 1e6

        # Grid coordinates
        coordinate = torch.stack([lx, ly], dim=1)  # [N, 2]

        return rotIdx, targetFLB, coordinate

    def prejudge(self, rotIdx, targetFLB, naiveMask=None, next_item_ids=None, envs_index=None):
        """
        Pre-judge whether placement is feasible (fully vectorized).

        Conditions checked:
        1. The object will not exceed container boundaries after placement
        2. There exists a valid placement position in naiveMask

        Args:
            rotIdx: [N] rotation index
            targetFLB: [N, 3] target position (unscaled bin local coordinates)
            naiveMask: Placeable mask
                      - None: use self.stored_naiveMask
                      - [N, rotNum, rangeX_A, rangeY_A]: specified mask
            next_item_ids: [N] item type IDs (0-based)
                          - None: use self.next_item_vec
            envs_index: Environment indices
                       - None: all environments
                       - [N] tensor: specified environments

        Returns:
            success: [N] boolean tensor indicating feasibility
        """
        # Handle environment indices
        if envs_index is None:
            envs_index = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(envs_index, torch.Tensor):
            envs_index = torch.tensor(envs_index, device=self.device, dtype=torch.long)

        num_envs = len(envs_index)

        # Handle item IDs
        if next_item_ids is None:
            next_item_ids = self.next_item_vec[envs_index, 0].long()
        elif not isinstance(next_item_ids, torch.Tensor):
            next_item_ids = torch.tensor(next_item_ids, device=self.device, dtype=torch.long)

        # Handle mask
        if naiveMask is None:
            naiveMask = self.stored_naiveMask[envs_index]

        # Ensure rotIdx is a tensor
        if not isinstance(rotIdx, torch.Tensor):
            rotIdx = torch.tensor(rotIdx, device=self.device, dtype=torch.long)

        # Initialize success flag
        success = torch.ones(num_envs, device=self.device, dtype=torch.bool)

        # ==================== 1. Check Boundaries ====================
        # Use the same size data as get_possible_position (space.boundingSize, unscaled)
        # This ensures consistency between prejudge and naiveMask boundary checks
        valid_item_ids = torch.clamp(next_item_ids, 0, self.space.boundingSize.shape[0] - 1)
        valid_rot_ids = torch.clamp(rotIdx, 0, self.rotNum - 1)
        extents = self.space.boundingSize[valid_item_ids, valid_rot_ids, :]  # [N, 3] unscaled

        # targetFLB and bin_dimension are both unscaled coordinates
        bin_dim = torch.tensor(self.bin_dimension, device=self.device, dtype=torch.float32)

        # Check if X boundary is exceeded
        exceed_x = (targetFLB[:, 0] + extents[:, 0] - bin_dim[0]) > 1e-6

        # Check if Y boundary is exceeded
        exceed_y = (targetFLB[:, 1] + extents[:, 1] - bin_dim[1]) > 1e-6

        # Update success flag
        success = success & (~exceed_x) & (~exceed_y)

        # ==================== 2. Check Mask ====================
        # Whether there exists a valid position in naiveMask
        mask_sum = naiveMask.view(num_envs, -1).sum(dim=1)
        success = success & (mask_sum > 0)

        # ==================== 3. Check Item ID Validity ====================
        # Item ID of -1 means all items have been placed
        invalid_item = next_item_ids < 0
        success = success & (~invalid_item)

        return success


    def step(self, action, envs_index=None):
        """
        Execute action (fully vectorized).

        Args:
            action: Action index
                   - int: all environments use the same action
                   - [N] tensor: action index per environment
            envs_index: Environment indices
                       - None: all environments
                       - [N] tensor: specified environments

        Returns:
            observation: [N, obs_len] observation
            reward: [N] reward
            done: [N] whether episode is done
            info: dict containing 'Valid', 'counter', 'ratio' tensors
        """
        # ==================== 1. Handle Input ====================
        if envs_index is None:
            envs_index = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(envs_index, torch.Tensor):
            envs_index = torch.tensor(envs_index, device=self.device, dtype=torch.long)

        num_envs = len(envs_index)

        if isinstance(action, int):
            action = torch.full((num_envs,), action, device=self.device, dtype=torch.long)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.long)

        # ==================== 2. Get Item Info ====================
        # Consistent with original: use item ID determined in cur_observation, no placed-check
        next_item_ids = self.next_item_vec[envs_index, 0].long()
        invalid_items = next_item_ids < 0  # Invalid item check (when all items are placed)
        safe_item_ids = torch.clamp(next_item_ids, 0, self.num_item_types - 1)

        # ==================== 3. Action to Position ====================
        rotIdx, targetFLB, coordinate = self.action_to_position(action, envs_index)
        rotation = self.transformation[rotIdx]  # [N, 4]


        # ==================== 4. Pre-Judge ====================
        naiveMask = self.stored_naiveMask[envs_index]
        success = self.prejudge(rotIdx, targetFLB, naiveMask, next_item_ids, envs_index)

        success = success & (~invalid_items)

        # Check validity of the selected position
        coord_x = torch.clamp(coordinate[:, 0], 0, naiveMask.shape[2] - 1)
        coord_y = torch.clamp(coordinate[:, 1], 0, naiveMask.shape[3] - 1)
        mask_at_pos = naiveMask[torch.arange(num_envs, device=self.device), rotIdx, coord_x, coord_y]
        success = success & (mask_at_pos > 0)

        # ==================== 5. Place Object ====================
        # Use item type ID as actor index (actors are created per type)
        # Note: sequence generation ensures each type appears only once, avoiding actor conflicts
        # actor index = item_type_id + 5 (first 5 are bin walls)
        actor_indices_local = safe_item_ids + 5  # [num_envs] item type -> actor
        sim_suc = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        if success.any():
            success_local = torch.where(success)[0]
            success_envs = envs_index[success_local]

            # Use interface.addObject to place objects
            self.interface.addObject(
                env_ids=success_envs,
                actor_indices=actor_indices_local[success_local],
                targetFLB=targetFLB[success_local],
                rotations=rotation[success_local],
                item_type_ids=safe_item_ids[success_local]  # Item type IDs
            )

            # ==================== 6. Physics Simulation ====================
            # Create global placed_actor_indices tensor
            # simulateToQuasistatic expects shape [num_envs], values < 5 mean no object placed
            global_placed_actors = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            global_placed_actors[success_envs] = actor_indices_local[success_local]

            if self.simulation:
                # Use simulateToQuasistatic to simulate until quasi-static
                # Returns (done_buf, success_buf)
                # success_buf: True means success, False means out of bounds
                _, success_buf = self.interface.simulateToQuasistatic(
                    placed_actor_indices=global_placed_actors,  # Global shape [num_envs]
                    linear_tol=0.025,    # Relaxed tolerance for faster convergence, 0.02
                    angular_tol=0.025,   # Relaxed tolerance for faster convergence, 0.02
                    min_iterations=10,  # Increased minimum iterations
                    max_iterations=50   # Increased maximum iterations, 60
                )
                # Only take results for successfully placed environments
                sim_suc[success_local] = success_buf[success_envs]
                # Also fail any env where a PREVIOUSLY placed item was knocked out
                all_items_ok = self._check_all_placed_items_bounds(success_envs)
                sim_suc[success_local] = sim_suc[success_local] & all_items_ok
            else:
                # Non-simulation mode, use simulateHeight to check height
                # Returns (done_buf, success_buf): True means success
                _, success_buf = self.interface.simulateHeight(
                    placed_actor_indices=global_placed_actors  # Global shape [num_envs]
                )
                sim_suc[success_local] = success_buf[success_envs]

        # ==================== 7. Compute Results ====================
        # Fully successful environments
        success_final = success & sim_suc
        failed_final = success & (~sim_suc)

        # done: True when pre-judge fails or simulation fails (object ejected from bin)
        # When simulation fails, the entire environment must be reset since other objects may be affected
        done = (~success) | failed_final

        # Reward: per-step volume reward + terminal packing efficiency reward
        reward = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        if success_final.any():
            # Per-step reward: item_volume / bin_volume * 10
            item_ratio = self.get_item_ratio(next_item_ids[success_final])
            volume_reward = item_ratio * 10
            reward[success_final] = volume_reward

        # Terminal reward: total packed volume / bin volume * 10, given at episode end.
        # Incentivises maximising overall packing density, not just placing individual items.
        # Resets packed volume immediately so the next episode starts clean.
        if done.any():
            terminal_bonus = self.total_packed_volume[done] / self.bin_volume * 10.0
            reward[done] = reward[done] + terminal_bonus
            self.total_packed_volume[done] = 0.0

        # ==================== 8. Update State ====================
        if success_final.any():
            success_local = torch.where(success_final)[0]
            success_envs = envs_index[success_local]

            # Update item_vec
            item_idx = self.item_idx[success_envs].long()
            valid_mask = item_idx < self.max_items
            if valid_mask.any():
                valid_envs = success_envs[valid_mask]
                valid_idx = item_idx[valid_mask]
                self.item_vec[valid_envs, valid_idx, 0] = next_item_ids[success_local][valid_mask].float()
                self.item_vec[valid_envs, valid_idx, -1] = 1

            # Update placed items mask (prevents same-type item from being regenerated)
            self.placed_items_mask[success_envs, safe_item_ids[success_local]] = True

            # Update count
            self.item_idx[success_envs] += 1

            # Update height map
            # [Use actual AABB] For regular objects (cubes), AABB is exact.
            # Use actual post-simulation positions so minor object movements are reflected correctly.
            # Key: pass item type ID to get correct mesh bounds
            self._update_heightmap_aabb(
                env_ids=success_envs,
                actor_indices=actor_indices_local[success_local],
                item_type_ids=safe_item_ids[success_local]  # Item type IDs
            )

            # Accumulate packed volume for terminal packing efficiency reward
            self.total_packed_volume[success_envs] += self.items_volume[next_item_ids[success_local] + 5]

        # ==================== 9. Handle Simulation Failures ====================
        # Environments with simulation failures are marked as done; the training loop will call reset.
        # However, we still need to immediately reset ejected objects to prevent inconsistent states in cur_observation.
        if failed_final.any():
            failed_local = torch.where(failed_final)[0]
            failed_envs = envs_index[failed_local]
            failed_actors = actor_indices_local[failed_local]
            failed_items = safe_item_ids[failed_local]


            # Reset ejected objects to initial positions (storage area)
            self.interface.actor_root_state_tensor[failed_envs, failed_actors] = \
                self.interface.init_actor_root_state_tensor[failed_envs, failed_actors]

            global_indices = self.interface.global_indices[failed_envs, failed_actors].to(torch.int32)
            self.interface.gym.set_actor_root_state_tensor_indexed(
                self.interface.sim,
                gymtorch.unwrap_tensor(self.interface.actor_root_state_tensor.view(-1, 13)),
                gymtorch.unwrap_tensor(global_indices),
                len(global_indices)
            )

        # ==================== 10. Update Item Queue ====================
        processed = success_final | failed_final
        if processed.any():
            processed_envs = envs_index[processed]
            self.item_creator.pop_first(env_indices=processed_envs)

        # ==================== 11. Get Observation ====================
        observation = self.cur_observation(genItem=True, envs_index=envs_index)

        # ==================== 12. Build Info ====================
        ratio = self.get_ratio()[envs_index]
        info = {
            'Valid': sim_suc,
            'counter': self.item_idx[envs_index].clone(),
            'ratio': torch.where(done, ratio, torch.zeros_like(ratio))
        }

        return observation, reward, done, info

    def _update_heightmap_aabb(self, env_ids, actor_indices, item_type_ids=None):
        """
        Update height map using AABB (fallback method, used when arbitrary rotations need handling).

        Args:
            env_ids: [N] environment indices
            actor_indices: [N] actor indices (used to get position in simulation)
            item_type_ids: [N] item type IDs (0-based, used to get correct mesh bounds)
                          If None, actor_indices are used (backward compatible, may be incorrect)
        """
        # Refresh state tensor to get actual post-simulation positions
        self.interface.gym.refresh_actor_root_state_tensor(self.interface.sim)

        # Get actual positions and quaternions
        actual_positions = self.interface.actor_root_state_tensor[env_ids, actor_indices, 0:3]
        actual_quaternions = self.interface.actor_root_state_tensor[env_ids, actor_indices, 3:7]

        # Key fix: use item type ID to get correct mesh bounds
        if item_type_ids is not None:
            bounds_indices = item_type_ids + 5  # item type -> bounds index
        else:
            bounds_indices = actor_indices  # backward compatible
        mesh_bounds = self.interface.mesh_bounds[bounds_indices, :, :3]

        # Use AABB to update height map
        # Key: world coordinates = bin local coordinates x defaultScale x meshScale
        # So converting back to bin local coordinates requires dividing by (defaultScale x meshScale)
        # This way, coordinates in the height map are in the same coordinate system as boundingSize in shapeDict
        actual_scale = self.interface.defaultScale * self.interface.meshScale

        self.space.place_item_with_actual_aabb(
            actual_positions=actual_positions,
            actual_quaternions=actual_quaternions,
            mesh_bounds=mesh_bounds,
            mesh_centroids=None,    # actual_positions is the OBJ origin, no centroid offset needed
            env_ids=env_ids,
            scale=actual_scale,     # Use defaultScale x meshScale
            height_margin=0.002,    # 2mm height safety margin
            xy_margin=0.0           # Consistent with original: no X/Y margin, tight packing
        )

    def select_valid_action_vectorized(self, envs_index=None):
        """
        Select action from valid candidate positions (vectorized).

        Args:
            envs_index: Environment indices
                       - None: all environments
                       - [N] tensor: specified environments

        Returns:
            action: [N] action indices
        """
        # Handle environment indices
        if envs_index is None:
            envs_index = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(envs_index, torch.Tensor):
            envs_index = torch.tensor(envs_index, device=self.device, dtype=torch.long)

        # Get validity of candidate points for specified environments
        validity = self.candidates[envs_index, :, 4]  # [N, selectedAction]
        valid_mask = validity > 0
        weights = valid_mask.float()

        # Default to first candidate when there are no valid positions
        no_valid = weights.sum(dim=1) == 0
        weights[no_valid, 0] = 1.0

        # Random sampling
        actions = torch.multinomial(weights, num_samples=1).squeeze(1)
        return actions

    def render(self):
        """Render the environment"""
        if hasattr(self.interface, 'render'):
            self.interface.render()


if __name__ == '__main__':
    import argparse

    # ==================== Test Configuration ====================
    NUM_ENVS = 100              # Number of parallel environments
    MAX_EPISODES = 100        # Maximum number of episodes
    MAX_STEPS_PER_EP = 50     # Maximum steps per episode
    HEADLESS = True          # Whether to run in headless mode
    RENDER_INTERVAL = 5       # Render interval (steps), reduces render frequency
    PRINT_INTERVAL = 5        # Print interval (steps)

    print("=" * 60)
    print("   PackingGame (IsaacGym) Reinforcement Learning Environment Test")
    print("=" * 60)

    # ==================== 1. Initialize Environment ====================
    print("\n[1] Initializing environment...")
    args = get_args()

    # For testing: reduce bin dimensions
    args.bin_dimension = np.array([0.35, 0.35, 0.35])

    env = PackingGame(
        args=args,
        num_envs=NUM_ENVS,
        sim_device=0,
        headless=HEADLESS
    )

    print(f"    - Number of environments: {env.num_envs}")
    print(f"    - Device: {env.device}")
    print(f"    - Number of item types: {env.num_item_types}")
    print(f"    - Number of rotations: {env.rotNum}")
    print(f"    - Bin dimensions: {env.bin_dimension}")
    print(f"    - Observation dimensions: {env.obs_len}")
    print(f"    - Action space: {env.action_space}")
    print(f"    - Observation space: {env.observation_space}")
    print(f"    - Global view: {env.globalView}")
    print(f"    - Physics simulation: {env.simulation}")

    # ==================== 2. Test reset ====================
    print("\n[2] Testing reset()...")

    # Full reset
    obs = env.reset()
    print(f"    - Observation shape after full reset: {obs.shape}")
    print(f"    - Observation range: [{obs.min().item():.4f}, {obs.max().item():.4f}]")

    # Partial reset
    partial_ids = torch.tensor([0, 2], device=env.device)
    obs_partial = env.reset(partial_ids)
    print(f"    - Observation shape after partial reset (env 0, 2): {obs_partial.shape}")

    # ==================== 3. Test cur_observation ====================
    print("\n[3] Testing cur_observation()...")
    obs = env.cur_observation(genItem=True)
    print(f"    - Observation shape: {obs.shape}")
    print(f"    - Next item IDs: {env.next_item_vec[:, 0].tolist()}")
    print(f"    - Number of candidate actions: {env.candidates.shape[1]}")
    print(f"    - Number of valid candidates: {(env.candidates[:, :, 4] > 0).sum(dim=1).tolist()}")

    # ==================== 4. Test action_to_position ====================
    print("\n[4] Testing action_to_position()...")
    test_action = torch.zeros(NUM_ENVS, device=env.device, dtype=torch.long)
    rotIdx, targetFLB, coordinate = env.action_to_position(test_action)
    print(f"    - Action 0 -> rotIdx: {rotIdx.tolist()}")
    print(f"    - Action 0 -> targetFLB: {targetFLB[0].tolist()}")
    print(f"    - Action 0 -> coordinate: {coordinate[0].tolist()}")

    # ==================== 5. Test prejudge ====================
    print("\n[5] Testing prejudge()...")
    naiveMask = env.stored_naiveMask
    next_item_ids = env.next_item_vec[:, 0].long()
    success = env.prejudge(rotIdx, targetFLB, naiveMask, next_item_ids)
    print(f"    - Pre-judge results: {success.tolist()}")

    # ==================== 6. Test Full Step Loop ====================
    print("\n[6] Starting reinforcement learning loop test...")
    print(f"    - Maximum episodes: {MAX_EPISODES}")
    print(f"    - Maximum steps per episode: {MAX_STEPS_PER_EP}")
    print("-" * 60)

    # Reset statistics (all tensors)
    obs = env.reset()
    episode_count = torch.zeros(NUM_ENVS, device=env.device, dtype=torch.int32)
    episode_rewards = torch.zeros(NUM_ENVS, device=env.device, dtype=torch.float32)
    episode_lengths = torch.zeros(NUM_ENVS, device=env.device, dtype=torch.int32)

    # History records (accumulated using tensors)
    history_rewards = torch.zeros(MAX_EPISODES * NUM_ENVS, device=env.device, dtype=torch.float32)
    history_lengths = torch.zeros(MAX_EPISODES * NUM_ENVS, device=env.device, dtype=torch.int32)
    history_ratios = torch.zeros(MAX_EPISODES * NUM_ENVS, device=env.device, dtype=torch.float32)

    global_step = 0
    total_episodes = 0

    try:
        while total_episodes < MAX_EPISODES * NUM_ENVS:
            # Select action (vectorized)
            action = env.select_valid_action_vectorized()

            # Execute action (vectorized)
            obs, reward, done, info = env.step(action)

            # Update episode statistics (vectorized)
            episode_rewards += reward
            episode_lengths += 1
            global_step += 1

            # Render
            if global_step % RENDER_INTERVAL == 0:
                env.render()

            # Print step info
            if global_step % PRINT_INTERVAL == 0:
                valid_count = info['Valid'].sum().item()
                print(f"    Step {global_step:4d} | "
                      f"Reward: {reward.sum().item():6.2f} | "
                      f"Valid: {valid_count}/{NUM_ENVS} | "
                      f"Done: {done.sum().item()}/{NUM_ENVS}")

            # Handle finished environments (vectorized)
            done_env_ids = torch.where(done)[0]
            num_done = len(done_env_ids)
            if num_done > 0:
                # Vectorized history recording
                end_idx = min(total_episodes + num_done, MAX_EPISODES * NUM_ENVS)
                record_count = end_idx - total_episodes

                history_rewards[total_episodes:end_idx] = episode_rewards[done_env_ids[:record_count]]
                history_lengths[total_episodes:end_idx] = episode_lengths[done_env_ids[:record_count]]
                history_ratios[total_episodes:end_idx] = info['ratio'][done_env_ids[:record_count]]

                # Vectorized count update
                episode_count[done_env_ids] += 1
                total_episodes += num_done

                # Print summary (no for-loop)
                avg_reward = episode_rewards[done_env_ids].mean().item()
                avg_length = episode_lengths[done_env_ids].float().mean().item()
                avg_ratio = info['ratio'][done_env_ids].mean().item()
                print(f"    [Episode {total_episodes:3d}] {num_done} envs done | "
                      f"AvgReward={avg_reward:6.2f}, AvgLen={avg_length:5.1f}, AvgRatio={avg_ratio:.2%}")

                # Vectorized reset
                episode_rewards[done_env_ids] = 0
                episode_lengths[done_env_ids] = 0
                obs = env.reset(done_env_ids)

            # Handle timed-out environments (vectorized)
            timeout_mask = episode_lengths >= MAX_STEPS_PER_EP
            timeout_env_ids = torch.where(timeout_mask)[0]
            num_timeout = len(timeout_env_ids)
            if num_timeout > 0:
                # Vectorized history recording
                end_idx = min(total_episodes + num_timeout, MAX_EPISODES * NUM_ENVS)
                record_count = end_idx - total_episodes

                history_rewards[total_episodes:end_idx] = episode_rewards[timeout_env_ids[:record_count]]
                history_lengths[total_episodes:end_idx] = episode_lengths[timeout_env_ids[:record_count]]
                history_ratios[total_episodes:end_idx] = env.get_ratio()[timeout_env_ids[:record_count]]

                episode_count[timeout_env_ids] += 1
                total_episodes += num_timeout

                # Print summary
                avg_reward = episode_rewards[timeout_env_ids].mean().item()
                avg_length = episode_lengths[timeout_env_ids].float().mean().item()
                print(f"    [Timeout {total_episodes:3d}] {num_timeout} envs timeout | "
                      f"AvgReward={avg_reward:6.2f}, AvgLen={avg_length:5.1f}")

                # Vectorized reset
                episode_rewards[timeout_env_ids] = 0
                episode_lengths[timeout_env_ids] = 0
                obs = env.reset(timeout_env_ids)

    except KeyboardInterrupt:
        print("\n    [User interrupted]")

    # ==================== 7. Output Statistics (Vectorized) ====================
    print("\n" + "=" * 60)
    print("   Test Result Statistics")
    print("=" * 60)

    if total_episodes > 0:
        # Slice valid records
        valid_rewards = history_rewards[:total_episodes]
        valid_lengths = history_lengths[:total_episodes].float()
        valid_ratios = history_ratios[:total_episodes]

        print(f"\nTotal episodes: {total_episodes}")
        print(f"Total steps: {global_step}")

        print(f"\nReward statistics:")
        print(f"    - Mean: {valid_rewards.mean().item():.2f}")
        print(f"    - Std: {valid_rewards.std().item():.2f}")
        print(f"    - Min: {valid_rewards.min().item():.2f}")
        print(f"    - Max: {valid_rewards.max().item():.2f}")

        print(f"\nEpisode length statistics:")
        print(f"    - Mean: {valid_lengths.mean().item():.1f}")
        print(f"    - Std: {valid_lengths.std().item():.1f}")
        print(f"    - Min: {valid_lengths.min().item():.0f}")
        print(f"    - Max: {valid_lengths.max().item():.0f}")

        print(f"\nFill ratio statistics:")
        print(f"    - Mean: {valid_ratios.mean().item():.2%}")
        print(f"    - Std: {valid_ratios.std().item():.2%}")
        print(f"    - Min: {valid_ratios.min().item():.2%}")
        print(f"    - Max: {valid_ratios.max().item():.2%}")
    else:
        print("\nNo episodes completed")

    # ==================== 8. Cleanup ====================
    print("\n[8] Cleaning up resources...")
    env.close()
    print("    Test complete!")
