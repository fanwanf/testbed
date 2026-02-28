from turtle import position
import numpy as np
import torch
import sys
import os
from operator import itemgetter
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from tools import gen_ray_origin_direction, getRotationMatrix
from arguments import get_args
from tools_isaacgym import shot_item_isaacgym

class SpaceIsaacGym:
    def __init__(self, bin_dimension, resolutionAct, resolutionH, boxPack = False,  ZRotNum = None,
                                        shotInfo = None, scale = None, device = None, num_envs = None,args = None):
        self.bin_dimension = torch.tensor(bin_dimension, device=device, dtype=torch.float32)
        self.device = device
        self.resolutionAct = resolutionAct
        self.resolutionH = resolutionH
        self.boxPack = boxPack
        self.ZRotNum = ZRotNum
        self.shotInfo = shotInfo
        self.num_envs = num_envs
        self.args = args

        # Compute stepSize
        self.stepSize = int(self.resolutionAct / self.resolutionH)
        assert self.stepSize == self.resolutionAct / self.resolutionH

        self.rotNum = ZRotNum  # default 8, representing 8 rotation angles
        self.scale = scale if scale is None else torch.tensor(scale, device=device, dtype=torch.float32)

        # Use torch.ceil instead of np.ceil and convert to int32
        bin_dim_xy = self.bin_dimension[0:2]  # get X and Y dimensions
        ranges_C = torch.ceil(bin_dim_xy / self.resolutionH).to(torch.int32)
        ranges_A = torch.ceil(bin_dim_xy / self.resolutionAct).to(torch.int32)

        self.rangeX_C, self.rangeY_C = ranges_C[0].item(), ranges_C[1].item()  # height map grid count 2*2
        self.rangeX_A, self.rangeY_A = ranges_A[0].item(), ranges_A[1].item()  # action space grid count 1*1

        # Height map: a 2D array recording the maximum Z coordinate (height) at each (X, Y) position in 3D space
        # Initialized to 0, consistent with the original version
        self.heightmapC = torch.zeros((self.num_envs,self.rangeX_C, self.rangeY_C), device=device, dtype=torch.float32)

        # Generate ray origins and directions for height map computation (for pybullet)
        self.ray_origins, self.ray_directions = gen_ray_origin_direction(self.rangeX_C, self.rangeY_C, self.resolutionH, self.boxPack, shift = 0.001)

        self.shotInfo = shotInfo

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(1, self.ZRotNum)
        for d in DownFaceList:
            for z in ZRotList:
                self.transformation.append(np.dot(z, d).reshape(-1))
        self.transformation = np.array(self.transformation)
        self.transformation_tensor = torch.tensor(self.transformation, device=device, dtype=torch.float32)

        # Map 2D grid indices to coordinate values, used for position scoring and sorting
        bottom = torch.arange(0, self.rangeX_A * self.rangeY_A, device=device, dtype=torch.int32).view(self.rangeX_A, self.rangeY_A)
        self.coors = torch.zeros((self.rangeX_A, self.rangeY_A, 2), device=device, dtype=torch.float32)
        self.coors[:,:,0] = bottom // self.rangeY_A
        self.coors[:,:,1] = bottom % self.rangeY_A

        obj_nums = len(self.args.shapeDict.keys())
        self.boundingSize = torch.zeros((obj_nums, self.rotNum, 3), device=device, dtype=torch.float32)
        self.rangeX_OH = torch.zeros((obj_nums, self.rotNum), device=device, dtype=torch.int32)
        self.rangeX_OA = torch.zeros((obj_nums, self.rotNum), device=device, dtype=torch.int32)
        self.rangeY_OH = torch.zeros((obj_nums, self.rotNum), device=device, dtype=torch.int32)
        self.rangeY_OA = torch.zeros((obj_nums, self.rotNum), device=device, dtype=torch.int32)
        self.coorX = torch.zeros((obj_nums, self.rotNum, self.rangeX_A,self.rangeY_A), device=device, dtype=torch.int32)
        self.coorY = torch.zeros((obj_nums, self.rotNum, self.rangeX_A,self.rangeY_A), device=device, dtype=torch.int32)
        self.posZ = torch.zeros((self.num_envs,self.rotNum,self.rangeX_A,self.rangeY_A), device=device, dtype=torch.float32)
        self.posZ[:] = 1e3
        self.posZvalid = torch.zeros((self.num_envs,self.rotNum,self.rangeX_A,self.rangeY_A), device=device, dtype=torch.float32)

        # First compute all object sizes and find the maximum size
        max_Sx_all = 0
        max_Sy_all = 0
        temp_heightMaps = {}  # temporary height map storage

        for i in self.args.shapeDict.keys():
            for j in range(self.rotNum):
                self.boundingSize[i, j] = torch.tensor(self.args.shapeDict[i][j].extents, device=device, dtype=torch.float32)
                self.rangeX_OH[i, j], self.rangeY_OH[i, j] = torch.ceil(self.boundingSize[i, j, 0:2] / self.resolutionH).to(torch.int32)
                self.rangeX_OA[i, j], self.rangeY_OA[i, j] = torch.ceil(self.boundingSize[i, j, 0:2] / self.resolutionAct).to(torch.int32)

                # Update maximum sizes
                max_Sx_all = max(max_Sx_all, self.rangeX_OH[i, j].item())
                max_Sy_all = max(max_Sy_all, self.rangeY_OH[i, j].item())

                # [rangeX_OH,rangeY_OH]
                heightMapT,heightMapB,maskH,maskB = shot_item_isaacgym(self.args.shapeDict[i][j],self.ray_origins,self.ray_directions,
                                                self.rangeX_OH[i, j],self.rangeY_OH[i, j],self.device)
                # Temporarily store height map
                temp_heightMaps[(i, j)] = (heightMapT, heightMapB, maskH, maskB)

        # Store height maps as tensors with shape [obj_nums, rotNum, max_Sx_all, max_Sy_all]
        self.heightMapT = torch.zeros(obj_nums, self.rotNum, max_Sx_all, max_Sy_all, device=device, dtype=torch.float32)
        self.heightMapB = torch.zeros(obj_nums, self.rotNum, max_Sx_all, max_Sy_all, device=device, dtype=torch.float32)
        self.maskH = torch.zeros(obj_nums, self.rotNum, max_Sx_all, max_Sy_all, device=device, dtype=torch.int32)
        self.maskB = torch.zeros(obj_nums, self.rotNum, max_Sx_all, max_Sy_all, device=device, dtype=torch.int32)

        # Fill height maps into uniform-sized tensors
        for i in self.args.shapeDict.keys():
            for j in range(self.rotNum):
                heightMapT, heightMapB, maskH, maskB = temp_heightMaps[(i, j)]
                current_Sx = self.rangeX_OH[i, j].item()
                current_Sy = self.rangeY_OH[i, j].item()

                # Pad height map to uniform size
                if current_Sx > 0 and current_Sy > 0:
                    self.heightMapT[i, j, :current_Sx, :current_Sy] = heightMapT
                    self.heightMapB[i, j, :current_Sx, :current_Sy] = heightMapB
                    self.maskH[i, j, :current_Sx, :current_Sy] = maskH
                    self.maskB[i, j, :current_Sx, :current_Sy] = maskB

        # Vectorized computation of coorX and coorY (replaces nested for loops)
        # coorX[i, j, k, l] = k * stepSize, coorY[i, j, k, l] = l * stepSize
        # These values are the same for all objects and rotation angles
        k_indices = torch.arange(self.rangeX_A, device=device, dtype=torch.int32)
        l_indices = torch.arange(self.rangeY_A, device=device, dtype=torch.int32)
        k_grid, l_grid = torch.meshgrid(k_indices, l_indices, indexing='ij')  # [rangeX_A, rangeY_A]

        # Broadcast to all objects and rotation angles
        self.coorX[:, :, :, :] = (k_grid * self.stepSize).unsqueeze(0).unsqueeze(0).expand(obj_nums, self.rotNum, -1, -1)
        self.coorY[:, :, :, :] = (l_grid * self.stepSize).unsqueeze(0).unsqueeze(0).expand(obj_nums, self.rotNum, -1, -1)



    def reset(self, env_ids=None):
        """
        Reset the space state for specified environments (multi-environment parallel, no for loops)

        Consistent with the reset function in the original space.py:
        - Reset height map heightmapC to 0
        - Reset position computation state

        Args:
            env_ids: [N] environment indices to reset
                     If None, reset all environments
        """
        if env_ids is None:
            # Reset all environments
            self.heightmapC[:] = 0
            self.posZ[:] = 1e3
            self.posZvalid[:] = 1e3
        else:
            # Convert to tensor if needed
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

            # If no environments to reset, return immediately
            if len(env_ids) == 0:
                return

            # Reset only specified environments (tensor indexing, no for loops)
            self.heightmapC[env_ids] = 0
            self.posZ[env_ids] = 1e3
            self.posZvalid[env_ids] = 1e3

    def place_item_trimesh(self, item_ids, rot_ids, positions, env_ids=None):
        """
        Incrementally update height map (multi-environment vectorized version)

        Consistent with place_item_trimesh logic in the original space.py:
        1. Compute bounds from item position and rotation
        2. Convert to grid indices
        3. Update heightmapC using precomputed height map data

        Args:
            item_ids: [N] item IDs (0-23, used to index heightMapT)
            rot_ids: [N] rotation angle indices
            positions: [N, 3] item placement positions (FLB coordinates, bin local frame, unscaled)
            env_ids: [N] environment indices to update; if None, assume N=num_envs
        """
        # 1. Process input arguments
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        if not isinstance(item_ids, torch.Tensor):
            item_ids = torch.tensor(item_ids, device=self.device, dtype=torch.long)
        if not isinstance(rot_ids, torch.Tensor):
            rot_ids = torch.tensor(rot_ids, device=self.device, dtype=torch.long)
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, device=self.device, dtype=torch.float32)

        num_items = len(env_ids)
        if num_items == 0:
            return

        # 2. Validate item_ids and filter invalid entries
        obj_nums = self.boundingSize.shape[0]
        valid_mask = (item_ids >= 0) & (item_ids < obj_nums)
        if not valid_mask.all():
            item_ids = item_ids[valid_mask]
            rot_ids = rot_ids[valid_mask]
            positions = positions[valid_mask]
            env_ids = env_ids[valid_mask]
            num_items = len(env_ids)
            if num_items == 0:
                return

        # 3. Compute item bounds (matches original: bounds = np.round(meshT.bounds, decimals=6))
        # boundingSize: [obj_nums, rotNum, 3]
        item_size = self.boundingSize[item_ids, rot_ids, :]  # [N, 3]
        bounds_min = positions  # FLB coordinate is bounds[0]
        bounds_max = positions + item_size  # [N, 3]

        # 4. Clamp bounds within bin and convert to grid indices
        # Matches original: minBoundsInt = np.floor(np.maximum(bounds[0], [0,0,0]) / self.resolutionH).astype(np.int32)
        zeros = torch.zeros(3, device=self.device, dtype=torch.float32)
        bounds_min_clamped = torch.maximum(bounds_min, zeros)
        bounds_max_clamped = torch.minimum(bounds_max, self.bin_dimension)

        minBoundsInt = torch.floor(bounds_min_clamped / self.resolutionH).to(torch.int32)  # [N, 3]
        maxBoundsInt = torch.ceil(bounds_max_clamped / self.resolutionH).to(torch.int32)  # [N, 3]

        # 5. Compute item occupancy range in height map
        # Matches original: boundingSizeInt = maxBoundsInt - minBoundsInt
        rangeX_O = (maxBoundsInt[:, 0] - minBoundsInt[:, 0])  # [N]
        rangeY_O = (maxBoundsInt[:, 1] - minBoundsInt[:, 1])  # [N]

        # Use precomputed sizes to ensure they don't exceed heightMapT bounds
        precomputed_rangeX = self.rangeX_OH[item_ids, rot_ids]  # [N]
        precomputed_rangeY = self.rangeY_OH[item_ids, rot_ids]  # [N]
        rangeX_O = torch.minimum(rangeX_O, precomputed_rangeX)
        rangeY_O = torch.minimum(rangeY_O, precomputed_rangeY)

        max_Sx = rangeX_O.max().item()
        max_Sy = rangeY_O.max().item()

        if max_Sx <= 0 or max_Sy <= 0:
            return

        # 6. Retrieve precomputed height map data
        # Matches original: heightMapH, maskH = shot_after_item_placement(...)
        heightMapT_batch = self.heightMapT[item_ids, rot_ids, :max_Sx, :max_Sy]  # [N, max_Sx, max_Sy]
        maskH_batch = self.maskH[item_ids, rot_ids, :max_Sx, :max_Sy]  # [N, max_Sx, max_Sy]

        # 7. Compute actual height after placement
        # heightMapH = heightMapT + posZ (item bottom height)
        posZ = positions[:, 2].unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        heightMapH = (heightMapT_batch + posZ) * maskH_batch  # [N, max_Sx, max_Sy]

        # 8. Build grid indices for batch update
        dx = torch.arange(max_Sx, device=self.device, dtype=torch.long)
        dy = torch.arange(max_Sy, device=self.device, dtype=torch.long)
        dx_grid, dy_grid = torch.meshgrid(dx, dy, indexing='ij')  # [max_Sx, max_Sy]

        # Expand to [N, max_Sx, max_Sy]
        dx_grid = dx_grid.unsqueeze(0).expand(num_items, -1, -1)
        dy_grid = dy_grid.unsqueeze(0).expand(num_items, -1, -1)

        # Compute absolute grid coordinates
        # Matches original: coorX, coorY = minBoundsInt[0:2]
        coorX = minBoundsInt[:, 0].unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        coorY = minBoundsInt[:, 1].unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        x_indices = coorX + dx_grid  # [N, max_Sx, max_Sy]
        y_indices = coorY + dy_grid  # [N, max_Sx, max_Sy]

        # Valid region mask (ensure indices are within each item's actual range)
        rangeX_exp = rangeX_O.unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        rangeY_exp = rangeY_O.unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        valid_region = (dx_grid < rangeX_exp) & (dy_grid < rangeY_exp)  # [N, max_Sx, max_Sy]

        # Boundary check
        x_indices = torch.clamp(x_indices, 0, self.rangeX_C - 1)
        y_indices = torch.clamp(y_indices, 0, self.rangeY_C - 1)

        # Environment indices
        env_indices = env_ids.unsqueeze(1).unsqueeze(2).expand(-1, max_Sx, max_Sy)  # [N, max_Sx, max_Sy]

        # 9. Update height map (take maximum)
        # Matches original: self.heightmapC[...] = np.maximum(self.heightmapC[...], heightMapH)
        current_heights = self.heightmapC[env_indices, x_indices, y_indices]  # [N, max_Sx, max_Sy]
        new_heights = torch.where(valid_region, torch.maximum(current_heights, heightMapH), current_heights)
        self.heightmapC[env_indices, x_indices, y_indices] = new_heights

    def place_item_with_actual_aabb(self, actual_positions, actual_quaternions, mesh_bounds,
                                     mesh_centroids=None, env_ids=None, scale=None,
                                     height_margin=0.0, xy_margin=0.0):
        """
        Update height map based on actual AABB (multi-environment vectorized version)

        Uses the actual position and quaternion after simulation to compute the AABB.
        Suitable for scenarios where objects may shift due to rotation.

        Args:
            actual_positions: [N, 3] actual world-space position after simulation (scaled, centroid position)
            actual_quaternions: [N, 4] actual quaternion after simulation (x, y, z, w)
            mesh_bounds: [N, 2, 3] local-frame mesh bounds [min, max] (scaled, relative to OBJ origin)
            mesh_centroids: [N, 3] mesh centroid position (scaled, relative to OBJ origin),
                           used to convert mesh_bounds to centroid-relative coordinates
            env_ids: [N] environment indices; if None, assume N=num_envs
            scale: [3] scale factor for converting world coordinates to bin local coordinates
            height_margin: height safety margin (unscaled), added to top height
            xy_margin: X/Y direction safety margin (unscaled), enlarges occupancy range
        """
        # 1. Process input arguments
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        if not isinstance(actual_positions, torch.Tensor):
            actual_positions = torch.tensor(actual_positions, device=self.device, dtype=torch.float32)
        if not isinstance(actual_quaternions, torch.Tensor):
            actual_quaternions = torch.tensor(actual_quaternions, device=self.device, dtype=torch.float32)
        if not isinstance(mesh_bounds, torch.Tensor):
            mesh_bounds = torch.tensor(mesh_bounds, device=self.device, dtype=torch.float32)

        num_items = len(env_ids)
        if num_items == 0:
            return

        if scale is None:
            scale = self.scale
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=self.device, dtype=torch.float32)

        # 2. Compute rotation matrix from quaternion
        # IsaacGym quaternion format: (x, y, z, w)
        # Use the same method as Interface_isaacgym.py for consistency
        from isaacgymenvs.utils.torch_jit_utils import quaternion_to_matrix

        # Convert quaternion format: IsaacGym (x,y,z,w) -> quaternion_to_matrix (w,x,y,z)
        quaternions_wxyz = torch.cat([actual_quaternions[..., 3:4], actual_quaternions[..., 0:3]], dim=-1)
        rot_matrices = quaternion_to_matrix(quaternions_wxyz)  # [N, 3, 3]

        # 3. Compute rotated AABB
        # mesh_bounds: [N, 2, 3] -> local_min [N, 3], local_max [N, 3]
        # Note: mesh_bounds are relative to OBJ origin (scaled)
        local_min = mesh_bounds[:, 0, :]  # [N, 3] (scaled, relative to OBJ origin)
        local_max = mesh_bounds[:, 1, :]  # [N, 3] (scaled, relative to OBJ origin)

        # If centroids are provided, convert bounds to centroid-relative coordinates
        # actual_positions is the centroid world coordinate, so we need centroid-relative bounds
        if mesh_centroids is not None:
            local_min = local_min - mesh_centroids  # [N, 3] relative to centroid
            local_max = local_max - mesh_centroids  # [N, 3] relative to centroid

        # Compute 8 corner points (relative to centroid/origin)
        corners = torch.zeros((num_items, 8, 3), device=self.device, dtype=torch.float32)
        corners[:, 0, :] = torch.stack([local_min[:, 0], local_min[:, 1], local_min[:, 2]], dim=1)
        corners[:, 1, :] = torch.stack([local_max[:, 0], local_min[:, 1], local_min[:, 2]], dim=1)
        corners[:, 2, :] = torch.stack([local_min[:, 0], local_max[:, 1], local_min[:, 2]], dim=1)
        corners[:, 3, :] = torch.stack([local_max[:, 0], local_max[:, 1], local_min[:, 2]], dim=1)
        corners[:, 4, :] = torch.stack([local_min[:, 0], local_min[:, 1], local_max[:, 2]], dim=1)
        corners[:, 5, :] = torch.stack([local_max[:, 0], local_min[:, 1], local_max[:, 2]], dim=1)
        corners[:, 6, :] = torch.stack([local_min[:, 0], local_max[:, 1], local_max[:, 2]], dim=1)
        corners[:, 7, :] = torch.stack([local_max[:, 0], local_max[:, 1], local_max[:, 2]], dim=1)

        # Rotate corners: [N, 8, 3] @ [N, 3, 3]^T -> [N, 8, 3]
        rotated_corners = torch.bmm(corners, rot_matrices.transpose(1, 2))

        # Translate to world coordinates (actual_positions is the centroid world coordinate)
        world_corners = rotated_corners + actual_positions.unsqueeze(1)  # [N, 8, 3]

        # Compute AABB (scaled world coordinates)
        world_min = world_corners.min(dim=1)[0]  # [N, 3]
        world_max = world_corners.max(dim=1)[0]  # [N, 3]

        # 4. Convert to bin local coordinates (unscaled)
        # The bin bottom has a z_offset=0.1 in world coordinates; subtract before dividing by scale
        z_offset = 0.1  # consistent with z_offset in Interface_isaacgym._create_env
        world_min_adjusted = world_min.clone()
        world_max_adjusted = world_max.clone()
        world_min_adjusted[:, 2] = world_min[:, 2] - z_offset
        world_max_adjusted[:, 2] = world_max[:, 2] - z_offset

        # Divide world coordinates by scale to get bin local coordinates
        bounds_local_min = world_min_adjusted / scale  # [N, 3]
        bounds_local_max = world_max_adjusted / scale  # [N, 3]

        # Apply X/Y safety margin (enlarge occupancy range)
        if xy_margin > 0:
            bounds_local_min[:, 0] = bounds_local_min[:, 0] - xy_margin
            bounds_local_min[:, 1] = bounds_local_min[:, 1] - xy_margin
            bounds_local_max[:, 0] = bounds_local_max[:, 0] + xy_margin
            bounds_local_max[:, 1] = bounds_local_max[:, 1] + xy_margin

        # 5. Compute height map grid indices
        zeros = torch.zeros(3, device=self.device, dtype=torch.float32)
        bounds_min_clamped = torch.maximum(bounds_local_min, zeros)
        bounds_max_clamped = torch.minimum(bounds_local_max, self.bin_dimension)

        minBoundsInt = torch.floor(bounds_min_clamped / self.resolutionH).to(torch.int32)  # [N, 3]
        maxBoundsInt = torch.ceil(bounds_max_clamped / self.resolutionH).to(torch.int32)  # [N, 3]

        # 6. Compute item occupancy range in height map
        rangeX_O = (maxBoundsInt[:, 0] - minBoundsInt[:, 0])  # [N]
        rangeY_O = (maxBoundsInt[:, 1] - minBoundsInt[:, 1])  # [N]

        # Filter valid placements (range > 0)
        valid_placement = (rangeX_O > 0) & (rangeY_O > 0)
        if not valid_placement.any():
            return

        # Filter valid items
        if not valid_placement.all():
            env_ids = env_ids[valid_placement]
            minBoundsInt = minBoundsInt[valid_placement]
            maxBoundsInt = maxBoundsInt[valid_placement]
            rangeX_O = rangeX_O[valid_placement]
            rangeY_O = rangeY_O[valid_placement]
            bounds_local_max = bounds_local_max[valid_placement]
            num_items = len(env_ids)

        max_Sx = rangeX_O.max().item()
        max_Sy = rangeY_O.max().item()

        # 7. Item top height (bin local coordinates, unscaled) + safety margin
        top_height = bounds_local_max[:, 2] + height_margin  # [N]

        # 8. Build grid indices for batch update
        dx = torch.arange(max_Sx, device=self.device, dtype=torch.long)
        dy = torch.arange(max_Sy, device=self.device, dtype=torch.long)
        dx_grid, dy_grid = torch.meshgrid(dx, dy, indexing='ij')  # [max_Sx, max_Sy]

        # Expand to [N, max_Sx, max_Sy]
        dx_grid = dx_grid.unsqueeze(0).expand(num_items, -1, -1)
        dy_grid = dy_grid.unsqueeze(0).expand(num_items, -1, -1)

        # Compute absolute grid coordinates
        coorX = minBoundsInt[:, 0].unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        coorY = minBoundsInt[:, 1].unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        x_indices = coorX + dx_grid  # [N, max_Sx, max_Sy]
        y_indices = coorY + dy_grid  # [N, max_Sx, max_Sy]

        # Valid region mask
        rangeX_exp = rangeX_O.unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        rangeY_exp = rangeY_O.unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
        valid_region = (dx_grid < rangeX_exp) & (dy_grid < rangeY_exp)  # [N, max_Sx, max_Sy]

        # Boundary check
        x_indices = torch.clamp(x_indices, 0, self.rangeX_C - 1)
        y_indices = torch.clamp(y_indices, 0, self.rangeY_C - 1)

        # Environment indices
        env_indices = env_ids.unsqueeze(1).unsqueeze(2).expand(-1, max_Sx, max_Sy)  # [N, max_Sx, max_Sy]

        # Height values expanded
        height_expanded = top_height.unsqueeze(1).unsqueeze(2).expand(-1, max_Sx, max_Sy)  # [N, max_Sx, max_Sy]

        # 9. Update height map (take maximum)
        current_heights = self.heightmapC[env_indices, x_indices, y_indices]  # [N, max_Sx, max_Sy]
        new_heights = torch.where(valid_region, torch.maximum(current_heights, height_expanded), current_heights)
        self.heightmapC[env_indices, x_indices, y_indices] = new_heights

    def get_possible_position(self, next_item_ID, device=None):
        """
        Compute all possible placement positions (multi-environment vectorized version)

        Consistent with get_possible_position logic in the original space.py:
        1. Iterate over each rotation angle (vectorized)
        2. Compute posZ = max((heightmapC - heightMapB) * maskB)
        3. Check posZ + boundingSize[2] <= bin_dimension[2]
        4. Check item stays within bin bounds
        5. Update naiveMask and posZmap

        Args:
            next_item_ID: [num_envs] item ID for each environment
            device: compute device; if None, use self.device

        Returns:
            naiveMask: [num_envs, rotNum, rangeX_A, rangeY_A] feasible position mask
        """
        if device is None:
            device = self.device

        # Ensure next_item_ID is long type
        if not isinstance(next_item_ID, torch.Tensor):
            next_item_ID = torch.tensor(next_item_ID, device=device, dtype=torch.long)
        next_item_ID = next_item_ID.long()

        num_envs = self.num_envs
        rotNum = self.rotNum

        # Initialize posZ
        self.posZ[:] = 1e3

        # 1. Get item sizes per environment per rotation angle
        # rangeX_OH/rangeY_OH: [obj_nums, rotNum] height map grid size
        # rangeX_OA/rangeY_OA: [obj_nums, rotNum] action space grid size
        rangeX_OH = self.rangeX_OH[next_item_ID, :].to(torch.long)  # [num_envs, rotNum]
        rangeY_OH = self.rangeY_OH[next_item_ID, :].to(torch.long)
        rangeX_OA = self.rangeX_OA[next_item_ID, :]  # [num_envs, rotNum]
        rangeY_OA = self.rangeY_OA[next_item_ID, :]

        # Find maximum sizes for batch processing
        max_Sx = rangeX_OH.max().item()
        max_Sy = rangeY_OH.max().item()

        if max_Sx == 0 or max_Sy == 0:
            naiveMask = torch.zeros((num_envs, rotNum, self.rangeX_A, self.rangeY_A),
                                    device=device, dtype=torch.int32)
            self.posZvalid[:] = 1e3
            return naiveMask

        # 2. Generate grid indices
        dx = torch.arange(max_Sx, device=device, dtype=torch.long)
        dy = torch.arange(max_Sy, device=device, dtype=torch.long)
        dx_grid, dy_grid = torch.meshgrid(dx, dy, indexing='ij')  # [max_Sx, max_Sy]

        # 3. Compute height map starting coordinates per action position
        # Matches original: coorX, coorY = X * self.stepSize, Y * self.stepSize
        action_x = torch.arange(self.rangeX_A, device=device, dtype=torch.long) * self.stepSize  # [rangeX_A]
        action_y = torch.arange(self.rangeY_A, device=device, dtype=torch.long) * self.stepSize  # [rangeY_A]

        # Expand to [num_envs, rotNum, rangeX_A, rangeY_A, max_Sx, max_Sy]
        coorX = action_x.view(1, 1, self.rangeX_A, 1, 1, 1)
        coorY = action_y.view(1, 1, 1, self.rangeY_A, 1, 1)
        dx_exp = dx_grid.view(1, 1, 1, 1, max_Sx, max_Sy)
        dy_exp = dy_grid.view(1, 1, 1, 1, max_Sx, max_Sy)

        # Compute height map indices
        x_indices = coorX + dx_exp  # [1, 1, rangeX_A, 1, max_Sx, max_Sy]
        y_indices = coorY + dy_exp  # [1, 1, 1, rangeY_A, 1, max_Sy]

        # Broadcast to full shape
        x_indices = x_indices.expand(num_envs, rotNum, self.rangeX_A, self.rangeY_A, max_Sx, max_Sy)
        y_indices = y_indices.expand(num_envs, rotNum, self.rangeX_A, self.rangeY_A, max_Sx, max_Sy)

        # Clamp to bounds
        x_indices = torch.clamp(x_indices, 0, self.rangeX_C - 1)
        y_indices = torch.clamp(y_indices, 0, self.rangeY_C - 1)

        # 4. Extract height map region
        # heightmapC: [num_envs, rangeX_C, rangeY_C]
        env_indices = torch.arange(num_envs, device=device).view(num_envs, 1, 1, 1, 1, 1)
        env_indices = env_indices.expand(num_envs, rotNum, self.rangeX_A, self.rangeY_A, max_Sx, max_Sy)

        region_heightmap = self.heightmapC[env_indices, x_indices, y_indices]
        # region_heightmap: [num_envs, rotNum, rangeX_A, rangeY_A, max_Sx, max_Sy]

        # 5. Get item bottom height map and mask
        # Matches original: heightMapB, maskB = self.shotInfo[next_item_ID][rotIdx]
        heightMapB = self.heightMapB[next_item_ID, :, :max_Sx, :max_Sy]  # [num_envs, rotNum, max_Sx, max_Sy]
        maskB = self.maskB[next_item_ID, :, :max_Sx, :max_Sy]

        # Expand dimensions
        heightMapB_exp = heightMapB.unsqueeze(2).unsqueeze(3)  # [num_envs, rotNum, 1, 1, max_Sx, max_Sy]
        maskB_exp = maskB.unsqueeze(2).unsqueeze(3)

        # 6. Create valid region mask (actual size differs per item rotation)
        pixel_x = torch.arange(max_Sx, device=device).view(1, 1, 1, 1, max_Sx, 1)
        pixel_y = torch.arange(max_Sy, device=device).view(1, 1, 1, 1, 1, max_Sy)
        rangeX_exp = rangeX_OH.view(num_envs, rotNum, 1, 1, 1, 1)
        rangeY_exp = rangeY_OH.view(num_envs, rotNum, 1, 1, 1, 1)

        valid_pixel_mask = (pixel_x < rangeX_exp) & (pixel_y < rangeY_exp)
        # valid_pixel_mask: [num_envs, rotNum, 1, 1, max_Sx, max_Sy]

        # 7. Compute posZ (original logic)
        # Matches original: posZ = np.max((self.heightmapC[...] - heightMapB) * maskB)
        # Only considers regions where the item bottom has material (maskB=1)
        diff = (region_heightmap - heightMapB_exp) * maskB_exp
        diff = torch.where(valid_pixel_mask, diff, torch.tensor(-1e6, device=device, dtype=diff.dtype))

        posZ_from_bottom = diff.view(num_envs, rotNum, self.rangeX_A, self.rangeY_A, -1).max(dim=-1)[0]

        # 7.5 [Key improvement] Account for drop path to avoid edge collisions
        # Problem: original posZ only considers regions where maskB=1
        # For an L-shaped item, the concave part (maskB=0) may have taller obstacles
        # The item edges will collide with these obstacles when dropping
        #
        # Solution: take the maximum of both:
        # 1. posZ_from_bottom: height computed from bottom contact
        # 2. max_height_in_aabb: maximum height map value within AABB
        # This ensures the item starts from a high enough position to avoid edge collisions

        # Compute maximum height map value within AABB (all pixels)
        heightmap_masked = torch.where(valid_pixel_mask, region_heightmap,
                                       torch.tensor(-1e6, device=device, dtype=region_heightmap.dtype))
        max_height_in_aabb = heightmap_masked.view(num_envs, rotNum, self.rangeX_A, self.rangeY_A, -1).max(dim=-1)[0]

        # posZ = max(bottom contact height, highest point in AABB)
        # This ensures the item starts above obstacles and avoids edge collisions
        posZ = torch.maximum(posZ_from_bottom, max_height_in_aabb)
        # posZ: [num_envs, rotNum, rangeX_A, rangeY_A]

        # 8. Height check
        # Matches original: if posZ + boundingSize[2] - self.bin_dimension[2] <= 0
        boundingSize_z = self.boundingSize[next_item_ID, :, 2]  # [num_envs, rotNum]
        boundingSize_z_exp = boundingSize_z.view(num_envs, rotNum, 1, 1)
        bin_height = self.bin_dimension[2]

        # Use round to avoid floating point errors; 1e-6 tolerance matches prejudge in binPhy
        height_diff = torch.round((posZ + boundingSize_z_exp - bin_height) * 1e6) / 1e6
        height_valid = height_diff <= 1e-6

        # 9. Boundary check (add boundary margin to prevent collision with bin walls)
        # Matches original: for X in range(self.rangeX_A - rangeX_OA + 1)
        action_pos_x = torch.arange(self.rangeX_A, device=device).view(1, 1, self.rangeX_A, 1)
        action_pos_y = torch.arange(self.rangeY_A, device=device).view(1, 1, 1, self.rangeY_A)

        # Boundary margin: shrink the placeable range inward (unit: action grid cells)
        # Adding margin prevents items from exceeding bin boundaries
        boundary_margin = 0  # 0 grid cells (item size check is already in naiveMask)

        max_action_x = (self.rangeX_A - rangeX_OA - boundary_margin).view(num_envs, rotNum, 1, 1)
        max_action_y = (self.rangeY_A - rangeY_OA - boundary_margin).view(num_envs, rotNum, 1, 1)
        min_action_x = boundary_margin
        min_action_y = boundary_margin

        # Note: original uses < (range - rangeOA + 1), equivalent to <= (range - rangeOA)
        # Now adding min and max constraints
        boundary_valid = (action_pos_x >= min_action_x) & (action_pos_x <= max_action_x) & \
                         (action_pos_y >= min_action_y) & (action_pos_y <= max_action_y)

        # 10. Combine conditions to generate naiveMask
        naiveMask = (height_valid & boundary_valid).to(torch.int32)

        # 11. Store posZ
        self.posZ[:] = posZ

        # 12. Update posZvalid (set invalid positions to 1e3)
        # Matches original: self.posZValid[invalidIndex] = 1e3
        self.posZvalid[:] = posZ.clone()
        self.posZvalid[naiveMask == 0] = 1e3

        # Save naiveMask for heuristic methods
        self.naiveMask = naiveMask

        return naiveMask




if __name__ == "__main__":
    args = get_args()
    space = SpaceIsaacGym(bin_dimension=[0.5, 0.5, 0.5], resolutionAct=0.02, resolutionH=0.01, boxPack=False, ZRotNum=8,
                                                    shotInfo=None, scale=[5, 5, 5], device=0, num_envs=4,args=args)
    a = space.get_possible_position(next_item_ID=torch.tensor([0,2,5,14], device=space.device), device=space.device)
    print(a)

    # Method 1: use operator.itemgetter (closest to "direct indexing", cleanest syntax, best performance)
    # Returns tuple: (shapeDict[0], shapeDict[1], shapeDict[2], shapeDict[3])
    # shapes = itemgetter(*index)(args.shapeDict)
    # print("Method 1 - itemgetter:", shapes)

    # # Method 2: use map function (functional style, returns list)
    # shapes_map = list(map(args.shapeDict.__getitem__, index))
    # print("Method 2 - map:", shapes_map)

    # # Method 3: use dict get method with map (safer, returns None for missing keys)
    # shapes_get = list(map(args.shapeDict.get, index))
    # print("Method 3 - map with get:", shapes_get)
