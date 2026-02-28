import math
import numpy as np
import trimesh
from isaacgym import gymapi, gymtorch, gymutil
import torch
import os
from isaacgymenvs.utils.torch_jit_utils import to_torch, quaternion_to_matrix
from .IRcreator import RandomItemCreator, RandomCateCreator, RandomInstanceCreator

class IsaacGymInterface:
    """Physics simulation interface for Isaac Gym"""

    def __init__(self, sim_device, graphics_device_id, headless,
                bin = [0.5, 0.5, 0.5], scale = [5.0, 5.0, 5.0], meshScale = 1.0,
                foldername = '/home/zzx/Documents/IR-BPP/dataset/blockout/shape_vhacd',
                maxBatch = 2,num_envs = 4):

        self.foldername = foldername
        if not os.path.exists(self.foldername):
            os.mkdir(foldername)

        # Scale factors
        self.bin = torch.tensor(bin,device=sim_device)  # Default bin dimensions
        self.defaultScale = torch.tensor(scale,device=sim_device)
        self.meshScale = meshScale  # Object scale factor

        self.num_envs = num_envs
        self.sim_device = sim_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless

        # Create simulation environment
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -10.0)  # Original PyBullet gravity
        self.sim_params.dt = 1.0 / 60.0

        # PhysX parameters: refer to official example settings
        self.sim_params.physx.solver_type = 1                # TGS solver
        self.sim_params.physx.num_position_iterations = 8    # Position iterations
        self.sim_params.physx.num_velocity_iterations = 1    # Velocity iterations
        self.sim_params.physx.contact_offset = 0.002         # Contact detection distance (official: 0.001-0.005)
        self.sim_params.physx.rest_offset = 0.0              # Rest offset
        self.sim_params.physx.friction_offset_threshold = 0.001
        self.sim_params.physx.friction_correlation_distance = 0.0005

        # if not self.headless:
        self.sim_params.use_gpu_pipeline = True

        self.sim = self.gym.create_sim(self.sim_device, self.graphics_device_id,
                                     gymapi.SIM_PHYSX, self.sim_params)

        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
            # Bind spacebar to pause
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")
            self.paused = False

        self.shapeMap = {} # Store loaded 3D shapes
        self.assets = {} # Store all loaded assets

        self.containerFolder = self.foldername + '/../box_{}_{}_{}'.format(*self.bin)
        if not os.path.exists(self.containerFolder):
            os.mkdir(self.containerFolder)

        # Load id2shape.pt file to get shape ordering
        self.id2shape = torch.load(os.path.join(self.foldername, '../id2shape.pt'))

        # Load all URDF files
        self._load_all_assets()

        # Create environments
        self._create_env()
        self._create_ground_plane()

        self.gym.prepare_sim(self.sim)
        self.step_simulation(steps=1)

        # Get root state tensor for all actors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.actor_root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.init_actor_root_state_tensor = torch.clone(self.actor_root_state_tensor)

        # Get global indices for all actors
        self.global_indices = torch.arange(self.num_envs*(len(self.actor_handles)//self.num_envs+5),device=self.sim_device,dtype=torch.int32).view(self.num_envs, -1)

        # Environment indices and object indices
        self.envs_indices = torch.arange(self.num_envs, device=self.sim_device)
        self.target_actor_indices = torch.randint(6, int(len(self.actor_handles)/self.num_envs)+5,(self.num_envs,),dtype=torch.int32) # random 6-29

        # Store linear velocity, angular velocity, and state for all objects
        self.linear_vel = torch.zeros(self.num_envs, int(len(self.actor_handles)/self.num_envs)+5, 3, device=self.sim_device)
        self.angular_vel = torch.zeros(self.num_envs, int(len(self.actor_handles)/self.num_envs)+5, 3, device=self.sim_device)

        self.AABBCompensation = torch.tensor([0.002, 0.002, 0.002],device=sim_device)
        self.meshDict={} # Store mesh data for all objects
        self.maxBatch = maxBatch

        self.reset_buf = torch.zeros(self.num_envs, device=self.sim_device)

        # Track disabled (fixed) objects in each environment
        # disabled_actors[env_idx, actor_idx] = 1 means the object is disabled
        num_actors_per_env = int(len(self.actor_handles) / self.num_envs) + 5 if hasattr(self, 'actor_handles') else 30
        self.disabled_actors = torch.zeros(self.num_envs, num_actors_per_env, device=self.sim_device, dtype=torch.bool)

        # Store fixed state (position and rotation) of disabled objects
        self.disabled_states = torch.zeros(self.num_envs, num_actors_per_env, 7, device=self.sim_device)

        # Load mesh data for all objects
        self._load_all_meshes()

    def _load_all_assets(self):
        """Load all URDF assets"""
        import glob

        # Get all URDF files
        urdf_files = glob.glob(os.path.join(self.foldername, "*.urdf"))

        # Create asset options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  # Allow objects to move
        asset_options.disable_gravity = False
        asset_options.density = 1.0  # Default density
        asset_options.linear_damping = 0.5   # Original PyBullet setting
        asset_options.angular_damping = 0.5  # Original PyBullet setting
        asset_options.use_mesh_materials = False

        # Disable VHACD convex decomposition (mesh already convex-decomposed in shape_vhacd directory)
        # This ensures collision shapes are consistent with the original mesh
        asset_options.vhacd_enabled = False
        asset_options.convex_decomposition_from_submeshes = True  # Use submesh as convex hull

        # Load each URDF file
        for idx, asset_name in self.id2shape.items():
            urdf_file = os.path.join(self.foldername, asset_name[0:-4] + '.urdf')
            asset_root = os.path.dirname(urdf_file)
            asset_filename = os.path.basename(urdf_file)
            asset_name = os.path.splitext(asset_filename)[0]  # Remove .urdf extension

            try:
                asset = self.gym.load_asset(self.sim, asset_root, asset_filename, asset_options)
                if asset is not None:
                    self.assets[asset_name] = asset
                    print(f"Loaded asset: {asset_name}")
                else:
                    print(f"Failed to load asset: {asset_name}")
            except Exception as e:
                print(f"Error loading asset {asset_name}: {e}")

        print(f"Total loaded assets: {len(self.assets)}")

    def _load_all_meshes(self):
        """Load mesh data for all objects into meshDict"""
        print("Loading mesh data for all assets...")

        # Use max key + 1 from id2shape to determine array size, ensuring consistent indexing
        max_idx = max(self.id2shape.keys()) if self.id2shape else 0
        num_slots = max_idx + 1 + 5  # +5 for the bin walls at the front
        self.mesh_bounds = torch.zeros(num_slots, 2, 4, device=self.sim_device)

        # Store centroid for each mesh (relative to OBJ origin, scaled)
        # This is the key to staying consistent with shapeDict!
        self.mesh_centroids = torch.zeros(num_slots, 3, device=self.sim_device)

        # Use key (idx) from id2shape as index, ensuring consistency with item_type_ids
        for idx, asset_name in self.id2shape.items():
            actor_idx = idx + 5  # actor index = item_type_id + 5
            # asset_name may be "xxx.obj" or "xxx"
            if asset_name.endswith('.obj'):
                obj_path = os.path.join(self.foldername, asset_name)
            else:
                obj_path = os.path.join(self.foldername, asset_name + '.obj')

            if os.path.exists(obj_path):
                try:
                    # Load mesh
                    mesh = trimesh.load(obj_path)

                    # Apply scaling: defaultScale * meshScale
                    scale_factors = self.defaultScale.cpu().numpy() * self.meshScale
                    mesh.apply_scale(scale_factors)

                    # Store centroid (scaled, relative to OBJ origin)
                    # This is the offset corresponding to mesh.apply_translation(-mesh.centroid) in load_mesh_plain
                    self.mesh_centroids[actor_idx] = torch.tensor(mesh.centroid, device=self.sim_device)

                    # Store mesh
                    self.meshDict[actor_idx] = mesh.copy()

                    # Store mesh bounds (relative to OBJ origin)
                    self.mesh_bounds[actor_idx, :, 0:3] = torch.tensor(mesh.bounds, device=self.sim_device)
                    self.mesh_bounds[actor_idx, :, 3] = 1.0

                except Exception as e:
                    print(f"Failed to load mesh for {asset_name}: {e}")
                    self.meshDict[actor_idx] = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
            else:
                print(f"OBJ file not found for {asset_name}: {obj_path}")
                self.meshDict[actor_idx] = trimesh.creation.box(extents=[0.1, 0.1, 0.1])

        print(f"Total loaded meshes: {len(self.meshDict)}")

    def _create_env(self, spacing=8.0, num_per_row=None):
        """Create simulation environments"""
        # Set environment layout
        if num_per_row is None:
            num_per_row = int(math.sqrt(self.num_envs))

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Calculate actual bin dimensions (considering scaling)
        scaled_bin = (self.bin * self.defaultScale).to(torch.float32)
        wall_thickness = 0.02  # Reduce wall thickness (2cm) to reduce collision interference
        z_offset = 0.1
        # Create box assets
        box_asset_options = gymapi.AssetOptions()
        box_asset_options.fix_base_link = True
        box_asset_options.disable_gravity = False

        # Bottom
        bottom_dims = gymapi.Vec3(scaled_bin[0] + 2*wall_thickness,
                                scaled_bin[1] + 2*wall_thickness, wall_thickness)
        bottom_asset = self.gym.create_box(self.sim, bottom_dims.x, bottom_dims.y,
                                            bottom_dims.z, box_asset_options)
        bottom_pose = gymapi.Transform()
        bottom_pose.p = gymapi.Vec3(scaled_bin[0]/2, scaled_bin[1]/2, -wall_thickness/2 + z_offset)

        # Left and right walls
        wall_dims = gymapi.Vec3(wall_thickness, scaled_bin[1], scaled_bin[2])
        wall_asset = self.gym.create_box(self.sim, wall_dims.x, wall_dims.y,
                                        wall_dims.z, box_asset_options)

        # Left wall
        left_pose = gymapi.Transform()
        left_pose.p = gymapi.Vec3(-wall_thickness/2, scaled_bin[1]/2, scaled_bin[2]/2 + z_offset)

        # Right wall
        right_pose = gymapi.Transform()
        right_pose.p = gymapi.Vec3(scaled_bin[0] + wall_thickness/2, scaled_bin[1]/2, scaled_bin[2]/2 + z_offset)

        # Front and back walls
        fb_wall_dims = gymapi.Vec3(scaled_bin[0], wall_thickness, scaled_bin[2])
        fb_wall_asset = self.gym.create_box(self.sim, fb_wall_dims.x, fb_wall_dims.y,
                                            fb_wall_dims.z, box_asset_options)

        # Front wall
        front_pose = gymapi.Transform()
        front_pose.p = gymapi.Vec3(scaled_bin[0]/2, scaled_bin[1] + wall_thickness/2, scaled_bin[2]/2 + z_offset)

        # Back wall
        back_pose = gymapi.Transform()
        back_pose.p = gymapi.Vec3(scaled_bin[0]/2, -wall_thickness/2, scaled_bin[2]/2 + z_offset)

        # Object handles
        self.envs = []
        self.bottom_handles = []
        self.left_handles = []
        self.right_handles = []
        self.front_handles = []
        self.back_handles = []
        self.actor_handles = []

        # Calculate grid layout for objects on the ground
        num_objects = len(self.assets)
        # Calculate number of rows and columns for the grid (approximately square)
        grid_cols = int(math.ceil(math.sqrt(num_objects)))
        grid_rows = int(math.ceil(num_objects / grid_cols))

        # Object spacing (ensure no collision, more compact than before)
        object_spacing = 0.15 * self.defaultScale[0]  # spacing = 0.15 * scale

        # Start position: place objects beside the bin
        start_x = float(scaled_bin[0]) + 2.0  # Start 2m to the right of the bin
        start_y = -2.0  # Slight offset
        ground_z = 0.1 * self.defaultScale[0]  # Height of object bottom above ground

        # Create all environments
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            bottom_handle = self.gym.create_actor(env_ptr, bottom_asset, bottom_pose, "bottom_wall", 0, 0)
            left_handle = self.gym.create_actor(env_ptr, wall_asset, left_pose, "left_wall", 0, 0)
            right_handle = self.gym.create_actor(env_ptr, wall_asset, right_pose, "right_wall", 0, 0)
            front_handle = self.gym.create_actor(env_ptr, fb_wall_asset, front_pose, "front_wall", 0, 0)
            back_handle = self.gym.create_actor(env_ptr, fb_wall_asset, back_pose, "back_wall", 0, 0)

            # Set rigid body properties for walls: high friction, low restitution, prevent objects from bouncing out
            wall_handles = [bottom_handle, left_handle, right_handle, front_handle, back_handle]
            for wh in wall_handles:
                shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, wh)
                for sp in shape_props:
                    sp.friction = 1.0          # High friction
                    sp.restitution = 0.0       # Zero restitution (no bounce)
                    sp.compliance = 0.0        # Zero compliance
                self.gym.set_actor_rigid_shape_properties(env_ptr, wh, shape_props)

            # Create one actor per asset, in the order of id2shape keys
            # This ensures actor_indices = item_type_id + 5 maps correctly
            for idx in sorted(self.id2shape.keys()):
                asset_name_full = self.id2shape[idx]
                # Remove .obj extension to get the asset key
                asset_name = asset_name_full[:-4] if asset_name_full.endswith('.obj') else asset_name_full
                asset = self.assets.get(asset_name)

                if asset is None:
                    print(f"Warning: asset {asset_name} not found for idx {idx}")
                    continue

                # Calculate position of the object in the grid
                row = idx // grid_cols
                col = idx % grid_cols

                # Calculate actual position
                pos_x = start_x + col * object_spacing
                pos_y = start_y + row * object_spacing
                pos_z = ground_z  # Place on the ground

                initial_pose = gymapi.Transform()
                initial_pose.p = gymapi.Vec3(pos_x, pos_y, pos_z)
                initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # No rotation

                actor_handle = self.gym.create_actor(env_ptr, asset, initial_pose, f"{asset_name}_env_{i}", 0, 0)

                # Object scale = defaultScale * meshScale
                self.gym.set_actor_scale(env_ptr, actor_handle, self.defaultScale[0].item() * self.meshScale)

                # Set rigid body properties for objects: low friction, low restitution
                # Set object physical properties (consistent with original PyBullet defaults)
                shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
                for sp in shape_props:
                    sp.friction = 0.5          # PyBullet default friction
                    sp.restitution = 0.0       # Zero restitution (no bounce)
                self.gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, shape_props)

                self.actor_handles.append(actor_handle)

            self.bottom_handles.append(bottom_handle)
            self.left_handles.append(left_handle)
            self.right_handles.append(right_handle)
            self.front_handles.append(front_handle)
            self.back_handles.append(back_handle)
            self.envs.append(env_ptr)

    def _create_ground_plane(self):
        """Create ground plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0

        self.gym.add_ground(self.sim, plane_params)

    def simulateToQuasistatic(self, placed_actor_indices=None, linear_tol=0.001, angular_tol=0.001,
                               max_iterations=500, min_iterations=10):
        """
        Simulate until quasi-static equilibrium (parallel multi-environment, each environment judged independently)

        Args:
            placed_actor_indices: [num_envs] Index of the currently placed object, used to check velocity and bounds
                                  Index < 5 means no object is placed in that environment, no need to check
                                  If None, return directly (no simulation)
            linear_tol: Linear velocity tolerance
            angular_tol: Angular velocity tolerance
            max_iterations: Maximum number of iterations
            min_iterations: Minimum number of iterations, ensures objects have enough time to fall

        Returns:
            done_buf: [num_envs] Simulation completion flag (all True)
            success_buf: [num_envs] Placement success flag (True = success, False = out of bounds)
        """
        # Initialize results
        done_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.sim_device)
        success_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.sim_device)

        # If no placed objects, return directly
        if placed_actor_indices is None:
            return done_buf, success_buf

        # Convert placed_actor_indices
        if not isinstance(placed_actor_indices, torch.Tensor):
            placed_actor_indices = torch.tensor(placed_actor_indices, device=self.sim_device, dtype=torch.long)

        # Get valid environment mask (index >= 5 means an object is placed)
        valid_env_mask = placed_actor_indices >= 5  # [num_envs]

        # If all environments have no placed objects, return directly
        if not valid_env_mask.any():
            return done_buf, success_buf

        linear_tol_sqr = linear_tol * linear_tol
        angular_tol_sqr = angular_tol * angular_tol

        env_indices = torch.arange(self.num_envs, device=self.sim_device)
        scaled_bin = self.bin * self.defaultScale

        # Whether each environment has come to rest (invalid environments are considered at rest)
        env_end = ~valid_env_mask  # [num_envs], invalid environments initialized as done

        iteration = 0

        # Simulate until all valid environments are at rest or max iterations reached
        while not env_end.all() and iteration < max_iterations:
            self.step_simulation()
            # Only render occasionally in non-headless mode to avoid performance issues
            if not self.headless and iteration % 10 == 0:
                self.render()
            iteration += 1

            # Do not check velocity within minimum iterations, ensure objects have enough time to fall
            if iteration < min_iterations:
                continue

            # Check whether each environment's velocity has reached rest
            linear_vel = self.actor_root_state_tensor[env_indices, placed_actor_indices, 7:10]  # [num_envs, 3]
            angular_vel = self.actor_root_state_tensor[env_indices, placed_actor_indices, 10:13]  # [num_envs, 3]

            linear_vel_sqr = torch.sum(linear_vel ** 2, dim=1)  # [num_envs]
            angular_vel_sqr = torch.sum(angular_vel ** 2, dim=1)  # [num_envs]

            # Each environment independently judges whether it is at rest
            env_static = (linear_vel_sqr <= linear_tol_sqr) & (angular_vel_sqr <= angular_tol_sqr)

            # Update environments that have come to rest (once at rest, state does not change)
            env_end = env_end | env_static

        # Check bounds (only check valid environments)
        mesh_bounds = self.get_trimesh_AABB(placed_actor_indices)  # [num_envs, 2, 4]

        placed_min = mesh_bounds[:, 0, :3]  # [num_envs, 3]
        placed_max = mesh_bounds[:, 1, :3]  # [num_envs, 3]
        midC = (placed_max + placed_min) / 2  # [num_envs, 3]

        # Bounds check (consistent with original: midC <= 0 or midC >= bin)
        out_of_bounds = (midC[:, 0] <= 0) | (midC[:, 0] >= scaled_bin[0]) | \
                       (midC[:, 1] <= 0) | (midC[:, 1] >= scaled_bin[1]) | \
                       (midC[:, 2] <= 0) | (midC[:, 2] >= scaled_bin[2])

        # Only mark out-of-bounds valid environments as failed
        success_buf[valid_env_mask & out_of_bounds] = False

        return done_buf, success_buf

    def simulateHeight(self, placed_actor_indices=None):
        """
        Check whether object height exceeds bin bounds (parallel multi-environment, each environment judged independently)

        Consistent with the simulateHeight function logic in the original Interface.py:
        Check if maxC[2] - bin[2] > 0, then height is exceeded

        Args:
            placed_actor_indices: [num_envs] Indices of objects to check
                                  Index < 5 means no object is placed in that environment, no need to check
                                  If None, return directly (no check)

        Returns:
            done_buf: [num_envs] Check completion flag (all True)
            success_buf: [num_envs] Height check pass flag (True = not exceeded, False = height exceeded)
        """
        # Initialize results
        done_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.sim_device)
        success_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.sim_device)

        # If no placed objects, return directly
        if placed_actor_indices is None:
            return done_buf, success_buf

        # Convert placed_actor_indices
        if not isinstance(placed_actor_indices, torch.Tensor):
            placed_actor_indices = torch.tensor(placed_actor_indices, device=self.sim_device, dtype=torch.long)

        # Get valid environment mask (index >= 5 means an object is placed)
        valid_env_mask = placed_actor_indices >= 5  # [num_envs]

        # If all environments have no placed objects, return directly
        if not valid_env_mask.any():
            return done_buf, success_buf

        # Use scaled bin dimensions (consistent with original)
        scaled_bin = self.bin * self.defaultScale

        # Get AABB
        mesh_bounds = self.get_trimesh_AABB(placed_actor_indices)  # [num_envs, 2, 4]

        # Get maximum Z coordinate of the placed object
        placed_max = mesh_bounds[:, 1, :3]  # [num_envs, 3]

        # Check whether height is exceeded (consistent with original: maxC[2] - bin[2] > 0)
        # Use round to 6 decimal places for comparison
        height_diff = placed_max[:, 2] - scaled_bin[2]
        out_of_height = torch.round(height_diff * 1e6) / 1e6 > 0

        # Only mark valid environments that exceeded height as failed
        success_buf[valid_env_mask & out_of_height] = False

        return done_buf, success_buf

    def refresh_tensor(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def step_simulation(self, steps=1):
        """Execute simulation steps"""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.refresh_tensor()

    def reset(self, env_ids): # env_ids: list of environment indices
        """
        Reset all actors in the specified environments to their initial state
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim_device, dtype=torch.long)

        if len(env_ids) == 0:
            return

        # Get global indices of actors to reset
        prop_indices = self.global_indices[env_ids, :].flatten().to(torch.int32)

        # Get flat view of initial states
        init_states_flat = self.init_actor_root_state_tensor.view(-1, 13)

        # Copy initial states to current state tensor (only for environments to reset)
        # Get flat view of current states first
        current_states_flat = self.actor_root_state_tensor.view(-1, 13)

        # Copy initial states to current states (by index)
        current_states_flat[prop_indices.long()] = init_states_flat[prop_indices.long()]

        # Apply state update to physics engine
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(current_states_flat),
            gymtorch.unwrap_tensor(prop_indices),
            len(prop_indices)
        )

        # Clear disabled object records for reset environments
        self.disabled_actors[env_ids] = False
        self.disabled_states[env_ids] = 0

    def render(self):
        """Render (supports spacebar to pause rendering while training continues)"""
        if not self.headless:
            # Check keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "pause" and evt.value > 0:
                    self.paused = not getattr(self, 'paused', False)
                    print(f"Rendering {'paused' if self.paused else 'resumed'} (training continues)")

            # Return directly when rendering is paused, training continues
            if getattr(self, 'paused', False):
                # Still need to process window events, otherwise window becomes unresponsive
                self.gym.poll_viewer_events(self.viewer)
                return

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def close(self):
        """Clean up resources"""
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def get_trimesh_AABB(self, placed_actor_indices=None):
        """
        Get trimesh bounding box of placed objects in specified environments (parallel multi-environment)

        Args:
            placed_actor_indices: [num_envs] Index of the placed object in each environment
                                  Index < 5 means no object is placed in that environment
                                  If None, return bounding boxes for all actors

        Returns:
            If placed_actor_indices is None:
                bounds: [num_envs, num_actors, 2, 4] bounding boxes for all actors
            If placed_actor_indices is provided:
                bounds: [num_envs, 2, 4] bounding box of the placed object for each environment
                        Invalid environments return zero values
        """
        # Get root state for all actors
        positions = self.actor_root_state_tensor[self.envs_indices, :, :3]  # [num_envs, num_actors, 3]
        orientation_quats = self.actor_root_state_tensor[self.envs_indices, :, 3:7]  # [num_envs, num_actors, 4] (x,y,z,w)
        # Convert quaternion format: IsaacGym (x,y,z,w) -> quaternion_to_matrix (w,x,y,z)
        orientation_quats_wxyz = torch.cat([orientation_quats[..., 3:4], orientation_quats[..., 0:3]], dim=-1)
        rotation_matrices = quaternion_to_matrix(orientation_quats_wxyz)  # [num_envs, num_actors, 3, 3]

        # Apply homogeneous transformation to original mesh coordinates
        homogeneous_transforms = self.create_homogeneous_transforms(rotation_matrices, positions)
        self.mesh_bounds_repeat = self.mesh_bounds.repeat(self.num_envs, 1, 1, 1)
        minC = self.mesh_bounds_repeat[:, :, 0, :].view(self.num_envs, -1, 1, 4)
        maxC = self.mesh_bounds_repeat[:, :, 1, :].view(self.num_envs, -1, 1, 4)

        min_bounds = homogeneous_transforms @ minC.transpose(2, 3)
        max_bounds = homogeneous_transforms @ maxC.transpose(2, 3)
        all_bounds = torch.cat([min_bounds.transpose(2, 3), max_bounds.transpose(2, 3)], dim=2)
        # all_bounds: [num_envs, num_actors, 2, 4]

        # If placed_actor_indices is not specified, return bounding boxes for all actors
        if placed_actor_indices is None:
            return all_bounds

        # Convert placed_actor_indices
        if not isinstance(placed_actor_indices, torch.Tensor):
            placed_actor_indices = torch.tensor(placed_actor_indices, device=self.sim_device, dtype=torch.long)

        # Get valid environment mask
        valid_env_mask = placed_actor_indices >= 5  # [num_envs]

        # Initialize results
        bounds = torch.zeros(self.num_envs, 2, 4, device=self.sim_device)

        # If no valid environments, return directly
        if not valid_env_mask.any():
            return bounds

        # Use advanced indexing to get bounding box for each environment's corresponding object
        env_indices = torch.arange(self.num_envs, device=self.sim_device)

        # Replace invalid indices with 0 (to avoid index out of bounds)
        safe_indices = torch.where(valid_env_mask, placed_actor_indices, torch.zeros_like(placed_actor_indices))

        # Extract bounding box for each environment's corresponding object
        extracted_bounds = all_bounds[env_indices, safe_indices]  # [num_envs, 2, 4]

        # Only keep data for valid environments
        bounds[valid_env_mask] = extracted_bounds[valid_env_mask]

        return bounds

    # Create homogeneous transformation matrices
    def create_homogeneous_transforms(self,rotation_matrices,translations):
        homogeneous_transforms = torch.zeros(self.num_envs, int(len(self.actor_handles)/self.num_envs)+5, 4, 4,
                                        dtype=torch.float32, device=self.sim_device)

        # Set rotation part (top-left 3x3)
        homogeneous_transforms[:, :, :3, :3] = rotation_matrices

        # Set translation part (top-right 3x1)
        homogeneous_transforms[:, :, :3, 3] = translations

        # Set homogeneous coordinate (bottom-right 1x1)
        homogeneous_transforms[:, :, 3, 3] = 1.0

        return homogeneous_transforms

    def adjustHeight(self, env_ids, actor_indices, heights):
        """
        Adjust the Z-axis height position of specified objects in specified environments

        Move the bottom of the object (FLB - Front Left Bottom) to the specified height

        Args:
            env_ids: [N] Environment index tensor
            actor_indices: [N] Actor index tensor for the corresponding environment
            heights: [N] Target height (unscaled value, multiplied by scale factor inside the function)

        Example:
            # Move object 5 in environment 0 to height 0.1, object 6 in environment 1 to height 0.2
            env_ids = torch.tensor([0, 1], device=sim_device)
            actor_indices = torch.tensor([5, 6], device=sim_device)
            heights = torch.tensor([0.1, 0.2], device=sim_device)
            interface.adjustHeight(env_ids, actor_indices, heights)
        """
        if len(env_ids) == 0:
            return

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim_device, dtype=torch.long)
        if not isinstance(actor_indices, torch.Tensor):
            actor_indices = torch.tensor(actor_indices, device=self.sim_device, dtype=torch.long)
        if not isinstance(heights, torch.Tensor):
            heights = torch.tensor(heights, device=self.sim_device, dtype=torch.float32)

        # Apply scale factor to target height
        scaled_heights = heights * self.defaultScale[2]

        # Get current position and rotation of the object
        current_positions = self.actor_root_state_tensor[env_ids, actor_indices, :3]  # [N, 3]
        current_orientations = self.actor_root_state_tensor[env_ids, actor_indices, 3:7]  # [N, 4] (x,y,z,w)

        # Compute rotation matrix of the object
        # Convert quaternion format: IsaacGym (x,y,z,w) -> quaternion_to_matrix (w,x,y,z)
        current_orientations_wxyz = torch.cat([current_orientations[..., 3:4], current_orientations[..., 0:3]], dim=-1)
        rotation_matrices = quaternion_to_matrix(current_orientations_wxyz)  # [N, 3, 3]

        # Get original mesh bounds and centroid (in local coordinate system)
        local_min = self.mesh_bounds[actor_indices, 0, :3]  # [N, 3]
        local_max = self.mesh_bounds[actor_indices, 1, :3]  # [N, 3]
        centroid = self.mesh_centroids[actor_indices]  # [N, 3]

        # Compute offset of bounds relative to centroid (consistent with addObject)
        rel_min = local_min - centroid  # [N, 3]
        rel_max = local_max - centroid  # [N, 3]

        # Compute minimum Z of 8 corner points relative to centroid after rotation
        # For Z coordinate, we only care which corner has minimum Z after rotation
        N = len(env_ids)
        corners_z = torch.zeros(N, 8, device=self.sim_device)
        corners = torch.zeros(N, 8, 3, device=self.sim_device)
        corners[:, 0] = torch.stack([rel_min[:, 0], rel_min[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 1] = torch.stack([rel_min[:, 0], rel_min[:, 1], rel_max[:, 2]], dim=1)
        corners[:, 2] = torch.stack([rel_min[:, 0], rel_max[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 3] = torch.stack([rel_min[:, 0], rel_max[:, 1], rel_max[:, 2]], dim=1)
        corners[:, 4] = torch.stack([rel_max[:, 0], rel_min[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 5] = torch.stack([rel_max[:, 0], rel_min[:, 1], rel_max[:, 2]], dim=1)
        corners[:, 6] = torch.stack([rel_max[:, 0], rel_max[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 7] = torch.stack([rel_max[:, 0], rel_max[:, 1], rel_max[:, 2]], dim=1)

        # Rotate corner points
        corners_T = corners.transpose(1, 2)  # [N, 3, 8]
        rotated_corners_T = torch.bmm(rotation_matrices, corners_T)  # [N, 3, 8]
        rotated_corners = rotated_corners_T.transpose(1, 2)  # [N, 8, 3]

        # Minimum Z relative to centroid after rotation
        rotated_rel_min_z = rotated_corners[:, :, 2].min(dim=1)[0]  # [N]

        # Centroid position after rotation (relative to OBJ origin)
        rotated_centroid = torch.bmm(rotation_matrices, centroid.unsqueeze(-1)).squeeze(-1)  # [N, 3]

        # Current world Z coordinate of the object's bottom
        # Bottom Z = actor position Z + rotated centroid Z + rotated minimum Z relative to centroid
        world_min_z = current_positions[:, 2] + rotated_centroid[:, 2] + rotated_rel_min_z

        # Compute the difference between current bottom Z and target height
        z_offset = scaled_heights - world_min_z

        # Update the Z coordinate of the object
        self.actor_root_state_tensor[env_ids, actor_indices, 2] = current_positions[:, 2] + z_offset

        # Get global indices for updating the physics engine
        global_indices = self.global_indices[env_ids, actor_indices].to(torch.int32)

        # Batch apply state updates to physics engine
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state_tensor.view(-1, 13)),
            gymtorch.unwrap_tensor(global_indices),
            len(global_indices)
        )

    def addObject(self, env_ids, actor_indices, targetFLB, rotations=None, item_type_ids=None):
        """
        Add an object to a specified position in the bin (move from storage area to target position)

        In IsaacGym, all objects are pre-created in the storage area. This function moves the
        specified object to the target position inside the bin and sets its rotation.

        Args:
            env_ids: [N] Environment index tensor
            actor_indices: [N] Actor index tensor for the corresponding environment (used for actors in physics simulation)
            targetFLB: [N, 3] Target position (Front-Left-Bottom, unscaled, multiplied by scale factor inside function)
            rotations: [N, 4] Target rotation (quaternion [x, y, z, w]), uses identity quaternion if None
            item_type_ids: [N] Item type ID (0-based), used to get correct mesh bounds
                          If None, uses actor_indices for indexing (backward compatibility)
        """
        if len(env_ids) == 0:
            return

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim_device, dtype=torch.long)
        if not isinstance(actor_indices, torch.Tensor):
            actor_indices = torch.tensor(actor_indices, device=self.sim_device, dtype=torch.long)
        if not isinstance(targetFLB, torch.Tensor):
            targetFLB = torch.tensor(targetFLB, device=self.sim_device, dtype=torch.float32)

        # Ensure targetFLB is 2-dimensional
        if targetFLB.dim() == 1:
            targetFLB = targetFLB.unsqueeze(0)

        # Handle rotation
        if rotations is None:
            rotations = torch.zeros(len(env_ids), 4, device=self.sim_device)
            rotations[:, 3] = 1.0  # Identity quaternion (0, 0, 0, 1)
        else:
            if not isinstance(rotations, torch.Tensor):
                rotations = torch.tensor(rotations, device=self.sim_device, dtype=torch.float32)
            if rotations.dim() == 1:
                rotations = rotations.unsqueeze(0)

        # Apply scale factor to target position
        scaled_targetFLB = targetFLB * self.defaultScale  # [N, 3]
        scaled_bin = self.bin * self.defaultScale

        # Get mesh bounds of the object (scaled, relative to OBJ origin)
        if item_type_ids is not None:
            if not isinstance(item_type_ids, torch.Tensor):
                item_type_ids = torch.tensor(item_type_ids, device=self.sim_device, dtype=torch.long)
            bounds_indices = item_type_ids + 5
        else:
            bounds_indices = actor_indices

        local_min = self.mesh_bounds[bounds_indices, 0, :3]
        local_max = self.mesh_bounds[bounds_indices, 1, :3]
        centroid = self.mesh_centroids[bounds_indices]

        # Compute offset of bounds relative to centroid
        rel_min = local_min - centroid
        rel_max = local_max - centroid

        # Compute rotation matrix
        rotations_wxyz = torch.cat([rotations[..., 3:4], rotations[..., 0:3]], dim=-1)
        rotation_matrices = quaternion_to_matrix(rotations_wxyz)

        # Compute 8 corner points of mesh relative to centroid
        N = len(env_ids)
        corners = torch.zeros(N, 8, 3, device=self.sim_device, dtype=torch.float32)
        corners[:, 0] = torch.stack([rel_min[:, 0], rel_min[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 1] = torch.stack([rel_min[:, 0], rel_min[:, 1], rel_max[:, 2]], dim=1)
        corners[:, 2] = torch.stack([rel_min[:, 0], rel_max[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 3] = torch.stack([rel_min[:, 0], rel_max[:, 1], rel_max[:, 2]], dim=1)
        corners[:, 4] = torch.stack([rel_max[:, 0], rel_min[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 5] = torch.stack([rel_max[:, 0], rel_min[:, 1], rel_max[:, 2]], dim=1)
        corners[:, 6] = torch.stack([rel_max[:, 0], rel_max[:, 1], rel_min[:, 2]], dim=1)
        corners[:, 7] = torch.stack([rel_max[:, 0], rel_max[:, 1], rel_max[:, 2]], dim=1)

        # Rotate corner points around centroid
        corners_T = corners.transpose(1, 2)
        rotated_corners_T = torch.bmm(rotation_matrices, corners_T)
        rotated_corners = rotated_corners_T.transpose(1, 2)

        # Get dimensions after rotation
        rotated_rel_min, _ = rotated_corners.min(dim=1)
        rotated_rel_max, _ = rotated_corners.max(dim=1)
        rotated_extents = rotated_rel_max - rotated_rel_min  # Object dimensions after rotation

        # Safety bounds check: use dimensions after rotation
        # Maximum FLB value = bin_size - rotated_extents
        max_flb = scaled_bin - rotated_extents
        max_flb = torch.clamp(max_flb, min=0)

        # Use torch.minimum/maximum to implement clamp (avoid mixing Number and Tensor)
        zeros = torch.zeros_like(scaled_targetFLB[:, 0])
        scaled_targetFLB[:, 0] = torch.maximum(zeros, torch.minimum(scaled_targetFLB[:, 0], max_flb[:, 0]))
        scaled_targetFLB[:, 1] = torch.maximum(zeros, torch.minimum(scaled_targetFLB[:, 1], max_flb[:, 1]))
        scaled_targetFLB[:, 2] = torch.maximum(zeros, torch.minimum(scaled_targetFLB[:, 2], max_flb[:, 2]))

        # Add z offset for bin bottom
        z_offset = 0.1
        scaled_targetFLB[:, 2] = scaled_targetFLB[:, 2] + z_offset

        # Compute target position for object centroid
        target_centroid = scaled_targetFLB - rotated_rel_min

        # Offset of centroid relative to OBJ origin after rotation
        # Note: centroid itself also changes position due to rotation (relative to OBJ origin)
        centroid_T = centroid.unsqueeze(-1)  # [N, 3, 1]
        rotated_centroid = torch.bmm(rotation_matrices, centroid_T).squeeze(-1)  # [N, 3]

        # Compute target position of OBJ origin (actor root position)
        # Actor origin = target centroid position - rotated centroid offset
        target_position = target_centroid - rotated_centroid  # [N, 3]

        # Update position and rotation of the object
        self.actor_root_state_tensor[env_ids, actor_indices, 0:3] = target_position
        self.actor_root_state_tensor[env_ids, actor_indices, 3:7] = rotations

        # Set velocity to zero
        self.actor_root_state_tensor[env_ids, actor_indices, 7:13] = 0.0

        # Get global indices for updating physics engine
        global_indices = self.global_indices[env_ids, actor_indices].to(torch.int32)

        # Batch apply state updates to physics engine
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state_tensor.view(-1, 13)),
            gymtorch.unwrap_tensor(global_indices),
            len(global_indices)
        )

    def getAllPositionAndOrientation(self, placed_actor_indices=None, inner=True):
        """
        Get position and orientation information of placed objects

        Args:
            placed_actor_indices: [num_envs, max_placed] Indices of placed objects in each environment, -1 means invalid
                                  If None or no valid objects, return None, None, None
            inner: Whether to return values in internal coordinate system (reserved parameter, compatible with original interface)

        Returns:
            If no placed objects:
                None, None, None
            If there are placed objects:
                positions: [num_envs, max_placed, 3] Positions of placed objects (invalid positions filled with 0)
                orientations: [num_envs, max_placed, 4] Quaternions of placed objects (invalid positions filled with identity quaternion)
                valid_mask: [num_envs, max_placed] Mask of valid objects
        """
        # If placed_actor_indices is not passed in, return empty
        if placed_actor_indices is None:
            return None, None, None

        # Convert to tensor if needed
        if not isinstance(placed_actor_indices, torch.Tensor):
            placed_actor_indices = torch.tensor(placed_actor_indices, device=self.sim_device, dtype=torch.long)

        # Get mask of valid objects (index >= 5 means valid, since 0-4 are bin walls)
        valid_mask = placed_actor_indices >= 5

        # If there are no valid objects, return empty
        if not valid_mask.any():
            return None, None, None

        # Initialize output tensors
        num_envs = placed_actor_indices.shape[0]
        max_placed = placed_actor_indices.shape[1]

        positions = torch.zeros(num_envs, max_placed, 3, device=self.sim_device)
        orientations = torch.zeros(num_envs, max_placed, 4, device=self.sim_device)
        orientations[:, :, 3] = 1.0  # Default identity quaternion (0, 0, 0, 1)

        # Use advanced indexing to get positions and orientations of placed objects
        # Replace invalid indices with 0 (to avoid index out of bounds), then filter with valid_mask
        safe_indices = torch.where(valid_mask, placed_actor_indices, torch.zeros_like(placed_actor_indices))

        # Create environment indices for advanced indexing
        env_indices = torch.arange(num_envs, device=self.sim_device).unsqueeze(1).expand(-1, max_placed)

        # Get positions and orientations
        all_positions = self.actor_root_state_tensor[:, :, :3]  # [num_envs, num_actors, 3]
        all_orientations = self.actor_root_state_tensor[:, :, 3:7]  # [num_envs, num_actors, 4]

        # Use advanced indexing to extract states of placed objects
        extracted_positions = all_positions[env_indices, safe_indices]  # [num_envs, max_placed, 3]
        extracted_orientations = all_orientations[env_indices, safe_indices]  # [num_envs, max_placed, 4]

        # Only keep data for valid objects
        positions[valid_mask] = extracted_positions[valid_mask]
        orientations[valid_mask] = extracted_orientations[valid_mask]

        return positions, orientations, valid_mask

    def disableObjects(self, env_ids, actor_indices, targetZ=None):
        """
        Disable dynamic properties (fix) of specified objects in specified environments

        Consistent with the disableObject function in the original Interface.py:
        - Optionally adjust object height to targetZ
        - Mark the object as disabled (fixed)
        - Store the fixed state (position and rotation) of the object
        - Set velocity to 0

        In IsaacGym, since directly setting mass to 0 is not supported,
        this is achieved by recording the fixed state and continuously resetting it during simulation.

        Args:
            env_ids: [N] Environment indices
            actor_indices: [N] Actor indices in the corresponding environment
            targetZ: [N] Optional target Z height (unscaled), adjusted first if provided

        Example:
            # Disable object 5 in environment 0 and object 6 in environment 1
            env_ids = torch.tensor([0, 1], device=sim_device)
            actor_indices = torch.tensor([5, 6], device=sim_device)
            interface.disableObjects(env_ids, actor_indices)

            # Disable and adjust height
            targetZ = torch.tensor([0.1, 0.15], device=sim_device)
            interface.disableObjects(env_ids, actor_indices, targetZ)
        """
        if len(env_ids) == 0:
            return

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim_device, dtype=torch.long)
        if not isinstance(actor_indices, torch.Tensor):
            actor_indices = torch.tensor(actor_indices, device=self.sim_device, dtype=torch.long)

        # If targetZ is provided, adjust height first
        if targetZ is not None:
            if not isinstance(targetZ, torch.Tensor):
                targetZ = torch.tensor(targetZ, device=self.sim_device, dtype=torch.float32)
            self.adjustHeight(env_ids, actor_indices, targetZ)

        # Set the object's velocity to 0 (linear and angular velocity)
        self.actor_root_state_tensor[env_ids, actor_indices, 7:13] = 0

        # Store the fixed state (position and rotation) of the object
        self.disabled_states[env_ids, actor_indices, :3] = self.actor_root_state_tensor[env_ids, actor_indices, :3]
        self.disabled_states[env_ids, actor_indices, 3:7] = self.actor_root_state_tensor[env_ids, actor_indices, 3:7]

        # Mark the object as disabled
        self.disabled_actors[env_ids, actor_indices] = True

        # Get global indices
        global_indices = self.global_indices[env_ids, actor_indices].to(torch.int32)

        # Apply state update to physics engine
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state_tensor.view(-1, 13)),
            gymtorch.unwrap_tensor(global_indices),
            len(global_indices)
        )

    def enforceDisabledObjects(self):
        """
        Enforce the fixed state of all disabled objects

        Call this function after each simulation step to reset the position and velocity
        of disabled objects to their fixed values.
        This is the method to simulate the PyBullet mass=0 effect in IsaacGym.
        """
        # Find all disabled objects
        disabled_mask = self.disabled_actors  # [num_envs, num_actors]

        if not disabled_mask.any():
            return

        # Get all (env_idx, actor_idx) pairs for disabled objects
        env_indices_matrix = torch.arange(self.num_envs, device=self.sim_device).unsqueeze(1).expand_as(disabled_mask)
        actor_indices_matrix = torch.arange(disabled_mask.shape[1], device=self.sim_device).unsqueeze(0).expand_as(disabled_mask)

        disabled_env_indices = env_indices_matrix[disabled_mask]
        disabled_actor_indices = actor_indices_matrix[disabled_mask]

        if len(disabled_env_indices) == 0:
            return

        # Reset position and rotation to fixed state
        self.actor_root_state_tensor[disabled_env_indices, disabled_actor_indices, :3] = \
            self.disabled_states[disabled_env_indices, disabled_actor_indices, :3]
        self.actor_root_state_tensor[disabled_env_indices, disabled_actor_indices, 3:7] = \
            self.disabled_states[disabled_env_indices, disabled_actor_indices, 3:7]

        # Set velocity to 0
        self.actor_root_state_tensor[disabled_env_indices, disabled_actor_indices, 7:13] = 0

        # Get global indices
        global_indices = self.global_indices[disabled_env_indices, disabled_actor_indices].to(torch.int32)

        # Apply state update
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state_tensor.view(-1, 13)),
            gymtorch.unwrap_tensor(global_indices),
            len(global_indices)
        )

    def enableObjects(self, env_ids=None):
        """
        Restore dynamic properties of all disabled objects in specified environments

        Consistent with the enableObjects function in the original Interface.py.

        Args:
            env_ids: [N] Environment indices to restore, if None then restore all environments
        """
        if env_ids is None:
            # Restore all environments
            self.disabled_actors[:] = False
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.sim_device, dtype=torch.long)
            # Only restore specified environments
            self.disabled_actors[env_ids] = False

    def clearDisabledObjects(self, env_ids):
        """
        Clear the disabled object records for specified environments (usually called on environment reset)

        Args:
            env_ids: [N] Environment indices to clear
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim_device, dtype=torch.long)

        self.disabled_actors[env_ids] = False
        self.disabled_states[env_ids] = 0

    def removeBody(self, env_ids, actor_indices):
        """
        "Remove" specified objects from specified environments (simulates removal by resetting objects to initial position)

        Note: IsaacGym does not support truly deleting actors during simulation,
        so deletion is simulated by resetting objects to their initial position (storage area).

        Args:
            env_ids: [N] Environment index tensor, specifies which environments to operate on
            actor_indices: [N] Actor index tensor in the corresponding environment, specifies which objects to delete
                          The object index for env_ids[i] is actor_indices[i]

        Example:
            # Delete object 5 in environment 0 and object 6 in environment 1
            env_ids = torch.tensor([0, 1], device=sim_device)
            actor_indices = torch.tensor([5, 6], device=sim_device)
            interface.removeBody(env_ids, actor_indices)

            # Delete object 5 in all environments (broadcast)
            env_ids = torch.arange(num_envs, device=sim_device)
            actor_indices = torch.full((num_envs,), 5, device=sim_device)
            interface.removeBody(env_ids, actor_indices)
        """
        if len(env_ids) == 0:
            return

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.sim_device, dtype=torch.long)
        if not isinstance(actor_indices, torch.Tensor):
            actor_indices = torch.tensor(actor_indices, device=self.sim_device, dtype=torch.long)

        # Ensure env_ids and actor_indices have the same shape
        assert env_ids.shape == actor_indices.shape, \
            f"env_ids and actor_indices must have the same shape: {env_ids.shape} vs {actor_indices.shape}"

        # Reset objects to initial position (storage area)
        self.actor_root_state_tensor[env_ids, actor_indices, :] = \
            self.init_actor_root_state_tensor[env_ids, actor_indices, :]

        # Get global indices for updating physics engine
        global_indices = self.global_indices[env_ids, actor_indices].to(torch.int32)

        # Batch apply state updates to physics engine
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state_tensor.view(-1, 13)),
            gymtorch.unwrap_tensor(global_indices),
            len(global_indices)
        )



if __name__ == '__main__':
    import time

    # Create interface (use fewer environments for easier observation)
    num_envs = 16
    interface = IsaacGymInterface(
        sim_device=0,
        graphics_device_id=0,
        headless=False,
        bin=[0.35, 0.35, 0.35],
        scale=[5.0, 5.0, 5.0],
        foldername='/home/zzx/Documents/IR-BPP/dataset/blockout/shape_vhacd',
        num_envs=num_envs
    )

    # ==================== Test Space + IRcreator functions ====================
    print("\n" + "=" * 50)
    print("[Space + IRcreator Integration Test] Random object placement test...")
    print("=" * 50)

    # Import SpaceIsaacGym and IRcreator
    from space_isaacgym import SpaceIsaacGym
    from IRcreator_isaacgym import RandomItemCreatorIsaacGym
    from arguments import get_args

    args = get_args()

    # Create Space instance
    space = SpaceIsaacGym(
        bin_dimension=[0.35, 0.35, 0.35],
        resolutionAct=0.05,  # Action resolution (5cm)
        resolutionH=0.01,    # Height map resolution (1cm)
        boxPack=False,
        ZRotNum=8,
        shotInfo=None,
        scale=[5.0, 5.0, 5.0],
        device=interface.sim_device,
        num_envs=num_envs,
        args=args
    )

    print(f"  Space created successfully:")
    print(f"    Height map size: [{space.rangeX_C}, {space.rangeY_C}]")
    print(f"    Action space size: [{space.rangeX_A}, {space.rangeY_A}]")
    print(f"    Number of objects: {space.boundingSize.shape[0]}")
    print(f"    Number of rotations: {space.rotNum}")

    # Create ItemCreator instance
    obj_nums = space.boundingSize.shape[0]  # Number of item types
    item_set = list(range(obj_nums))  # Available item IDs: 0 to obj_nums-1

    item_creator = RandomItemCreatorIsaacGym(
        item_set=item_set,
        num_envs=num_envs,
        device=interface.sim_device,
        max_queue_length=100
    )

    print(f"  ItemCreator created successfully:")
    print(f"    Number of items: {item_creator.num_items}")
    print(f"    Maximum queue length: {item_creator.max_queue_length}")

    # Reset all environments
    interface.reset(torch.arange(num_envs, device=interface.sim_device))
    space.reset()
    item_creator.reset()

    # Record placed objects in each environment
    placed_count = torch.zeros(num_envs, device=interface.sim_device, dtype=torch.long)
    reset_count = torch.zeros(num_envs, device=interface.sim_device, dtype=torch.long)  # Reset count
    used_items = torch.zeros(num_envs, obj_nums, device=interface.sim_device, dtype=torch.bool)  # Used item types

    print("\n[Integration Test] Continuously randomly placing objects (auto-reset when no candidate positions or no available items)...")
    print(f"  Number of item types: {obj_nums}")
    print("Press Ctrl+C to exit\n")

    placement_round = 0
    total_placements = 0

    try:
        while True:  # Infinite loop
            placement_round += 1

            # 1. Use ItemCreator to preview the next item
            # Ensure each environment's queue has enough items
            next_item_preview = item_creator.preview(length=1)  # [num_envs, 1]
            next_item_ID = next_item_preview[:, 0]  # [num_envs]

            # Check if any environment's item has already been used (actor is already in the bin)
            # If used_items[env, next_item_ID[env]] == True, need to regenerate
            env_indices = torch.arange(num_envs, device=interface.sim_device)
            already_used = used_items[env_indices, next_item_ID]

            # For environments with used items, pop the queue head and generate new ones
            retry_count = 0
            while already_used.any() and retry_count < obj_nums:
                # Pop the queue head for environments with used items
                used_envs = env_indices[already_used]
                item_creator.pop_first(env_indices=used_envs)

                # Preview again
                next_item_preview = item_creator.preview(length=1)
                next_item_ID = next_item_preview[:, 0]
                already_used = used_items[env_indices, next_item_ID]
                retry_count += 1

            # Check which environments have no available items (all items have been used)
            available_count = (~used_items).sum(dim=1)
            no_available_items = available_count == 0

            # 2. Call get_possible_position to get feasible positions
            naiveMask = space.get_possible_position(next_item_ID)

            # Count the number of feasible positions for each environment
            valid_positions_count = naiveMask.sum(dim=(1, 2, 3))  # [num_envs]

            # 3. Check which environments have no feasible positions or no available items, need reset
            need_reset = (valid_positions_count == 0) | no_available_items

            if need_reset.any():
                reset_env_ids = torch.arange(num_envs, device=interface.sim_device)[need_reset]
                reset_reasons = []
                for i, env_id in enumerate(reset_env_ids.tolist()):
                    if no_available_items[env_id]:
                        reset_reasons.append(f"env{env_id}:no available items")
                    else:
                        reset_reasons.append(f"env{env_id}:no candidate positions")

                print(f"\n--- Round {placement_round} reset: {', '.join(reset_reasons)} ---")
                print(f"  Placement count before reset: {placed_count[need_reset].tolist()}")
                print(f"  Queue length before reset: {item_creator.queue_lengths[need_reset].tolist()}")

                # Reset these environments
                interface.reset(reset_env_ids)
                space.reset(reset_env_ids)
                item_creator.reset(env_indices=reset_env_ids)  # Reset item queue

                # Important: run a few simulation steps to ensure objects move to initial position
                for _ in range(10):
                    interface.step_simulation()
                    interface.render()

                # Update counts and used item flags
                reset_count[need_reset] += 1
                placed_count[need_reset] = 0
                used_items[need_reset] = False  # Clear used flags

                print(f"  Cumulative reset count: {reset_count.tolist()}")

                # Re-fetch items and feasible positions
                next_item_preview = item_creator.preview(length=1)
                next_item_ID = next_item_preview[:, 0]

                naiveMask = space.get_possible_position(next_item_ID)
                valid_positions_count = naiveMask.sum(dim=(1, 2, 3))

            # 4. Determine which environments can place items
            can_place = valid_positions_count > 0

            if not can_place.any():
                print(f"  Round {placement_round}: no environments can place items")
                continue

            # 5. Select random position for all placeable environments (vectorized)
            rotNum = space.rotNum
            rangeX_A = space.rangeX_A
            rangeY_A = space.rangeY_A

            naiveMask_flat = naiveMask.view(num_envs, -1)
            posZvalid_flat = space.posZvalid.view(num_envs, -1)

            # Randomly select valid positions
            random_scores = torch.rand_like(naiveMask_flat, dtype=torch.float32)
            random_scores[naiveMask_flat == 0] = -1e9
            selected_flat_idx = random_scores.argmax(dim=1)

            # Convert flat index to (rot_idx, action_x, action_y)
            rot_ids = selected_flat_idx // (rangeX_A * rangeY_A)
            remainder = selected_flat_idx % (rangeX_A * rangeY_A)
            action_x = remainder // rangeY_A
            action_y = remainder % rangeY_A

            # Get placement height
            posZ = posZvalid_flat[torch.arange(num_envs, device=interface.sim_device), selected_flat_idx]

            # Compute FLB coordinates
            targetFLB_x = action_x.float() * space.resolutionAct
            targetFLB_y = action_y.float() * space.resolutionAct
            # Add safety margin to prevent penetration (1cm, unscaled coordinates)
            # Larger margin compensates for: object simulation movement, discretization error, irregular shapes
            safety_margin = 0.01
            # Ensure bottom does not penetrate (posZ minimum is 0)
            targetFLB_z = torch.clamp(posZ, min=0.0) + safety_margin
            targetFLB = torch.stack([targetFLB_x, targetFLB_y, targetFLB_z], dim=1)

            # Actor index: 5 + item type ID
            actor_indices = 5 + next_item_ID

            # 6. Only operate on placeable environments
            env_ids_valid = torch.arange(num_envs, device=interface.sim_device)[can_place]
            actor_indices_valid = actor_indices[can_place]
            targetFLB_valid = targetFLB[can_place]
            rot_ids_valid = rot_ids[can_place]
            next_item_ID_valid = next_item_ID[can_place]

            # 7. Compute rotation quaternion
            # Note: getRotationMatrix returns rotations in the order [0, 90, 180, 270, 45, 135, 225, 315] degrees
            # Not uniformly distributed [0, 45, 90, ...]
            rot_angle_map = torch.tensor([0, 90, 180, 270, 45, 135, 225, 315],
                                          device=interface.sim_device, dtype=torch.float32)
            angles_degrees = rot_angle_map[rot_ids_valid]  # Get actual angles (degrees)
            angles = angles_degrees * (3.14159265 / 180.0)  # Convert to radians
            half_angles = angles / 2
            rotations_valid = torch.zeros((len(env_ids_valid), 4), device=interface.sim_device, dtype=torch.float32)
            rotations_valid[:, 2] = torch.sin(half_angles)
            rotations_valid[:, 3] = torch.cos(half_angles)

            # 8. Batch place objects
            interface.addObject(env_ids_valid, actor_indices_valid, targetFLB_valid, rotations_valid)

            # 9. Simulate (let objects fall and settle)
            for _ in range(100):
                interface.step_simulation()
                interface.render()

            # 10. Update height map (based on actual simulation position, with safety margin)
            # Note: refresh_tensor() is already included in step_simulation(), no need to refresh again
            actual_positions = interface.actor_root_state_tensor[env_ids_valid, actor_indices_valid, 0:3]
            actual_quaternions = interface.actor_root_state_tensor[env_ids_valid, actor_indices_valid, 3:7]
            mesh_bounds_batch = interface.mesh_bounds[actor_indices_valid, :, :3]

            space.place_item_with_actual_aabb(
                actual_positions=actual_positions,
                actual_quaternions=actual_quaternions,
                mesh_bounds=mesh_bounds_batch,
                env_ids=env_ids_valid,
                scale=interface.defaultScale,
                height_margin=0.02,  # 2cm Z direction safety margin (compensates object movement and shape error)
                xy_margin=0.05       # 5cm X/Y direction safety margin
            )

            # 11. Update placement count and used item flags
            placed_count[can_place] += 1
            total_placements += can_place.sum().item()

            # Mark used item types (vectorized)
            used_items[env_ids_valid, next_item_ID_valid] = True

            # 12. Pop placed items from ItemCreator queue
            item_creator.pop_first(env_indices=env_ids_valid)

            # Periodically print status (every round for debugging)
            print(f"--- Round {placement_round} ---")
            print(f"  Placement environments: {env_ids_valid.tolist()}, items: {next_item_ID_valid.tolist()}")
            print(f"  Target FLB (unscaled): {targetFLB_valid[:, :].tolist()}")
            print(f"  Actual position (scaled): {actual_positions[:, :].tolist()}")
            print(f"  Height map maximum: {[f'{v:.3f}' for v in space.heightmapC.max(dim=2)[0].max(dim=1)[0].tolist()]}")
            print(f"  Height map updated region top: {(actual_positions[:, 2] - 0.1).tolist()}")  # Subtract z_offset

    except KeyboardInterrupt:
        print(f"\n\n[User interrupted]")
        print(f"  Total rounds: {placement_round}")
        print(f"  Total placements: {total_placements}")
        print(f"  Placements per environment: {placed_count.tolist()}")
        print(f"  Reset count per environment: {reset_count.tolist()}")
        print(f"  Final queue lengths: {item_creator.queue_lengths.tolist()}")

    interface.close()
    print("Program finished")
