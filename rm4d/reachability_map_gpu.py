import numpy as np
import torch
from .reachability_map import ReachabilityMap4D


class ReachabilityMap4DGPU(ReachabilityMap4D):
    """
    GPU-accelerated version of ReachabilityMap4D.
    Provides batch operations for reachability checking.
    """
    
    def __init__(self, *args, use_gpu=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu and self.map is not None:
            # Move map to GPU
            self.gpu_map = torch.from_numpy(self.map).cuda()
            print(f"Reachability map moved to GPU: {self.gpu_map.device}")
        else:
            self.gpu_map = None
            if use_gpu and not torch.cuda.is_available():
                print("Warning: GPU requested but CUDA not available. Using CPU.")
    
    @classmethod
    def from_file(cls, filename, use_gpu=True):
        """Load from file with GPU support."""
        d = np.load(filename, allow_pickle=True).item()
        rm = cls(xy_limits=d['xy_limits'],
                 z_limits=d['z_limits'],
                 voxel_res=d['voxel_res'],
                 n_bins_theta=d['n_bins_theta'],
                 no_map=True,
                 use_gpu=use_gpu)
        rm.map = d['map']
        
        # Initialize GPU map if needed
        if rm.use_gpu:
            rm.gpu_map = torch.from_numpy(rm.map).cuda()
            print(f"Reachability map moved to GPU: {rm.gpu_map.device}")
        
        print(f'{cls.__name__} loaded from {filename}')
        rm.print_structure()
        return rm
    
    def are_poses_reachable_gpu(self, poses):
        """
        GPU-accelerated batch reachability checking.
        
        Args:
            poses: (N, 4, 4) tensor of end-effector poses
            
        Returns:
            (N,) boolean tensor indicating which poses are reachable
        """
        if not self.use_gpu or self.gpu_map is None:
            # Fall back to CPU implementation
            return self.are_poses_reachable_cpu(poses)
        
        # Convert to GPU tensor if needed
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).cuda().float()
        elif poses.device.type != 'cuda':
            poses = poses.cuda().float()
        
        # Get indices for all poses
        indices = self._get_indices_batch_gpu(poses)
        
        # Check reachability using GPU map
        reachable = self.gpu_map[indices[:, 0], indices[:, 1], 
                                indices[:, 2], indices[:, 3]]
        
        return reachable
    
    def _get_indices_batch_gpu(self, poses):
        """
        Get map indices for batch of poses on GPU.
        
        Args:
            poses: (N, 4, 4) tensor of poses on GPU
            
        Returns:
            (N, 4) tensor of indices [z_idx, th_idx, x_idx, y_idx]
        """
        # Extract positions and orientations
        positions = poses[:, :3, 3]  # (N, 3)
        rotations = poses[:, :3, :3]  # (N, 3, 3)
        
        # Calculate p_z, theta, x_star, y_star for all poses
        p_z = positions[:, 2]  # (N,)
        
        # Calculate theta from rotation matrix
        rz_z = rotations[:, 2, 2]  # (N,)
        theta = torch.acos(torch.clamp(rz_z, -1.0, 1.0))  # (N,)
        
        # Calculate canonical base positions
        p_x, p_y = positions[:, 0], positions[:, 1]  # (N,), (N,)
        
        # Get rotation matrices for 2D rotation
        rz_x, rz_y = rotations[:, 0, 2], rotations[:, 1, 2]  # (N,), (N,)
        psi = torch.atan2(rz_y, rz_x)  # (N,)
        
        # Build 2D rotation matrices
        cos_psi = torch.cos(psi)  # (N,)
        sin_psi = torch.sin(psi)  # (N,)
        
        # Apply rotation: [cos_psi, sin_psi; -sin_psi, cos_psi] @ [-p_x, -p_y]
        x_star = cos_psi * (-p_x) + sin_psi * (-p_y)  # (N,)
        y_star = -sin_psi * (-p_x) + cos_psi * (-p_y)  # (N,)
        
        # Convert to indices
        z_idx = ((p_z - self.z_limits[0]) / self.voxel_res).long()
        theta_idx = ((theta - self.theta_limits[0]) / self.theta_res).long()
        x_idx = ((x_star - self.xy_limits[0]) / self.voxel_res).long()
        y_idx = ((y_star - self.xy_limits[0]) / self.voxel_res).long()
        
        # Clamp indices to valid range
        z_idx = torch.clamp(z_idx, 0, self.n_bins_z - 1)
        theta_idx = torch.clamp(theta_idx, 0, self.n_bins_theta - 1)
        x_idx = torch.clamp(x_idx, 0, self.n_bins_xy - 1)
        y_idx = torch.clamp(y_idx, 0, self.n_bins_xy - 1)
        
        return torch.stack([z_idx, theta_idx, x_idx, y_idx], dim=1)
    
    def are_poses_reachable_cpu(self, poses):
        """
        CPU fallback for batch reachability checking.
        
        Args:
            poses: (N, 4, 4) array of end-effector poses
            
        Returns:
            (N,) boolean array indicating which poses are reachable
        """
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        
        results = np.zeros(len(poses), dtype=bool)
        for i, pose in enumerate(poses):
            try:
                indices = self.get_indices_for_ee_pose(pose)
                results[i] = self.is_reachable(indices)
            except (IndexError, ValueError):
                results[i] = False
        
        return results
    
    def are_poses_reachable(self, poses):
        """
        Batch reachability checking with automatic GPU/CPU selection.
        
        Args:
            poses: (N, 4, 4) array/tensor of end-effector poses
            
        Returns:
            (N,) boolean array/tensor indicating which poses are reachable
        """
        if self.use_gpu and self.gpu_map is not None:
            return self.are_poses_reachable_gpu(poses)
        else:
            return self.are_poses_reachable_cpu(poses)
    
    def are_positions_reachable_with_orientation_threshold(
            self, positions, threshold=0.5):
        """
        GPU-accelerated version of checking if multiple positions are reachable 
        by checking if more than the specified threshold of orientation bins 
        can reach each position.

        Args:
            positions: array-like, shape (N, 3) where each row is [x, y, z]
            threshold: float, minimum fraction of orientation bins that
                      must be reachable (default 0.5 for 50%)
        Returns:
            ndarray of bool, shape (N,) indicating reachability for each position
        """
        
        # For most use cases, if threshold <= 0.1, we can use fast position-only check
        if threshold <= 0.1:
            print(f"Using FAST position-only check for {len(positions)} positions")
            return self._fast_position_reachability_check(positions)
       
        if not self.use_gpu or self.gpu_map is None:
            # Fall back to CPU implementation from parent class
            return super().are_positions_reachable_with_orientation_threshold(
                positions, threshold)
        
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        n_positions = positions.shape[0]
        
        # Convert to GPU tensors
        positions_gpu = torch.from_numpy(positions).cuda().float()
        
        # Pre-compute all theta and psi values
        n_psi_samples = 36
        theta_values = (self.theta_limits[0] +
                       (np.arange(self.n_bins_theta) + 0.5) *
                       self.theta_res)
        psi_values = 2 * np.pi * np.arange(n_psi_samples) / n_psi_samples
        
        # Convert to GPU tensors
        theta_values_gpu = torch.from_numpy(theta_values).cuda().float()
        psi_values_gpu = torch.from_numpy(psi_values).cuda().float()
        
        # Create meshgrids for vectorized computation
        theta_grid, psi_grid = torch.meshgrid(theta_values_gpu, psi_values_gpu, indexing='ij')
        theta_flat = theta_grid.flatten()
        psi_flat = psi_grid.flatten()
        theta_idx_flat = torch.repeat_interleave(torch.arange(self.n_bins_theta, device='cuda'), n_psi_samples)
        
        # Pre-compute trigonometric values
        sin_theta = torch.sin(theta_flat)
        cos_psi = torch.cos(psi_flat)
        sin_psi = torch.sin(psi_flat)
        
        # Compute z indices for all positions
        z_indices = ((positions_gpu[:, 2] - self.z_limits[0]) / self.voxel_res)
        valid_z_mask = ((z_indices >= 0) & (z_indices < self.n_bins_z))
        z_indices = torch.floor(z_indices).long()
        
        # Initialize results
        results = torch.zeros(n_positions, dtype=torch.bool, device='cuda')
        
        # Only process positions with valid z indices
        if not torch.any(valid_z_mask):
            return results.cpu().numpy()
            
        # Filter to valid positions
        valid_positions = positions_gpu[valid_z_mask]
        valid_z_indices = z_indices[valid_z_mask]
        n_valid_positions = len(valid_positions)
        
        # Vectorized computation for all position-orientation combinations
        x_pos = valid_positions[:, 0:1]  # Shape: (n_valid_positions, 1)
        y_pos = valid_positions[:, 1:2]  # Shape: (n_valid_positions, 1)
        
        # Broadcast to all orientations
        # rz components from rotation matrix
        rz_x = sin_theta * cos_psi  # Shape: (n_orientations,)
        rz_y = sin_theta * sin_psi  # Shape: (n_orientations,)
        psi_canonical = torch.atan2(rz_y, rz_x)  # Shape: (n_orientations,)
        
        # Canonical rotation matrices (vectorized)
        cos_psi_can = torch.cos(psi_canonical)  # Shape: (n_orientations,)
        sin_psi_can = torch.sin(psi_canonical)  # Shape: (n_orientations,)
        
        # Apply canonical transformation for all positions
        # Broadcasting: (n_valid_positions, 1) with (n_orientations,)
        x_star = cos_psi_can * (-x_pos) + sin_psi_can * (-y_pos)
        y_star = -sin_psi_can * (-x_pos) + cos_psi_can * (-y_pos)
        # Result shape: (n_valid_positions, n_orientations)
        
        # Convert to indices
        x_indices = (x_star - self.xy_limits[0]) / self.voxel_res
        y_indices = (y_star - self.xy_limits[0]) / self.voxel_res
        
        # Validity mask
        valid_mask = ((x_indices >= 0) & (x_indices < self.n_bins_xy) &
                      (y_indices >= 0) & (y_indices < self.n_bins_xy))
        
        x_indices = torch.floor(x_indices).long()
        y_indices = torch.floor(y_indices).long()
        
        # OPTIMIZED: Fully vectorized reachability checking without sequential loop
        valid_results = torch.zeros(n_valid_positions, dtype=torch.bool, device='cuda')
        
        # Create arrays for all valid combinations
        pos_indices, orient_indices = torch.where(valid_mask)
        if len(pos_indices) > 0:
            # Get corresponding map indices
            map_z_indices = valid_z_indices[pos_indices]
            map_theta_indices = theta_idx_flat[orient_indices]
            map_x_indices = x_indices[pos_indices, orient_indices]
            map_y_indices = y_indices[pos_indices, orient_indices]
            
            # Batch lookup in reachability map
            reachable_values = self.gpu_map[map_z_indices, map_theta_indices,
                                          map_x_indices, map_y_indices]
            
            # OPTIMIZED: Vectorized counting using scatter_add
            reachable_counts = torch.zeros(n_valid_positions, device='cuda', dtype=torch.float32)
            total_counts = torch.zeros(n_valid_positions, device='cuda', dtype=torch.float32)
            
            # Sum reachable orientations per position
            reachable_counts.scatter_add_(0, pos_indices, reachable_values.float())
            total_counts.scatter_add_(0, pos_indices, torch.ones_like(reachable_values.float()))
            
            # Compute reachability fractions
            valid_total_mask = total_counts > 0
            reachable_fractions = torch.zeros_like(reachable_counts)
            reachable_fractions[valid_total_mask] = (reachable_counts[valid_total_mask] / 
                                                   total_counts[valid_total_mask])
            
            # Check threshold
            valid_results = reachable_fractions >= threshold
        
        # Map back to original position indices
        results[valid_z_mask] = valid_results
        
        return results.cpu().numpy()
    
    def _fast_position_reachability_check(self, positions):
        """
        Ultra-fast position reachability check that doesn't compute orientations.
        Checks if ANY orientation at each position is reachable.
        """
        if not self.use_gpu or self.gpu_map is None:
            return self._fast_position_reachability_check_cpu(positions)
            
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        n_positions = positions.shape[0]
        positions_gpu = torch.from_numpy(positions).cuda().float()
        
        # Convert positions to voxel indices
        x_indices = ((positions_gpu[:, 0] - self.xy_limits[0]) / self.voxel_res).long()
        y_indices = ((positions_gpu[:, 1] - self.xy_limits[0]) / self.voxel_res).long()
        z_indices = ((positions_gpu[:, 2] - self.z_limits[0]) / self.voxel_res).long()
        
        # Check bounds
        valid_mask = ((x_indices >= 0) & (x_indices < self.n_bins_xy) &
                     (y_indices >= 0) & (y_indices < self.n_bins_xy) &
                     (z_indices >= 0) & (z_indices < self.n_bins_z))
        
        results = torch.zeros(n_positions, dtype=torch.bool, device='cuda')
        
        if torch.any(valid_mask):
            valid_x = x_indices[valid_mask]
            valid_y = y_indices[valid_mask]
            valid_z = z_indices[valid_mask]
            
            # Check if ANY orientation at this position is reachable
            # Sum across theta dimension (dimension 1) to see if any orientation works
            position_reachability = torch.any(self.gpu_map[valid_z, :, valid_x, valid_y], dim=1)
            results[valid_mask] = position_reachability
        
        return results.cpu().numpy()
    
    def _fast_position_reachability_check_cpu(self, positions):
        """CPU fallback for fast position check"""
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        n_positions = positions.shape[0]
        results = np.zeros(n_positions, dtype=bool)
        
        for i, pos in enumerate(positions):
            try:
                # Convert to voxel indices
                x_idx = int((pos[0] - self.xy_limits[0]) / self.voxel_res)
                y_idx = int((pos[1] - self.xy_limits[0]) / self.voxel_res)
                z_idx = int((pos[2] - self.z_limits[0]) / self.voxel_res)
                
                # Check bounds
                if (0 <= x_idx < self.n_bins_xy and 
                    0 <= y_idx < self.n_bins_xy and 
                    0 <= z_idx < self.n_bins_z):
                    # Check if ANY orientation is reachable
                    results[i] = np.any(self.map[z_idx, :, x_idx, y_idx])
            except (IndexError, ValueError):
                results[i] = False
        
        return results
    
    def get_reachability_scores_gpu(self, poses):
        """
        GPU-accelerated batch reachability scoring.
        
        Args:
            poses: (N, 4, 4) tensor of end-effector poses
            
        Returns:
            (N,) tensor of reachability scores (0.0 or 1.0)
        """
        if not self.use_gpu or self.gpu_map is None:
            # Fall back to CPU implementation
            return self.get_reachability_scores_cpu(poses)
        
        # Convert to GPU tensor if needed
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).cuda().float()
        elif poses.device.type != 'cuda':
            poses = poses.cuda().float()
        
        # Get indices for all poses
        indices = self._get_indices_batch_gpu(poses)
        
        # Get reachability scores using GPU map
        scores = self.gpu_map[indices[:, 0], indices[:, 1], 
                             indices[:, 2], indices[:, 3]].float()
        
        return scores
    
    def get_reachability_scores_cpu(self, poses):
        """
        CPU fallback for batch reachability scoring.
        
        Args:
            poses: (N, 4, 4) array of end-effector poses
            
        Returns:
            (N,) array of reachability scores (0.0 or 1.0)
        """
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        
        scores = np.zeros(len(poses), dtype=float)
        for i, pose in enumerate(poses):
            try:
                indices = self.get_indices_for_ee_pose(pose)
                scores[i] = float(self.is_reachable(indices))
            except (IndexError, ValueError):
                scores[i] = 0.0
        
        return scores
    
    def get_reachability_scores(self, poses):
        """
        Batch reachability scoring with automatic GPU/CPU selection.
        
        Args:
            poses: (N, 4, 4) array/tensor of end-effector poses
            
        Returns:
            (N,) array/tensor of reachability scores (0.0 or 1.0)
        """
        if self.use_gpu and self.gpu_map is not None:
            return self.get_reachability_scores_gpu(poses)
        else:
            return self.get_reachability_scores_cpu(poses) 