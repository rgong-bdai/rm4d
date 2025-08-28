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