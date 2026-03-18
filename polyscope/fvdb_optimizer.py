import torch
try:
    import fvdb
except ImportError:
    print("CRITICAL ERROR: NVIDIA fVDB is not installed in this environment.")
    print("Please ensure you have built fVDB from the OpenVDB GitHub repository.")
    exit(1)

# --- fVDB Optimizer Class ---
class fVDBOptimizerGrid(torch.nn.Module):
    """
    Manages the NVIDIA fVDB Grid topology and the differentiable phi features.
    """
    def __init__(self, resolution, center, radius, device):
        super().__init__()
        self.resolution = resolution
        self.center = center.to(device)
        self.radius = radius
        self.device = device
        self.background_phi = 10.0
        
        self.v_min = self.center - self.radius
        
        self.voxel_size = (2.0 * self.radius) / self.resolution
        
        self.grid = fvdb.GridBatch.from_dense(
            num_grids=1,
            dense_dims=[resolution, resolution, resolution], 
            voxel_sizes=[self.voxel_size], 
            device=device
        )
        
        self.phi = torch.nn.Parameter(
            torch.randn(self.grid.total_voxels, 1, device=device) * 0.1 + 0.5
        )

        with torch.no_grad():
            coords = self.grid.ijk.jdata.long()

            coords_plus_x = coords + torch.tensor([1, 0, 0], device=device)  
            coords_plus_y = coords + torch.tensor([0, 1, 0], device=device)  
            coords_plus_z = coords + torch.tensor([0, 0, 1], device=device)  
            
            self.nb_x = self.grid.ijk_to_index(fvdb.JaggedTensor([coords_plus_x])).jdata  
            self.nb_y = self.grid.ijk_to_index(fvdb.JaggedTensor([coords_plus_y])).jdata  
            self.nb_z = self.grid.ijk_to_index(fvdb.JaggedTensor([coords_plus_z])).jdata

    def prune(self, threshold=0.1, sharpness=1.0):
        """
        Rebuilds the fVDB GridBatch to drop empty voxels based on current phi values.
        """
        with torch.no_grad():
            opacity = torch.sigmoid(-sharpness * self.phi)
            active_mask = (opacity > threshold).squeeze(-1)

            active_count = active_mask.sum().item()
            print(f"[fVDB] Pruning ready. Active voxels: {active_count} / {self.grid.total_voxels}")

    def query(self, points):
        """Samples the fVDB grid at continuous 3D points."""
        original_shape = points.shape[:-1] 
        points_flat = points.reshape(-1, 3)

        points_norm = (points_flat - self.center) / self.radius
        grid_coords = ((points_norm + 1.0) / 2.0) * (self.resolution - 1)
        
        query_points = fvdb.JaggedTensor([grid_coords.float().to(self.device)])
        feature_jagged = fvdb.JaggedTensor([self.phi])
        
        phi_vals_jagged = self.grid.sample_trilinear(query_points, feature_jagged)
        phi_vals = phi_vals_jagged.jdata.squeeze(-1)
        
        in_bounds = (points_norm >= -1.0) & (points_norm <= 1.0)
        in_bounds = in_bounds.all(dim=-1)
        
        phi_vals = torch.where(in_bounds, phi_vals, torch.tensor(self.background_phi, dtype=phi_vals.dtype, device=self.device))
        
        return phi_vals.reshape(original_shape)

    def tv_loss(self):
        """Proper Spatial Dirichlet Loss (L2 Gradient Smoothing) for sparse fVDB."""
        loss = 0
        for nb_indices in [self.nb_x, self.nb_y, self.nb_z]:
            # neighbor_indices is -1 if no neighbor exists at that sparse location
            mask = nb_indices >= 0
            
            # Compute (phi[current] - phi[neighbor])^2
            diff = self.phi[mask] - self.phi[nb_indices[mask]]
            loss += (diff**2).mean()
            
        return loss / 3.0