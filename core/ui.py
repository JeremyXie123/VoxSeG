"""
Polyscope UI components for interactive box setup.
"""

import math
import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim

from core.camera import get_batch_viewmats, get_batch_Ks, compute_orbit_radius, compute_widget_focal_length


class PromptBoxUI:
    """Interactive Polyscope UI for configuring the segmentation prompt box."""
    
    BOX_NAME = "Prompt Box"
    
    def __init__(self, args, default_center: np.ndarray):
        """
        Initialize the prompt box UI state.
        
        Args:
            args: Argparse namespace with box_center, box_size, box_angles, 
                  focal_length, padding, elevation_min, elevation_max, 
                  num_rings, cameras_per_ring, label
            default_center: Default box center if not specified in args (e.g., splat centroid)
        """
        self.resolution = args.resolution
        
        # Camera/view settings
        self.focal_length = args.focal_length
        self.padding = args.padding
        self.elevation_min = args.elevation_min
        self.elevation_max = args.elevation_max
        self.num_rings = args.num_rings
        self.cameras_per_ring = args.cameras_per_ring
        self.label = args.label
        
        # Build initial transform
        self.transform = self._build_transform(args, default_center)
    
    def _build_transform(self, args, default_center: np.ndarray) -> np.ndarray:
        """Build initial box transform from args or defaults."""
        transform = np.eye(4, dtype=np.float32)
        
        if args.box_center is not None:
            transform[:3, 3] = np.array(args.box_center, dtype=np.float32)
        else:
            transform[:3, 3] = default_center
        
        if args.box_angles is not None:
            angles_rad = np.radians(args.box_angles)
            ax, ay, az = angles_rad
            cx, sx = np.cos(ax), np.sin(ax)
            cy, sy = np.cos(ay), np.sin(ay)
            cz, sz = np.cos(az), np.sin(az)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            R = Rz @ Ry @ Rx
            transform[:3, :3] = R
        
        if args.box_size is not None:
            scale = np.array(args.box_size, dtype=np.float32)
            transform[:3, 0] *= scale[0]
            transform[:3, 1] *= scale[1]
            transform[:3, 2] *= scale[2]
        
        return transform
    
    def _create_unit_box(self):
        """Create a unit cube centered at origin with vertices at ±0.5, plus axis indicators."""
        corners = np.array([
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            [ 0.0,  0.0,  0.0],  # center
            [ 0.5,  0.0,  0.0],  # +X (red)
            [ 0.0, -0.5,  0.0],  # -Y = UP in neg_y_up (green)
            [ 0.0,  0.0, -0.5],  # -Z (blue)
        ], dtype=np.float32)
        
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
            [8, 9], [8, 10], [8, 11],        # axis indicators
        ])
        
        return corners, edges
    
    def register(self):
        """Register the prompt box in Polyscope with gizmo enabled."""
        corners, edges = self._create_unit_box()
        
        box_net = ps.register_curve_network(self.BOX_NAME, corners, edges, radius=0.002)
        box_net.set_color((1.0, 0.5, 0.0))
        
        edge_colors = np.zeros((len(edges), 3), dtype=np.float32)
        edge_colors[:12] = [1.0, 0.5, 0.0]  # orange for box edges
        edge_colors[12] = [1.0, 0.0, 0.0]   # red for X axis
        edge_colors[13] = [0.0, 1.0, 0.0]   # green for Y axis (up)
        edge_colors[14] = [0.0, 0.0, 1.0]   # blue for Z axis
        box_net.add_color_quantity("axis_colors", edge_colors, defined_on='edges', enabled=True)
        
        box_net.set_transform(self.transform)
        box_net.set_transform_gizmo_enabled(True)
    
    def _get_transform(self) -> np.ndarray:
        """Get the current transform of the prompt box."""
        if ps.has_curve_network(self.BOX_NAME):
            return ps.get_curve_network(self.BOX_NAME).get_transform().copy()
        return self.transform.copy()
    
    def get_center(self) -> np.ndarray:
        """Get the world-space center of the prompt box."""
        return self._get_transform()[:3, 3].copy()
    
    def get_up_axis(self) -> np.ndarray:
        """Get the world-space up axis (direction the green edge points)."""
        transform = self._get_transform()
        y_axis = transform[:3, 1].copy()
        return -y_axis / np.linalg.norm(y_axis)
    
    def get_properties(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract center, size, and rotation angles from the box transform.
        
        Returns:
            center: (3,) box center position
            size: (3,) box dimensions
            angles_deg: (3,) rotation angles in degrees (XYZ Euler)
        """
        transform = self._get_transform()
        center = transform[:3, 3]
        
        scale_x = np.linalg.norm(transform[:3, 0])
        scale_y = np.linalg.norm(transform[:3, 1])
        scale_z = np.linalg.norm(transform[:3, 2])
        size = np.array([scale_x, scale_y, scale_z])
        
        R = transform[:3, :3].copy()
        R[:, 0] /= scale_x
        R[:, 1] /= scale_y
        R[:, 2] /= scale_z
        
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            angle_x = np.arctan2(R[2, 1], R[2, 2])
            angle_y = np.arctan2(-R[2, 0], sy)
            angle_z = np.arctan2(R[1, 0], R[0, 0])
        else:
            angle_x = np.arctan2(-R[1, 2], R[1, 1])
            angle_y = np.arctan2(-R[2, 0], sy)
            angle_z = 0
        
        angles_deg = np.degrees([angle_x, angle_y, angle_z])
        
        return center, size, angles_deg
    
    def get_rotation(self) -> np.ndarray:
        """
        Extract the rotation matrix from the box transform.
        
        Returns:
            R: (3, 3) rotation matrix (orthonormal, scale removed)
        """
        transform = self._get_transform()
        
        scale_x = np.linalg.norm(transform[:3, 0])
        scale_y = np.linalg.norm(transform[:3, 1])
        scale_z = np.linalg.norm(transform[:3, 2])
        
        R = transform[:3, :3].copy()
        R[:, 0] /= scale_x
        R[:, 1] /= scale_y
        R[:, 2] /= scale_z
        
        return R
    
    def clear_camera_previews(self):
        """Remove all camera preview objects from Polyscope."""
        for i in range(500):
            name = f"Cam_{i}"
            if ps.has_camera_view(name):
                ps.remove_camera_view(name)
    
    def register_cameras(self, viewmats: torch.Tensor, Ks: torch.Tensor, 
                         masked_rgbs: list = None, widget_focal_length: float = 0.02,
                         color: tuple = (0.6, 0.2, 0.8)):
        """
        Register camera frustums in Polyscope.
        
        Args:
            viewmats: (N, 4, 4) world-to-camera matrices
            Ks: (N, 3, 3) intrinsic matrices
            masked_rgbs: Optional list of RGB images to attach to cameras
            widget_focal_length: Size of camera widget in scene
            color: RGB tuple for camera widget color
        """
        for i in range(len(viewmats)):
            c2w = torch.linalg.inv(viewmats[i]).cpu().numpy()
            root = c2w[:3, 3]
            look_dir = c2w[:3, 2]
            up_dir = -c2w[:3, 1]
            
            focal_px = Ks[i, 0, 0].item()
            width = Ks[i, 0, 2].item() * 2
            height = Ks[i, 1, 2].item() * 2
            fov_y = 2 * math.atan(height / (2 * focal_px)) * (180 / math.pi)
            
            params = ps.CameraParameters(
                ps.CameraIntrinsics(fov_vertical_deg=fov_y, aspect=width/height),
                ps.CameraExtrinsics(root=root, look_dir=look_dir, up_dir=up_dir)
            )
            
            cam = ps.register_camera_view(f"Cam_{i}", params)
            cam.set_widget_focal_length(widget_focal_length)
            cam.set_widget_color(color)
            
            if masked_rgbs is not None:
                cam.add_color_image_quantity(f"MaskedView_{i}", masked_rgbs[i], 
                                            enabled=True, show_in_camera_billboard=True)
    
    def preview_cameras(self):
        """Compute orbit cameras and display frustums in Polyscope."""
        self.clear_camera_previews()
        
        center, size, _ = self.get_properties()
        
        # Update Polyscope viewer FOV to match
        fov_y_deg = 2 * math.atan(self.resolution / (2 * self.focal_length)) * (180 / math.pi)
        ps.set_view_camera_parameters(ps.CameraParameters(
            ps.CameraIntrinsics(fov_vertical_deg=fov_y_deg, aspect=1.0),
            ps.get_view_camera_parameters().get_extrinsics()
        ))
        
        cam_radius = compute_orbit_radius(size, focal_length=self.focal_length, 
                                          image_size=self.resolution, padding=self.padding)
        
        viewmats, cameras_per_ring = get_batch_viewmats(
            center=self.get_center(), radius=cam_radius,
            num_rings=int(self.num_rings), cameras_per_ring=int(self.cameras_per_ring),
            elevation_min=self.elevation_min, elevation_max=self.elevation_max,
            up_axis=self.get_up_axis(), device='cpu'
        )
        num_views = len(viewmats)
        Ks = get_batch_Ks(self.focal_length, self.resolution, self.resolution, num_views, device='cpu')
        
        widget_size = compute_widget_focal_length(cam_radius, cameras_per_ring)
        self.register_cameras(viewmats, Ks, widget_focal_length=widget_size, color=(0.6, 0.2, 0.8))
        
        print(f"[Preview] {num_views} cameras ({int(self.num_rings)} rings × {int(self.cameras_per_ring)}/ring) "
              f"at radius {cam_radius:.2f}, elevation {self.elevation_min:.1f}°-{self.elevation_max:.1f}°")
    
    def make_ui_callback(self):
        """Create the ImGui callback for the prompt box interface."""
        
        def ui_callback():
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if not psim.TreeNode("Prompt Box Setup"):
                return

            psim.Separator()
            psim.Text("Object Label (for SAM3):")
            _, self.label = psim.InputText("##label", self.label)
            
            psim.Separator()
            psim.Text("Adjust the orange box to fit the target object.")
            psim.Text("Green edge = UP axis for camera orbit.")
            psim.Spacing()
            psim.Text("Tip: Right-click 'Prompt Box' > Transform")
            psim.Text("  to enable scaling and rotation controls.")
            
            psim.Separator()
            
            center, size, angles_deg = self.get_properties()
            psim.Text(f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            psim.Text(f"Size:   ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")
            psim.Text(f"Angles: ({angles_deg[0]:.1f}, {angles_deg[1]:.1f}, {angles_deg[2]:.1f})°")
            
            psim.Separator()
            psim.Text("Camera Settings:")
            
            _, self.num_rings = psim.InputInt("Num Rings", int(self.num_rings))
            _, self.cameras_per_ring = psim.InputInt("Cameras/Ring", int(self.cameras_per_ring))
            psim.Text(f"Total views: {int(self.num_rings) * int(self.cameras_per_ring)}")
            _, self.focal_length = psim.DragFloat("Focal Length", self.focal_length, 1.0, 100.0, 1500.0)
            _, self.padding = psim.DragFloat("Padding", self.padding, 0.05, 0.5, 3.0)
            _, self.elevation_min = psim.DragFloat("Elevation Min", self.elevation_min, 1.0, -89.0, 89.0)
            _, self.elevation_max = psim.DragFloat("Elevation Max", self.elevation_max, 1.0, -89.0, 89.0)
            
            psim.Separator()
            
            if psim.Button("Preview Cameras"):
                self.preview_cameras()
            
            psim.SameLine()
            if psim.Button("Clear Cameras"):
                self.clear_camera_previews()
            
            psim.Separator()
            psim.TextColored((0.7, 0.7, 0.7, 1.0), "Close window to start pipeline.")

            psim.TreePop()

        return ui_callback
    
    def print_properties(self):
        """Print box properties for reproducible scripted runs."""
        center, size, angles_deg = self.get_properties()
        print(f"\n[Box Properties] For scripted runs:")
        print(f"  --box_center {center[0]:.4f} {center[1]:.4f} {center[2]:.4f}")
        print(f"  --box_size {size[0]:.4f} {size[1]:.4f} {size[2]:.4f}")
        print(f"  --box_angles {angles_deg[0]:.2f} {angles_deg[1]:.2f} {angles_deg[2]:.2f}")
        if self.label.strip():
            print(f"  --label \"{self.label}\"")