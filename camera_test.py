import polyscope as ps
import numpy as np

ps.init()

# Create a camera view from parameters
intrinsics = ps.CameraIntrinsics(fov_vertical_deg=60, aspect=2)
extrinsics = ps.CameraExtrinsics(root=(2., 2., 2.), look_dir=(-1., -1.,-1.), up_dir=(0.,1.,0.))
params = ps.CameraParameters(intrinsics, extrinsics)
cam = ps.register_camera_view("cam", params)

# Set some options for the camera view
# these can also be set as keyword args in register_camera_view()
cam.set_widget_focal_length(0.05)          # size of displayed widget (relative value)
cam.set_widget_thickness(0.05)             # thickness of widget lines
cam.set_widget_color((0.25, 0.25, 0.25))   # color of widget lines


# Add an image to be displayed in the camera frame
w = 600
h = 300
cam.add_scalar_image_quantity("scalar_img", np.zeros((h, w)),
                              enabled=True, show_in_camera_billboard=True)

ps.show()