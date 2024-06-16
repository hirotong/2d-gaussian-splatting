import numpy as np
import plotly.graph_objects as go


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * np.array([-2, 1.5, 4])
    up1 = 0.5 * np.array([0, 1.5, 4])
    up2 = 0.5 * np.array([0, 2, 4])
    b = 0.5 * np.array([2, 1.5, 4])
    c = 0.5 * np.array([-2, -1.5, 4])
    d = 0.5 * np.array([2, -1.5, 4])
    C = np.zeros(3)
    F = np.array([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = np.stack([x for x in camera_points]) * scale
    # lines[:, 1] = -lines[:, 1]
    lines[:, 2] = -lines[:, 2]
    
    return lines


def add_camera_trace(
    fig: go.Figure, poses: np.ndarray, trace_name: str, camera_scale: float = 0.3
):
    """
    Adds a camera trace to the plotly figure.
    """
    cam_wires = get_camera_wireframe(camera_scale)
    cam_wires = cam_wires[None].repeat(len(poses), axis=0)
    cam_wires_trans = (
        np.matmul(poses[:, :3, :3], cam_wires.transpose(0, 2, 1)) + poses[:, :3, 3:]
    )
    cam_wires_trans = cam_wires_trans.transpose(0, 2, 1)
    nan_array = np.array([[float("nan")] * 3])

    all_cam_wires = cam_wires_trans[0]
    for wire in cam_wires_trans[1:]:
        all_cam_wires = np.concatenate([all_cam_wires, nan_array, wire], axis=0)

    x, y, z = all_cam_wires.T

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker={"size": 1}, name=trace_name))
