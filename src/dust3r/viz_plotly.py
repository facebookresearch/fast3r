import numpy as np
import plotly.graph_objects as go
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy.spatial.transform import Rotation


class SceneViz:
    def __init__(self):
        self.fig = go.Figure()

    def export_html(self, filename="scene_visualization.html"):
        """Exports the current figure to a self-contained HTML file."""
        # Use Plotly's write_html to save the figure as a standalone HTML file
        self.fig.write_html(filename, include_plotlyjs='cdn', full_html=True)

        print(f"Visualization exported to {filename}")

    def add_pointcloud(self, pts3d, color, mask=None, point_size=1, view_idx=0):
        """Adds a point cloud to the Plotly figure using original colors."""
        pts3d = np.array(pts3d)
        if mask is None:
            mask = np.ones(pts3d.shape[:2], dtype=bool)

        masked_pts = pts3d[mask]
        masked_color = color[mask]

        # Add point cloud with original colors and adjustable point size
        self.fig.add_trace(go.Scatter3d(
            x=masked_pts[:, 0],
            y=masked_pts[:, 1],
            z=masked_pts[:, 2],
            mode='markers',
            marker=dict(size=point_size, color=masked_color.reshape(-1, 3), opacity=0.8),
            name=f"Point Cloud {view_idx}"
        ))
        return self

    def clamp_color(self, rgb_color):
        """Ensure that RGB values are clamped between 0 and 255."""
        return tuple(max(0, min(255, int(c))) for c in rgb_color)

    def add_camera(self, pose_c2w, focal=None, color=(0, 0, 0), image=None, imsize=None, cam_size=0.02, view_idx=0, enable_color_image=True):
        """Adds a camera frustum to the plot, with an optional image texture."""
        focal = focal if focal is not None else 500
        # Clamp the color values to ensure valid RGB range
        clamped_color = self.clamp_color(color)
        color_str = f'rgb{clamped_color}'

        # Create frustum and image surface or mesh3d
        frustum_traces = create_camera_frustum_with_image(
            pose_c2w, focal=focal, H=imsize[0] if imsize else 1080, W=imsize[1] if imsize else 1920,
            image=image, screen_width=cam_size, color=color_str, view_idx=view_idx, enable_color_image=enable_color_image
        )

        # Add the traces for both frustum and image
        for trace in frustum_traces:
            if trace:
                trace.update(legendgroup=f"frustum_{id(pose_c2w)}", showlegend=True)
                self.fig.add_trace(trace)

        return self

    def add_cameras(self, poses, focals=None, images=None, imsizes=None, colors=None, cam_size=0.02, enable_color_image=True):
        """Add multiple cameras with adjustable frustum size."""
        def get(arr, idx):
            return None if arr is None else arr[idx]

        for i, pose_c2w in enumerate(poses):
            self.add_camera(
                pose_c2w,
                focal=get(focals, i),
                image=get(images, i),
                color=get(colors, i),
                imsize=get(imsizes, i),
                cam_size=cam_size,  # Frustum size control
                view_idx=i,  # Use view index for naming
                enable_color_image=enable_color_image  # Flag to enable or disable colored images
            )
        return self

    def show(self, point_size=1, viewer=None):
        self.fig.update_layout(
            title="Camera Poses and Point Clouds",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=800,
        )
        self.fig.show()
        return self


def image2zvals(img, n_colors=64, n_training_pixels=1000):
    """Quantize the image using KMeans for color mapping."""
    rows, cols, _ = img.shape
    img = np.clip(img / 255.0, 0, 1)  # Normalize the image

    # Flatten and shuffle for KMeans clustering
    observations = img[:, :, :3].reshape(rows * cols, 3)
    training_pixels = shuffle(observations, random_state=42)[:n_training_pixels]

    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(training_pixels)
    codebook = kmeans.cluster_centers_
    indices = kmeans.predict(observations)

    z_vals = indices.astype(float) / (n_colors - 1)  # Normalize indices to [0, 1]
    z_vals = z_vals.reshape(rows, cols)

    # Create Plotly color scale
    scale = np.linspace(0, 1, n_colors)
    colors = (codebook * 255).astype(np.uint8)
    plotly_colorscale = [[s, f'rgb{tuple(c)}'] for s, c in zip(scale, colors)]

    return z_vals, plotly_colorscale


def regular_triangles(rows, cols):
    """Generate regular triangles for a mesh."""
    triangles = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            k = j + i * cols
            triangles.extend([[k, k + cols, k + 1 + cols], [k, k + 1 + cols, k + 1]])
    return np.array(triangles)


def mesh_data(img, resolution=128, n_colors=64, n_training_pixels=1000):
    """Generate mesh data with quantized color intensities for the image."""
    img_downsampled = np.array(Image.fromarray(img).resize((resolution, resolution)))

    # Quantize the downsampled image
    z_vals, pl_colorscale = image2zvals(img_downsampled, n_colors=n_colors, n_training_pixels=n_training_pixels)

    # Generate triangles
    rows, cols, _ = img_downsampled.shape
    triangles = regular_triangles(rows, cols)
    I, J, K = triangles.T

    # Assign intensity to each triangle
    zc = z_vals.flatten()[triangles]
    tri_color_intensity = [zc[k][2] if k % 2 else zc[k][1] for k in range(len(zc))]

    return I, J, K, tri_color_intensity, pl_colorscale


def generate_meshgrid(frustum_points_world, resolution):
    """Generate a meshgrid and calculate X, Y, Z for the surface or mesh3d."""
    img_x, img_y, img_z = frustum_points_world[1:5, 0], frustum_points_world[1:5, 1], frustum_points_world[1:5, 2]
    u = np.linspace(0, 1, resolution)
    v = np.linspace(0, 1, resolution)
    uu, vv = np.meshgrid(u, v)

    X = img_x[0] * (1 - uu) * (1 - vv) + img_x[1] * uu * (1 - vv) + img_x[3] * (1 - uu) * vv + img_x[2] * uu * vv
    Y = img_y[0] * (1 - uu) * (1 - vv) + img_y[1] * uu * (1 - vv) + img_y[3] * (1 - uu) * vv + img_y[2] * uu * vv
    Z = img_z[0] * (1 - uu) * (1 - vv) + img_z[1] * uu * (1 - vv) + img_z[3] * (1 - uu) * vv + img_z[2] * uu * vv

    return X, Y, Z


def create_mesh3d(pose_c2w, img, frustum_points_world, resolution=64, n_colors=64, view_idx=0):
    """Creates a Mesh3d object for the image texture mapping."""
    X, Y, Z = generate_meshgrid(frustum_points_world, resolution)

    # Get the mesh data
    I, J, K, tri_color_intensity, pl_colorscale = mesh_data(img, resolution=resolution, n_colors=n_colors)

    # Create the Mesh3d trace
    mesh3d_trace = go.Mesh3d(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),  # Use frustum base interpolation
        i=I, j=J, k=K,
        intensity=tri_color_intensity,
        intensitymode="cell",
        colorscale=pl_colorscale,
        showscale=False,
        name=f"Image {view_idx}"
    )

    return mesh3d_trace


def create_surface(frustum_points_world, img_downsampled, z_vals, resolution=64, view_idx=0):
    """Creates a Surface object with grayscale for faster rendering."""
    X, Y, Z = generate_meshgrid(frustum_points_world, resolution)

    # Create a grayscale surface
    surface_trace = go.Surface(
        x=X, y=Y, z=Z, surfacecolor=z_vals, colorscale='gray', showscale=False, name=f"Image {view_idx}"
    )

    return surface_trace


def create_camera_frustum_with_image(pose_c2w, focal, H, W, image=None, screen_width=0.02, color='blue', resolution=128, view_idx=0, enable_color_image=True):
    """Creates a frustum for a camera and optionally adds an image to the frustum."""
    if image is not None:
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)

    # Calculate frustum size based on screen width and focal length
    depth = focal * screen_width / H
    hw_ratio = W / H

    # Define frustum points in camera space
    frustum_points = np.array([
        [0, 0, 0],  # Camera origin
        [-hw_ratio * depth, -depth, depth],  # Bottom left
        [hw_ratio * depth, -depth, depth],  # Bottom right
        [hw_ratio * depth, depth, depth],  # Top right
        [-hw_ratio * depth, depth, depth],  # Top left
    ])

    # Transform points to world coordinates
    frustum_points_homogeneous = np.hstack([frustum_points, np.ones((frustum_points.shape[0], 1))])
    frustum_points_world = (pose_c2w @ frustum_points_homogeneous.T).T[:, :3]

    # Define edges of the frustum
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    x_vals, y_vals, z_vals = [], [], []
    for edge in edges:
        x_vals += [frustum_points_world[edge[0], 0], frustum_points_world[edge[1], 0], None]
        y_vals += [frustum_points_world[edge[0], 1], frustum_points_world[edge[1], 1], None]
        z_vals += [frustum_points_world[edge[0], 2], frustum_points_world[edge[1], 2], None]

    # Add frustum lines
    frustum_trace = go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals, mode='lines',
        line=dict(color=color, width=2), name=f"Camera {view_idx}"
    )

    # Optionally add image to the base of the frustum
    image_surface_trace = None
    if image is not None:
        if enable_color_image:
            # plotly doesn't natively support texture mapping, so we use Mesh3d for colored image
            # see: https://github.com/empet/Texture-mapping-with-Plotly/blob/main/Texture-mapping-surface.ipynb
            # Create the Mesh3d for colored image
            image_surface_trace = create_mesh3d(pose_c2w, img=image, frustum_points_world=frustum_points_world, resolution=resolution, n_colors=64, view_idx=view_idx)
        else:
            # Downsample and use grayscale for faster performance
            img_downsampled = np.array(Image.fromarray(image).resize((resolution, resolution)))
            z_vals = np.mean(img_downsampled / 255.0, axis=-1)
            image_surface_trace = create_surface(frustum_points_world, img_downsampled, z_vals, resolution=resolution, view_idx=view_idx)

    return [frustum_trace, image_surface_trace] if image_surface_trace else [frustum_trace]
