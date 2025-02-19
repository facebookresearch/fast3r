import time
import threading
import numpy as np
from tqdm.auto import tqdm
import imageio.v3 as iio
from matplotlib import cm

import viser
import viser.transforms as tf
from src.dust3r.utils.device import to_numpy
from src.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

def start_visualization(output, min_conf_thr_percentile=10, global_conf_thr_value_to_drop_view=1.5, port=8020):
    # Create the viser server on the specified port
    server = viser.ViserServer(host='127.0.0.1', port=port)

    # Estimate camera poses
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output['preds'], niter_PnP=100, focal_length_estimation_method='first_view_from_global_head'
    )
    poses_c2w = poses_c2w_batch[0]  # Assuming batch size of 1

    # Set the upward direction to negative Y-axis
    server.scene.set_up_direction((0.0, -1.0, 0.0))
    server.scene.world_axes.visible = False  # Optional: Hide world axes

    num_frames = len(output['preds'])

    # Prepare lists to store per-frame data
    frame_data_list = []

    # Generate colors for frustums and points in rainbow order
    def rainbow_color(n, total):
        import colorsys
        hue = n / total
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return rgb

    # Add playback UI
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider("Point size", min=0.000001, max=0.002, step=1e-5, initial_value=0.0005)
        gui_frustum_size_percent = server.gui.add_slider("Camera Size (%)", min=0.1, max=10.0, step=0.1, initial_value=2.0)
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=True)
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=0.25, max=60, step=0.25, initial_value=10)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("0.5", "1", "10", "20", "30", "60"))

    # Add point cloud options UI
    with server.gui.add_folder("Point Cloud Options"):
        gui_show_global = server.gui.add_checkbox("Global", False)
        gui_show_local = server.gui.add_checkbox("Local", True)

    # Add view options UI
    with server.gui.add_folder("View Options"):
        gui_show_high_conf = server.gui.add_checkbox("Show High-Conf Views", True)
        gui_show_low_conf = server.gui.add_checkbox("Show Low-Conf Views", False)
        gui_global_conf_threshold = server.gui.add_slider("High/Low Conf threshold value", min=1.0, max=12.0, step=0.1, initial_value=global_conf_thr_value_to_drop_view)
        gui_min_conf_percentile = server.gui.add_slider("Per-View conf percentile", min=0, max=100, step=1, initial_value=min_conf_thr_percentile)

    # Add color options UI
    with server.gui.add_folder("Color Options"):
        gui_show_confidence = server.gui.add_checkbox("Show Confidence", False)
        gui_rainbow_color = server.gui.add_checkbox("Rainbow Colors", False)

    button_render_gif = server.gui.add_button("Render a GIF")

    # Frame step buttons
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = float(gui_framerate_options.value)

    server.scene.add_frame("/cams", show_axes=False)

    # First pass: Collect data and compute scene extent
    cumulative_pts = []

    for i in tqdm(range(num_frames)):
        pred = output['preds'][i]
        view = output['views'][i]

        # Extract global and local points and confidences
        pts3d_global = to_numpy(pred['pts3d_in_other_view'].cpu().squeeze())
        conf_global = to_numpy(pred['conf'].cpu().squeeze())
        pts3d_local = to_numpy(pred['pts3d_local_aligned_to_global'].cpu().squeeze())
        conf_local = to_numpy(pred['conf_local'].cpu().squeeze())
        img_rgb = to_numpy(view['img'].cpu().squeeze().permute(1, 2, 0))

        # Reshape and flatten data
        pts3d_global = pts3d_global.reshape(-1, 3)
        pts3d_local = pts3d_local.reshape(-1, 3)
        img_rgb = img_rgb.reshape(-1, 3)
        conf_global = conf_global.flatten()
        conf_local = conf_local.flatten()

        cumulative_pts.append(pts3d_global)

        # Store per-frame data
        frame_data = {}

        # Sort points by confidence in descending order
        # For global point cloud
        sort_indices_global = np.argsort(-conf_global)
        sorted_conf_global = conf_global[sort_indices_global]
        sorted_pts3d_global = pts3d_global[sort_indices_global]
        sorted_img_rgb_global = img_rgb[sort_indices_global]

        # For local point cloud
        sort_indices_local = np.argsort(-conf_local)
        sorted_conf_local = conf_local[sort_indices_local]
        sorted_pts3d_local = pts3d_local[sort_indices_local]
        sorted_img_rgb_local = img_rgb[sort_indices_local]

        # Normalize colors
        colors_rgb_global = ((sorted_img_rgb_global + 1) * 127.5).astype(np.uint8) / 255.0  # Values in [0,1]
        colors_rgb_local = ((sorted_img_rgb_local + 1) * 127.5).astype(np.uint8) / 255.0  # Values in [0,1]

        # Precompute confidence-based colors
        conf_norm_global = (sorted_conf_global - sorted_conf_global.min()) / (sorted_conf_global.max() - sorted_conf_global.min() + 1e-8)
        conf_norm_local = (sorted_conf_local - sorted_conf_local.min()) / (sorted_conf_local.max() - sorted_conf_local.min() + 1e-8)
        colormap = cm.turbo
        colors_confidence_global = colormap(conf_norm_global)[:, :3]  # Values in [0,1]
        colors_confidence_local = colormap(conf_norm_local)[:, :3]  # Values in [0,1]

        # Rainbow color for the frame's points
        rainbow_color_for_frame = rainbow_color(i, num_frames)
        colors_rainbow_global = np.tile(rainbow_color_for_frame, (sorted_pts3d_global.shape[0], 1))
        colors_rainbow_local = np.tile(rainbow_color_for_frame, (sorted_pts3d_local.shape[0], 1))

        # Compute initial high-confidence flag based on global confidence
        max_conf_global = conf_global.max()
        is_high_confidence = max_conf_global >= gui_global_conf_threshold.value

        # Camera parameters
        c2w = poses_c2w[i]
        height, width = view['img'].shape[2], view['img'].shape[3]
        focal_length = estimated_focals[0][i]
        img_rgb_reshaped = img_rgb.reshape(height, width, 3)
        img_rgb_normalized = ((img_rgb_reshaped + 1) * 127.5).astype(np.uint8)  # Values in [0,255]
        img_downsampled = img_rgb_normalized[::4, ::4]  # Keep as uint8

        # Store all precomputed data
        frame_data['sorted_pts3d_global'] = sorted_pts3d_global
        frame_data['colors_rgb_global'] = colors_rgb_global
        frame_data['colors_confidence_global'] = colors_confidence_global
        frame_data['colors_rainbow_global'] = colors_rainbow_global

        frame_data['sorted_pts3d_local'] = sorted_pts3d_local
        frame_data['colors_rgb_local'] = colors_rgb_local
        frame_data['colors_confidence_local'] = colors_confidence_local
        frame_data['colors_rainbow_local'] = colors_rainbow_local

        frame_data['max_conf_global'] = max_conf_global
        frame_data['is_high_confidence'] = is_high_confidence

        frame_data['c2w'] = c2w
        frame_data['height'] = height
        frame_data['width'] = width
        frame_data['focal_length'] = focal_length
        frame_data['img_downsampled'] = img_downsampled
        frame_data['rainbow_color'] = rainbow_color_for_frame

        frame_data_list.append(frame_data)

    # Compute scene extent and max_extent
    cumulative_pts_combined = np.concatenate(cumulative_pts, axis=0)
    min_coords = np.min(cumulative_pts_combined, axis=0)
    max_coords = np.max(cumulative_pts_combined, axis=0)
    scene_extent = max_coords - min_coords
    max_extent = np.max(scene_extent)

    # Now create the visualization nodes
    for i in tqdm(range(num_frames)):
        frame_data = frame_data_list[i]

        # Initialize frame node
        frame_node = server.scene.add_frame(f"/cams/t{i}", show_axes=False)

        # Initialize point cloud nodes
        # Global point cloud
        point_node_global = server.scene.add_point_cloud(
            name=f"/pts3d_global/t{i}",
            points=frame_data['sorted_pts3d_global'],
            colors=frame_data['colors_rgb_global'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,  # Initially hidden
        )

        # Local point cloud
        point_node_local = server.scene.add_point_cloud(
            name=f"/pts3d_local/t{i}",
            points=frame_data['sorted_pts3d_local'],
            colors=frame_data['colors_rgb_local'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=True if frame_data_list[i]['is_high_confidence'] else False,
        )

        # Compute frustum parameters
        c2w = frame_data['c2w']
        rotation_matrix = c2w[:3, :3]
        position = c2w[:3, 3]
        rotation_quaternion = tf.SO3.from_matrix(rotation_matrix).wxyz

        fov = 2 * np.arctan2(frame_data['height'] / 2, frame_data['focal_length'])
        aspect_ratio = frame_data['width'] / frame_data['height']
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)

        frustum_node = server.scene.add_camera_frustum(
            name=f"/cams/t{i}/frustum",
            fov=fov,
            aspect=aspect_ratio,
            scale=frustum_scale,
            color=frame_data['rainbow_color'],
            image=frame_data['img_downsampled'],
            wxyz=rotation_quaternion,
            position=position,
            visible=True if frame_data_list[i]['is_high_confidence'] else False,
        )

        # Store nodes
        frame_data['frame_node'] = frame_node
        frame_data['point_node_global'] = point_node_global
        frame_data['point_node_local'] = point_node_local
        frame_data['frustum_node'] = frustum_node

    # Set initial visibility
    for frame_data in frame_data_list:
        frame_data['frame_node'].visible = False
        frame_data['point_node_global'].visible = False
        frame_data['point_node_local'].visible = False
        frame_data['frustum_node'].visible = False

    def update_visibility():
        current_timestep = int(gui_timestep.value)
        with server.atomic():
            for i in range(num_frames):
                frame_data = frame_data_list[i]
                if i <= current_timestep:
                    is_high_confidence = frame_data['is_high_confidence']
                    show_frame = False
                    if is_high_confidence and gui_show_high_conf.value:
                        show_frame = True
                    if not is_high_confidence and gui_show_low_conf.value:
                        show_frame = True

                    # Update visibility based on global point cloud confidence
                    frame_data['frame_node'].visible = show_frame
                    frame_data['frustum_node'].visible = show_frame

                    # Show/hide global point cloud
                    frame_data['point_node_global'].visible = show_frame and gui_show_global.value

                    # Show/hide local point cloud
                    frame_data['point_node_local'].visible = show_frame and gui_show_local.value
                else:
                    frame_data['frame_node'].visible = False
                    frame_data['frustum_node'].visible = False
                    frame_data['point_node_global'].visible = False
                    frame_data['point_node_local'].visible = False
        server.flush()

    @gui_timestep.on_update
    def _(_) -> None:
        update_visibility()

    @gui_point_size.on_update
    def _(_) -> None:
        with server.atomic():
            for frame_data in frame_data_list:
                frame_data['point_node_global'].point_size = gui_point_size.value
                frame_data['point_node_local'].point_size = gui_point_size.value
        server.flush()

    @gui_frustum_size_percent.on_update
    def _(_) -> None:
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)
        with server.atomic():
            for frame_data in frame_data_list:
                frame_data['frustum_node'].scale = frustum_scale
        server.flush()

    @gui_show_confidence.on_update
    def _(_) -> None:
        update_point_cloud_colors()

    @gui_rainbow_color.on_update
    def _(_) -> None:
        update_point_cloud_colors()

    @gui_show_global.on_update
    def _(_) -> None:
        update_visibility()

    @gui_show_local.on_update
    def _(_) -> None:
        update_visibility()

    def update_point_cloud_colors():
        with server.atomic():
            for frame_data in frame_data_list:
                num_points_to_show_global = frame_data.get('num_points_to_show_global', len(frame_data['sorted_pts3d_global']))
                num_points_to_show_local = frame_data.get('num_points_to_show_local', len(frame_data['sorted_pts3d_local']))

                # Update global point cloud colors
                if gui_show_confidence.value:
                    colors_global = frame_data['colors_confidence_global'][:num_points_to_show_global]
                elif gui_rainbow_color.value:
                    colors_global = frame_data['colors_rainbow_global'][:num_points_to_show_global]
                else:
                    colors_global = frame_data['colors_rgb_global'][:num_points_to_show_global]
                frame_data['point_node_global'].colors = colors_global

                # Update local point cloud colors
                if gui_show_confidence.value:
                    colors_local = frame_data['colors_confidence_local'][:num_points_to_show_local]
                elif gui_rainbow_color.value:
                    colors_local = frame_data['colors_rainbow_local'][:num_points_to_show_local]
                else:
                    colors_local = frame_data['colors_rgb_local'][:num_points_to_show_local]
                frame_data['point_node_local'].colors = colors_local
        server.flush()

    @gui_show_high_conf.on_update
    def _(_) -> None:
        update_visibility()

    @gui_show_low_conf.on_update
    def _(_) -> None:
        update_visibility()

    @gui_global_conf_threshold.on_update
    def _(_) -> None:
        # Update high-confidence flags based on new threshold
        for frame_data in frame_data_list:
            is_high_confidence = frame_data['max_conf_global'] >= gui_global_conf_threshold.value
            frame_data['is_high_confidence'] = is_high_confidence
        update_visibility()

    @gui_min_conf_percentile.on_update
    def _(_) -> None:
        # Update number of points to display based on percentile
        percentile = gui_min_conf_percentile.value
        with server.atomic():
            for frame_data in frame_data_list:
                # For global point cloud
                total_points_global = len(frame_data['sorted_pts3d_global'])
                num_points_to_show_global = int(total_points_global * (100 - percentile) / 100)
                num_points_to_show_global = max(1, num_points_to_show_global)  # Ensure at least one point
                frame_data['num_points_to_show_global'] = num_points_to_show_global
                frame_data['point_node_global'].points = frame_data['sorted_pts3d_global'][:num_points_to_show_global]

                # For local point cloud
                total_points_local = len(frame_data['sorted_pts3d_local'])
                num_points_to_show_local = int(total_points_local * (100 - percentile) / 100)
                num_points_to_show_local = max(1, num_points_to_show_local)  # Ensure at least one point
                frame_data['num_points_to_show_local'] = num_points_to_show_local
                frame_data['point_node_local'].points = frame_data['sorted_pts3d_local'][:num_points_to_show_local]

            # Update colors
            update_point_cloud_colors()
        server.flush()

    def playback_loop():
        while True:
            if gui_playing.value:
                gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
            time.sleep(1.0 / gui_framerate.value)

    playback_thread = threading.Thread(target=playback_loop)
    playback_thread.start()

    @button_render_gif.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            print("Error: No client connected.")
            return
        try:
            images = []
            original_timestep = gui_timestep.value
            original_playing = gui_playing.value
            gui_playing.value = False
            fps = gui_framerate.value
            for i in range(num_frames):
                gui_timestep.value = i
                time.sleep(0.1)
                image = client.get_render(height=720, width=1280)
                images.append(image)
            gif_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps, loop=0)
            client.send_file_download("visualization.gif", gif_bytes)
            gui_timestep.value = original_timestep
            gui_playing.value = original_playing
        except Exception as e:
            print(f"Error while rendering GIF: {e}")

    public_url = server.request_share_url()
    return server

# Start the visualization server
# server = start_visualization(
#     output=output,
#     min_conf_thr_percentile=10,
#     global_conf_thr_value_to_drop_view=1.5,
#     port=8020
# )
