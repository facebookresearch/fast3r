# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import threading
import numpy as np
from tqdm.auto import tqdm
import imageio.v3 as iio
from matplotlib import cm
import cv2
from scipy import ndimage
import torch

import viser
import viser.transforms as tf
from fast3r.dust3r.utils.device import to_numpy
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# ----------------- Helper Functions -----------------

def detect_sky_mask(img_rgb):
    """
    Detect sky pixels using HSV color space and morphological operations.
    Args:
        img_rgb: RGB image normalized to [-1, 1]
    Returns:
        Boolean mask (as int8) where True indicates non-sky pixels.
    """
    img = ((img_rgb + 1) * 127.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([105, 50, 140])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_light_blue = np.array([95, 5, 150])
    upper_light_blue = np.array([145, 100, 255])
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    lower_white = np.array([0, 0, 235])
    upper_white = np.array([180, 10, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = mask_blue | mask_light_blue | mask_white

    height = mask.shape[0]
    upper_third = int(height * 0.4)
    upper_region = hsv[:upper_third, :, :]
    mask[:upper_third, :] |= ((upper_region[:, :, 1] < 50) & (upper_region[:, :, 2] > 150))

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = mask.astype(bool)
    labels, num_labels = ndimage.label(mask)
    if num_labels > 0:
        top_row_labels = set(labels[0, :])
        top_row_labels.discard(0)
        if top_row_labels:
            mask = np.isin(labels, list(top_row_labels))
            labels, num_labels = ndimage.label(mask)
            if num_labels > 0:
                sizes = ndimage.sum(mask, labels, range(1, num_labels + 1))
                mask_size = mask.size
                big_enough = sizes > mask_size * 0.01
                mask = np.isin(labels, np.where(big_enough)[0] + 1)
    return (~mask).astype(np.int8)

def is_outdoor_scene(frame_data_list):
    sky_ratios = []
    for fd in frame_data_list:
        mask = fd.get('sorted_not_sky_global', np.ones(1))
        sky_ratio = 1.0 - np.mean(mask)
        sky_ratios.append(float(sky_ratio))
    significant = sum(1 for ratio in sky_ratios if ratio > 0.2)
    return significant >= len(sky_ratios) / 4

# ----------------- Update Handlers -----------------
# These functions are lightweight and respond only to their respective events.
def update_frame_visibility(server, frame_data_list, gui_timestep, num_frames, gui_show_global, gui_show_local, gui_show_high_conf, gui_show_low_conf):
    current = int(gui_timestep.value)
    with server.atomic():
        for i in range(num_frames):
            fd = frame_data_list[i]
            # For simplicity, we show the frame if i <= current.
            show_frame = (i <= current)
            fd['frame_node'].visible = show_frame
            fd['frustum_node'].visible = show_frame
            fd['point_node_global'].visible = show_frame and gui_show_global.value
            fd['point_node_local'].visible = show_frame and gui_show_local.value
        server.flush()

def update_point_cloud_colors(server, frame_data_list, gui_timestep, gui_show_confidence_color, gui_rainbow_color_option, gui_show_global, gui_show_local):
    with server.atomic():
        for i in range(len(frame_data_list)):
            fd = frame_data_list[i]
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global']
                colors_local = fd['colors_confidence_local']
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global']
                colors_local = fd['colors_rainbow_local']
            else:
                colors_global = fd['colors_rgb_global']
                colors_local = fd['colors_rgb_local']
            fd['point_node_global'].colors = colors_global if gui_show_global.value else []
            fd['point_node_local'].colors = colors_local if gui_show_local.value else []
            server.flush()

def update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option):
    for i in range(len(frame_data_list)):
        fd = frame_data_list[i]
        total_global = len(fd['sorted_pts3d_global'])
        total_local = len(fd['sorted_pts3d_local'])

        # Calculate number of points to show based on the percentile
        num_global = max(1, int(total_global * (100 - gui_min_conf_percentile.value) / 100))
        num_local = max(1, int(total_local * (100 - gui_min_conf_percentile.value) / 100))

        # Apply sky masking if enabled
        if gui_mask_sky.value:
            mask_global = fd['sorted_not_sky_global'][:num_global]
            mask_local = fd['sorted_not_sky_local'][:num_local]
            # Filter points based on the mask
            pts3d_global = fd['sorted_pts3d_global'][:num_global][mask_global > 0]
            pts3d_local = fd['sorted_pts3d_local'][:num_local][mask_local > 0]
            
            # Select the appropriate colors based on the active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_confidence_local'][:num_local][mask_local > 0]
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_rainbow_local'][:num_local][mask_local > 0]
            else:
                colors_global = fd['colors_rgb_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_rgb_local'][:num_local][mask_local > 0]
        else:
            pts3d_global = fd['sorted_pts3d_global'][:num_global]
            pts3d_local = fd['sorted_pts3d_local'][:num_local]

            # Select the appropriate colors based on the active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global'][:num_global]
                colors_local = fd['colors_confidence_local'][:num_local]
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global'][:num_global]
                colors_local = fd['colors_rainbow_local'][:num_local]
            else:
                colors_global = fd['colors_rgb_global'][:num_global]
                colors_local = fd['colors_rgb_local'][:num_local]

        # Update point clouds
        fd['point_node_global'].points = pts3d_global
        fd['point_node_local'].points = pts3d_local
        # update colors
        fd['point_node_global'].colors = colors_global
        fd['point_node_local'].colors = colors_local
                
        server.flush()

# ----------------- Playback Loop -----------------
def playback_loop(gui_playing, gui_timestep, num_frames, gui_framerate):
    while True:
        if gui_playing.value:
            gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
        time.sleep(1.0 / float(gui_framerate.value))

def bind_update(widget, update_func):
    widget.on_update(lambda _: update_func())

# ----------------- Main Visualization Function -----------------
def start_visualization(output, min_conf_thr_percentile=10, global_conf_thr_value_to_drop_view=1.5, port=8020, point_size=0.0004):
    server = viser.ViserServer(host='127.0.0.1', port=port)
    server.gui.set_panel_label("Show Controls")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)

    @server.on_client_connect
    def on_client_connect(client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = (-0.00141163, -0.01910395, -0.06794288)
            client.camera.look_at = (-0.00352821, -0.01143425, 0.0154939)
        client.flush()

    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output['preds'], niter_PnP=100, focal_length_estimation_method='first_view_from_global_head'
    )
    poses_c2w = poses_c2w_batch[0]

    server.scene.set_up_direction((0.0, -1.0, 0.0))
    server.scene.world_axes.visible = False

    num_frames = len(output['preds'])
    frame_data_list = []
    cumulative_pts = []

    # ----------------- Grouped GUI Controls -----------------
    with server.gui.add_folder("Point and Camera Options", expand_by_default=False):
        gui_point_size = server.gui.add_slider("Point Size", min=1e-6, max=0.002, step=1e-5, initial_value=point_size)
        gui_frustum_size_percent = server.gui.add_slider("Camera Size (%)", min=0.1, max=10.0, step=0.1, initial_value=2.0)
        gui_mask_sky = server.gui.add_checkbox("Mask Sky", True)
        gui_show_confidence_color = server.gui.add_checkbox("Show Confidence", False)
        gui_rainbow_color_option = server.gui.add_checkbox("Color by View", False)

    with server.gui.add_folder("Playback Options", expand_by_default=False):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0)
        gui_next_frame = server.gui.add_button("Next Frame")
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider("FPS", min=0.25, max=60, step=0.25, initial_value=10)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("0.5", "1", "10", "20", "30", "60"))

    with server.gui.add_folder("Pointmap Head Options", expand_by_default=False):
        gui_show_global = server.gui.add_checkbox("Global", False)
        gui_show_local = server.gui.add_checkbox("Local", True)

    with server.gui.add_folder("Confidence Options", expand_by_default=False):
        gui_show_high_conf = server.gui.add_checkbox("Show High-Conf Views", True)
        gui_show_low_conf = server.gui.add_checkbox("Show Low-Conf Views", False)
        gui_global_conf_threshold = server.gui.add_slider("High/Low Conf Threshold", min=1.0, max=12.0, step=0.1, initial_value=global_conf_thr_value_to_drop_view)
        gui_min_conf_percentile = server.gui.add_slider("Per-View Conf Percentile", min=0, max=100, step=1, initial_value=min_conf_thr_percentile)

    button_render_gif = server.gui.add_button("Render a GIF")

    @gui_next_frame.on_click
    def next_frame(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def prev_frame(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def playing_update(_):
        state = gui_playing.value
        gui_timestep.disabled = state
        gui_next_frame.disabled = state
        gui_prev_frame.disabled = state

    @gui_framerate_options.on_click
    def fps_options(_):
        gui_framerate.value = float(gui_framerate_options.value)

    server.scene.add_frame("/cams", show_axes=False)

    # ----------------- Frame Processing -----------------
    for i in tqdm(range(num_frames)):
        pred = output['preds'][i]
        view = output['views'][i]

        img_rgb_orig = to_numpy(view['img'].cpu().squeeze().permute(1,2,0))
        not_sky_mask = detect_sky_mask(img_rgb_orig).flatten().astype(np.int8)

        pts3d_global = to_numpy(pred['pts3d_local_aligned_to_global'].cpu().squeeze()).reshape(-1, 3)
        conf_global = to_numpy(pred['conf'].cpu().squeeze()).flatten()
        pts3d_local = to_numpy(pred['pts3d_in_other_view'].cpu().squeeze()).reshape(-1, 3)
        conf_local = to_numpy(pred['conf_local'].cpu().squeeze()).flatten()
        img_rgb = to_numpy(view['img'].cpu().squeeze().permute(1,2,0))
        img_rgb_flat = img_rgb.reshape(-1, 3)

        cumulative_pts.append(pts3d_global)

        sort_idx_global = np.argsort(-conf_global)
        sorted_conf_global = conf_global[sort_idx_global]
        sorted_pts3d_global = pts3d_global[sort_idx_global]
        sorted_img_rgb_global = img_rgb_flat[sort_idx_global]
        sorted_not_sky_global = not_sky_mask[sort_idx_global]

        sort_idx_local = np.argsort(-conf_local)
        sorted_conf_local = conf_local[sort_idx_local]
        sorted_pts3d_local = pts3d_local[sort_idx_local]
        sorted_img_rgb_local = img_rgb_flat[sort_idx_local]
        sorted_not_sky_local = not_sky_mask[sort_idx_local]

        colors_rgb_global = ((sorted_img_rgb_global + 1) * 127.5).astype(np.uint8) / 255.0
        colors_rgb_local = ((sorted_img_rgb_local + 1) * 127.5).astype(np.uint8) / 255.0

        conf_norm_global = (sorted_conf_global - sorted_conf_global.min()) / (sorted_conf_global.max() - sorted_conf_global.min() + 1e-8)
        conf_norm_local = (sorted_conf_local - sorted_conf_local.min()) / (sorted_conf_local.max() - sorted_conf_local.min() + 1e-8)
        colormap = cm.turbo
        colors_confidence_global = colormap(conf_norm_global)[:, :3]
        colors_confidence_local = colormap(conf_norm_local)[:, :3]

        def rainbow_color(n, total):
            import colorsys
            hue = n / total
            return colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        rainbow_color_for_frame = rainbow_color(i, num_frames)
        colors_rainbow_global = np.tile(rainbow_color_for_frame, (sorted_pts3d_global.shape[0], 1))
        colors_rainbow_local = np.tile(rainbow_color_for_frame, (sorted_pts3d_local.shape[0], 1))

        max_conf_global = conf_global.max()
        is_high_confidence = max_conf_global >= gui_global_conf_threshold.value

        c2w = poses_c2w[i]
        height, width = view['img'].shape[2], view['img'].shape[3]
        focal_length = estimated_focals[0][i]
        img_rgb_reshaped = img_rgb.reshape(height, width, 3)
        img_rgb_normalized = ((img_rgb_reshaped + 1) * 127.5).astype(np.uint8)
        img_downsampled = img_rgb_normalized[::4, ::4]

        frame_data = {
            'sorted_pts3d_global': sorted_pts3d_global,
            'colors_rgb_global': colors_rgb_global,
            'colors_confidence_global': colors_confidence_global,
            'colors_rainbow_global': colors_rainbow_global,
            'sorted_pts3d_local': sorted_pts3d_local,
            'colors_rgb_local': colors_rgb_local,
            'colors_confidence_local': colors_confidence_local,
            'colors_rainbow_local': colors_rainbow_local,
            'sorted_not_sky_global': sorted_not_sky_global,
            'sorted_not_sky_local': sorted_not_sky_local,
            'max_conf_global': float(max_conf_global),
            'is_high_confidence': is_high_confidence,
            'c2w': c2w,
            'height': height,
            'width': width,
            'focal_length': focal_length,
            'img_downsampled': img_downsampled,
            'rainbow_color': rainbow_color_for_frame,
        }
        frame_data_list.append(frame_data)

    # Percentile for scene extent calculation (10th to 90th percentile by default)
    extent_percentile = 80
    cumulative_pts_combined = np.concatenate(cumulative_pts, axis=0)
    # Calculate percentiles for each coordinate
    min_coords = np.percentile(cumulative_pts_combined, 100 - extent_percentile, axis=0)
    max_coords = np.percentile(cumulative_pts_combined, extent_percentile, axis=0)
    scene_extent = max_coords - min_coords
    max_extent = np.max(scene_extent)

    # ----------------- Create Visualization Nodes -----------------
    for i in tqdm(range(num_frames)):
        fd = frame_data_list[i]
        frame_node = server.scene.add_frame(f"/cams/t{i}", show_axes=False)

        point_node_global = server.scene.add_point_cloud(
            name=f"/pts3d_global/t{i}",
            points=fd['sorted_pts3d_global'],
            colors=fd['colors_rgb_global'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )
        point_node_local = server.scene.add_point_cloud(
            name=f"/pts3d_local/t{i}",
            points=fd['sorted_pts3d_local'],
            colors=fd['colors_rgb_local'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=True if fd['is_high_confidence'] else False,
        )

        rotation_matrix = fd['c2w'][:3, :3]
        position = fd['c2w'][:3, 3]
        rotation_quaternion = tf.SO3.from_matrix(rotation_matrix).wxyz
        fov = 2 * np.arctan2(fd['height'] / 2, fd['focal_length'])
        aspect_ratio = fd['width'] / fd['height']
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)

        frustum_node = server.scene.add_camera_frustum(
            name=f"/cams/t{i}/frustum",
            fov=fov,
            aspect=aspect_ratio,
            scale=frustum_scale,
            color=fd['rainbow_color'],
            image=fd['img_downsampled'],
            wxyz=rotation_quaternion,
            position=position,
            visible=True if fd['is_high_confidence'] else False,
        )

        fd['frame_node'] = frame_node
        fd['point_node_global'] = point_node_global
        fd['point_node_local'] = point_node_local
        fd['frustum_node'] = frustum_node

    # Initially set all nodes hidden
    for fd in frame_data_list:
        fd['frame_node'].visible = False
        fd['point_node_global'].visible = False
        fd['point_node_local'].visible = False
        fd['frustum_node'].visible = False
    server.flush()

    # Initialize timestep to show all frames and disable playing
    gui_timestep.value = num_frames - 1
    gui_playing.value = False

    # Scene type detection and sky masking initialization
    is_outdoor = is_outdoor_scene(frame_data_list)
    gui_mask_sky.value = is_outdoor

    print("\nScene type detection:")
    sky_ratios = [1.0 - np.mean(fd['sorted_not_sky_global']) for fd in frame_data_list]
    significant = sum(1 for r in sky_ratios if r > 0.2)
    print(f"- Found {significant}/{len(sky_ratios)} frames with significant sky presence (>20% sky pixels)")
    print(f"- Scene classified as: {'outdoor' if is_outdoor else 'indoor'}, setting mask_sky to {is_outdoor}")

    # Initial visibility setup
    with server.atomic():
        for i in range(num_frames):
            fd = frame_data_list[i]
            fd['frame_node'].visible = True
            fd['frustum_node'].visible = True if fd['is_high_confidence'] else False
            
            # Set up initial points with sky masking if needed
            pts3d_global = fd['sorted_pts3d_global']
            pts3d_local = fd['sorted_pts3d_local']
            
            # Select appropriate colors based on active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global']
                colors_local = fd['colors_confidence_local']
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global']
                colors_local = fd['colors_rainbow_local']
            else:
                colors_global = fd['colors_rgb_global']
                colors_local = fd['colors_rgb_local']
            
            if is_outdoor:  # Apply sky masking if outdoor scene
                mask_global = fd['sorted_not_sky_global']
                mask_local = fd['sorted_not_sky_local']
                pts3d_global = pts3d_global[mask_global > 0]
                pts3d_local = pts3d_local[mask_local > 0]
                colors_global = colors_global[mask_global > 0]
                colors_local = colors_local[mask_local > 0]
            
            # Update point clouds
            fd['point_node_global'].points = pts3d_global
            fd['point_node_local'].points = pts3d_local
            fd['point_node_global'].colors = colors_global
            fd['point_node_local'].colors = colors_local
            fd['point_node_global'].visible = gui_show_global.value
            fd['point_node_local'].visible = gui_show_local.value

    server.flush()

    # ----------------- GUI Callback Updates -----------------
    @gui_timestep.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i <= current:
                    fd['frame_node'].visible = True
                    # Set frustum visibility based on confidence settings
                    if fd['is_high_confidence']:
                        fd['frustum_node'].visible = gui_show_high_conf.value
                    else:
                        fd['frustum_node'].visible = gui_show_low_conf.value
                    fd['point_node_global'].visible = gui_show_global.value
                    fd['point_node_local'].visible = gui_show_local.value
                else:
                    fd['frame_node'].visible = False
                    fd['frustum_node'].visible = False
                    fd['point_node_global'].visible = False
                    fd['point_node_local'].visible = False
        server.flush()

    @gui_point_size.on_update
    def _(_):
        with server.atomic():
            for fd in frame_data_list:
                fd['point_node_global'].point_size = gui_point_size.value
                fd['point_node_local'].point_size = gui_point_size.value
        server.flush()

    @gui_frustum_size_percent.on_update
    def _(_):
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)
        with server.atomic():
            for fd in frame_data_list:
                fd['frustum_node'].scale = frustum_scale
        server.flush()

    @gui_show_confidence_color.on_update
    def _(_):
        # Make options mutually exclusive
        if gui_show_confidence_color.value and gui_rainbow_color_option.value:
            gui_rainbow_color_option.value = False
        
        # Update colors for all visible points
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_rainbow_color_option.on_update
    def _(_):
        # Make options mutually exclusive
        if gui_rainbow_color_option.value and gui_show_confidence_color.value:
            gui_show_confidence_color.value = False
            
        # Update colors for all visible points
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_min_conf_percentile.on_update
    def _(_):
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_mask_sky.on_update
    def _(_):
        # For each visible frame, update filtering if mask sky changes.
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_show_global.on_update
    def _(_):
        with server.atomic():
            for i in range(int(gui_timestep.value)+1):
                frame_data_list[i]['point_node_global'].visible = gui_show_global.value
        server.flush()

    @gui_show_local.on_update
    def _(_):
        with server.atomic():
            for i in range(int(gui_timestep.value)+1):
                frame_data_list[i]['point_node_local'].visible = gui_show_local.value
        server.flush()

    @gui_show_high_conf.on_update
    def _(_):
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i <= int(gui_timestep.value):
                    # Hide frustum and points if high confidence views are disabled
                    if fd['is_high_confidence'] and gui_show_high_conf.value:
                        fd['frustum_node'].visible = gui_show_high_conf.value
                        fd['point_node_global'].visible = gui_show_global.value and gui_show_high_conf.value
                        fd['point_node_local'].visible = gui_show_local.value and gui_show_high_conf.value
                    else:
                        fd['frustum_node'].visible = False  # Hide if not high confidence
                        fd['point_node_global'].visible = False  # Hide if not high confidence
                        fd['point_node_local'].visible = False  # Hide if not high confidence
        server.flush()

    @gui_show_low_conf.on_update
    def _(_):
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i <= int(gui_timestep.value):
                    # Hide frustum and points if low confidence views are disabled
                    if not fd['is_high_confidence'] and gui_show_low_conf.value:
                        fd['frustum_node'].visible = gui_show_low_conf.value
                        fd['point_node_global'].visible = gui_show_global.value and gui_show_low_conf.value
                        fd['point_node_local'].visible = gui_show_local.value and gui_show_low_conf.value
                    else:
                        fd['frustum_node'].visible = False  # Hide if high confidence
                        fd['point_node_global'].visible = False  # Hide if high confidence
                        fd['point_node_local'].visible = False  # Hide if high confidence
        server.flush()

    @gui_global_conf_threshold.on_update
    def _(_):
        for fd in frame_data_list:
            fd['is_high_confidence'] = fd['max_conf_global'] >= gui_global_conf_threshold.value
        server.flush()

    # ----------------- Start Playback Loop -----------------
    def local_playback_loop():
        while True:
            if gui_playing.value:
                gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
            time.sleep(1.0 / float(gui_framerate.value))
    playback_thread = threading.Thread(target=local_playback_loop)
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
                images.append(client.get_render(height=720, width=1280))
            gif_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps, loop=0)
            client.send_file_download("visualization.gif", gif_bytes)
            gui_timestep.value = original_timestep
            gui_playing.value = original_playing
        except Exception as e:
            print(f"Error while rendering GIF: {e}")

    public_url = server.request_share_url()
    return server

# Example usage:
# server = start_visualization(output=your_output_dict, min_conf_thr_percentile=10, global_conf_thr_value_to_drop_view=1.5, port=8020)
