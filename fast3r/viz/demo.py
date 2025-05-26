# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
⚡ Fast3R 3D Reconstruction Demo ⚡
===================================
Upload multiple unordered images of a scene, and Fast3R predicts 3D reconstructions and camera poses in one forward pass.
The images do not need to come from the same camera (e.g. iPhone, DSLR) and can be in different aspect ratios.
"""

import os
import time
import shutil
import torch
import gradio as gr
import multiprocessing as mp
from rich.console import Console
import argparse
import json

from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.utils.image import load_images
from fast3r.viz.viser_visualizer import start_visualization
from fast3r.viz.video_utils import extract_frames_from_video
from fast3r.utils.checkpoint_utils import load_model

# Add these global variables at the module level, after imports
global_manager_req_queue = None
global_manager_resp_queue = None

# -------------------------------
# run_viser_server
# -------------------------------
def run_viser_server(pipe_conn, output, min_conf_thr_percentile, global_conf_thr_value_to_drop_view, point_size=0.0004):
    """
    Launches the visualization server and sends its share URL.
    """
    try:
        server = start_visualization(
            output=output,
            min_conf_thr_percentile=min_conf_thr_percentile,
            global_conf_thr_value_to_drop_view=global_conf_thr_value_to_drop_view,
            point_size=point_size,
        )
        share_url = server.request_share_url()
        pipe_conn.send({"share_url": share_url})
        pipe_conn.close()
        while True:
            time.sleep(3600)
    except Exception as e:
        try:
            pipe_conn.send({"error": str(e)})
        except Exception:
            pass
        pipe_conn.close()


# -------------------------------
# ViserServerManager
# -------------------------------
class ViserServerManager:
    """
    Manages visualization servers launched as separate processes.
    """
    def __init__(self, req_queue, resp_queue):
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.servers = {}  # server_id -> server info
        self.session_servers = {}  # session_id -> list of server_ids
        self.console = Console()
        self.next_server_id = 1

    def run(self):
        self.console.log("[bold green]ViserServerManager started[/bold green]")
        while True:
            try:
                cmd = self.req_queue.get(timeout=1)
            except Exception:
                continue

            # Extract message_id if present
            message_id = cmd.get("message_id")
            
            if cmd["cmd"] == "launch":
                server_id = self.next_server_id
                self.next_server_id += 1
                session_id = cmd.get("session_id", f"default_session_{server_id}")
                
                self.console.log(f"Launching viser server with id {server_id} for session {session_id} (message_id: {message_id})")
                try:
                    output = cmd["output"]
                    parent_conn, child_conn = mp.Pipe()
                    p = mp.Process(
                        target=run_viser_server,
                        args=(
                            child_conn,
                            output,
                            cmd.get("min_conf_thr_percentile", 10),
                            cmd.get("global_conf_thr_value_to_drop_view", 1.5),
                            cmd.get("point_size", 0.0004),
                        )
                    )
                    p.start()
                    child_conn.close()
                    result = parent_conn.recv()
                    if "error" in result:
                        self.console.log(f"[red]Error launching server: {result['error']}[/red]")
                        self.resp_queue.put({
                            "cmd": "launch", 
                            "error": result["error"],
                            "message_id": message_id
                        })
                        p.terminate()
                        p.join(timeout=5)
                    else:
                        share_url = result["share_url"]
                        self.servers[server_id] = {"share_url": share_url, "process": p, "session_id": session_id}
                        
                        # Track which servers belong to which session
                        if session_id not in self.session_servers:
                            self.session_servers[session_id] = []
                        self.session_servers[session_id].append(server_id)
                        
                        self.console.log(f"Server {server_id} launched with URL {share_url} (pid: {p.pid}) for session {session_id}")
                        self.resp_queue.put({
                            "cmd": "launch", 
                            "server_id": server_id, 
                            "session_id": session_id,
                            "share_url": share_url,
                            "message_id": message_id
                        })
                except Exception as e:
                    self.console.log(f"[red]Error launching server: {e}[/red]")
                    self.resp_queue.put({
                        "cmd": "launch", 
                        "error": str(e),
                        "message_id": message_id
                    })

            elif cmd["cmd"] == "terminate_server":
                server_id = cmd["server_id"]
                if server_id in self.servers:
                    process = self.servers[server_id].get("process")
                    session_id = self.servers[server_id].get("session_id")
                    self.console.log(f"Terminating server with id {server_id} (pid: {process.pid if process else 'N/A'}) from session {session_id}")
                    try:
                        if process is not None:
                            process.kill()
                            process.join(timeout=10)
                        
                        # Remove from session tracking
                        if session_id in self.session_servers and server_id in self.session_servers[session_id]:
                            self.session_servers[session_id].remove(server_id)
                            if not self.session_servers[session_id]:  # If empty list
                                del self.session_servers[session_id]
                        
                        del self.servers[server_id]
                        self.resp_queue.put({
                            "cmd": "terminate_server", 
                            "server_id": server_id, 
                            "status": "terminated",
                            "message_id": message_id
                        })
                    except Exception as e:
                        self.console.log(f"[red]Error terminating server {server_id}: {e}[/red]")
                        self.resp_queue.put({
                            "cmd": "terminate_server", 
                            "server_id": server_id, 
                            "error": str(e),
                            "message_id": message_id
                        })
                else:
                    self.console.log(f"[red]Server with id {server_id} not found[/red]")
                    self.resp_queue.put({
                        "cmd": "terminate_server", 
                        "server_id": server_id, 
                        "error": "ID not found",
                        "message_id": message_id
                    })
            
            elif cmd["cmd"] == "terminate_session":
                session_id = cmd["session_id"]
                if session_id in self.session_servers:
                    server_ids = self.session_servers[session_id].copy()  # Copy to avoid modification during iteration
                    self.console.log(f"Terminating all servers for session {session_id}: {server_ids}")
                    
                    terminated_servers = []
                    errors = []
                    
                    for server_id in server_ids:
                        if server_id in self.servers:
                            process = self.servers[server_id].get("process")
                            try:
                                if process is not None:
                                    process.kill()
                                    process.join(timeout=10)
                                del self.servers[server_id]
                                terminated_servers.append(server_id)
                            except Exception as e:
                                errors.append(f"Error terminating server {server_id}: {e}")
                    
                    # Clean up session tracking
                    if session_id in self.session_servers:
                        del self.session_servers[session_id]
                    
                    self.resp_queue.put({
                        "cmd": "terminate_session", 
                        "session_id": session_id, 
                        "terminated_servers": terminated_servers,
                        "errors": errors,
                        "status": "terminated" if not errors else "partial_termination",
                        "message_id": message_id
                    })
                else:
                    self.console.log(f"[red]No servers found for session {session_id}[/red]")
                    self.resp_queue.put({
                        "cmd": "terminate_session", 
                        "session_id": session_id, 
                        "error": "Session ID not found",
                        "message_id": message_id
                    })
            else:
                self.console.log(f"Unknown command: {cmd}")
                self.resp_queue.put({
                    "cmd": "error", 
                    "error": "Unknown command",
                    "message_id": message_id
                })


# -------------------------------
# start_manager
# -------------------------------
def _manager_run(req_queue, resp_queue):
    mgr = ViserServerManager(req_queue, resp_queue)
    mgr.run()


def start_manager():
    """
    Starts the ViserServerManager.
    
    Returns: req_queue, resp_queue, manager_process.
    """
    req_queue = mp.Queue()
    resp_queue = mp.Queue()
    manager_process = mp.Process(
        target=_manager_run,
        args=(req_queue, resp_queue),
    )
    manager_process.start()
    return req_queue, resp_queue, manager_process


# -------------------------------
# update_gallery
# -------------------------------
def update_gallery(files):
    """
    Returns a list of file paths for gallery preview.
    """
    if files is None:
        return []
    preview = []
    for f in files:
        if isinstance(f, str):
            preview.append(f)
        elif isinstance(f, dict) and "data" in f:
            preview.append(f["data"])
    return preview


# -------------------------------
# process_images
# -------------------------------
def process_images(uploaded_files, video_file, state,
                   model, lit_module, device,
                   global_manager_req_queue, global_manager_resp_queue, 
                   output_dir, examples_dir,
                   image_size=512, rotate_clockwise_90=False, crop_to_landscape=False):
    """
    Processes input images/video:
      - Saves files to the output directory (unless it's an example)
      - Runs model inference.
      - Launches the visualization server.
    
    Args:
        image_size: Resolution to resize images to (224 or 512)
        rotate_clockwise_90: Whether to rotate images 90 degrees clockwise
        crop_to_landscape: Whether to crop images to landscape orientation
    
    This function yields a consistent 6-tuple on every update:
      1. loading_message: HTML
      2. vis_message: HTML
      3. feedback_column: gr.Column
      4. viser_iframe: HTML
      5. status_box: Textbox
      6. state: State
    """
    if not uploaded_files:
        yield (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "⚠️ Please upload at least one image or video, or click on an example scene. ⚠️",
            state
        )
        return
    
    start_total = time.time()

    # Create timestamp for the current session
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S-") + str(int(time.time() * 1000))
    state["current_timestamp"] = timestamp    

    # Save files to output directory (unless it's an example)
    is_example = state.get("is_example", False)
    filelist = []
    
    if not is_example and uploaded_files:
        # Create session directories
        save_dir = os.path.join(output_dir, "no_feedback", timestamp)
        img_dir = os.path.join(save_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        # Save all files
        for i, file_obj in enumerate(uploaded_files):
            src_path = file_obj[0]
            dst_path = os.path.join(img_dir, f'image_{i}.jpg')
            shutil.copy2(src_path, dst_path)
            filelist.append(dst_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'feedback_type': 'no_feedback',
            'num_images': len(filelist),
            'source_type': 'video' if video_file else 'images'
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        # For examples, just use the original file paths but still save metadata
        filelist = [file_obj[0] if isinstance(file_obj, tuple) else file_obj 
                   for file_obj in uploaded_files]
        
        # Create directory for example scene metadata
        example_dir = os.path.join(output_dir, "example_scenes", timestamp)
        os.makedirs(example_dir, exist_ok=True)
        
        # Save metadata for the example scene
        example_metadata = {
            'timestamp': timestamp,
            'num_images': len(filelist),
            'source_type': 'example',
            'example_name': os.path.basename(video_file) if video_file else None
        }
        with open(os.path.join(example_dir, 'metadata.json'), 'w') as f:
            json.dump(example_metadata, f, indent=2)
    
    # Define a constant loading message.
    loading_html = """
    <div class="loading-box">
        <div class="loading-title">🚀 Fast3R is working its magic! ✨</div>
        <div class="loading-subtitle">Preparing visualization</div>
        <div>
            <span class="loading-emoji">🎨</span>
            <span class="loading-emoji">🌟</span>
            <span class="loading-emoji">🔮</span>
        </div>
        <p>Visualizing your 3D scene, please wait...</p>
    </div>
    <style>
        .loading-subtitle::after {
            content: '';
            animation: dots 2s steps(1, end) infinite;
        }
        @keyframes dots {
            0%, 20% { content: ''; }
            40% { content: '.'; }
            60% { content: '..'; }
            80% { content: '...'; }
            100% { content: ''; }
        }
    </style>
    """
    # Yield intermediate update after saving files.
    yield (
        gr.update(value=loading_html, visible=True),
        gr.HTML(visible=False),
        gr.Column(visible=False),
        gr.HTML(visible=False),
        f"Processing {len(filelist)} images...\nLoading and cropping images...",
        state,
    )
    
    end_image = time.time()
    image_prep_time = end_image - start_total
    
    # Load and resize images
    start_load_time = time.time()
    
    imgs = load_images(
        filelist,
        size=image_size,
        verbose=True,
        rotate_clockwise_90=rotate_clockwise_90,
        crop_to_landscape=crop_to_landscape,
    )
    end_load_time = time.time()
    load_time = end_load_time - start_load_time
    print(f"Image loading and cropping time: {load_time:.2f} seconds")
    
    yield (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        f"Processing {len(filelist)} images...\nImage loading and cropping time: {load_time:.2f} sec.\nRunning model inference...",
        state,
    )
    
    # Run inference directly
    output_dict, profiling_info = inference(
        imgs,
        model,
        device,
        dtype=torch.float32,
        # dtype=torch.bfloat16,
        verbose=True,
        profiling=True,
    )
    model_forward_time = profiling_info['total_time']

    yield (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        f"Processing {len(filelist)} images...\nImage loading and cropping time: {load_time:.2f} sec.\nModel inference time: {model_forward_time:.2f} sec.\nPreparing visualization...",
        state,
    )
    
    # Process predictions and move tensors to CPU.
    try:
        for pred in output_dict['preds']:
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    pred[k] = v.cpu()
        for view in output_dict['views']:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.cpu()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: {e}")
    
    # Align points.
    start_vis_prep = time.time()
    lit_module.align_local_pts3d_to_global(
        preds=output_dict['preds'],
        views=output_dict['views'],
        min_conf_thr_percentile=85
    )
    
    # Generate a unique message ID for this request
    message_id = f"msg_{int(time.time()*1000)}_{os.getpid()}_{id(state)}"
    
    # Launch the visualization server with message_id
    session_id = state.get("session_id")
    if not session_id:  # if no session_id, create a new one
        session_id = f"session_{int(time.time()*1000)}"
        state["session_id"] = session_id
    
    cmd = {
        "cmd": "launch",
        "output": output_dict,
        "min_conf_thr_percentile": 65 if video_file and os.path.basename(video_file) == "family.mp4" else 10,
        "point_size": 0.0001 if video_file and os.path.basename(video_file) == "family.mp4" else 0.0004,
        "global_conf_thr_value_to_drop_view": 1.5,
        "session_id": session_id,
        "message_id": message_id
    }
    global_manager_req_queue.put(cmd)
    
    # Wait for server response with dynamic loading, but only accept responses with matching message_id
    loading_dots = [".", "..", "..."]
    loading_idx = 0
    start_wait = time.time()
    timeout = 600
    while True:
        # Check for timeout first
        if time.time() - start_wait > timeout:
            raise gr.Error("Timeout waiting for visualization server. Please try again.")
        
        try:
            resp = global_manager_resp_queue.get_nowait()
            # Only accept responses with matching message_id
            if resp.get("message_id") == message_id:
                break
            else:
                # Put back responses meant for other sessions
                global_manager_resp_queue.put(resp)
                time.sleep(0.1)  # Small delay to avoid CPU spinning
        except:
            # Update loading animation
            dots = loading_dots[loading_idx]
            loading_idx = (loading_idx + 1) % len(loading_dots)
            status_text = f"Processing {len(filelist)} images...\nImage loading and cropping time: {load_time:.2f} sec.\nModel inference time: {model_forward_time:.2f} sec.\nPreparing visualization{dots}"
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                status_text,
                state
            )
            time.sleep(0.3)
    
    if "error" in resp:
        share_url = f"ERROR: {resp['error']}"
    else:
        share_url = resp["share_url"]
    
    end_vis_prep = time.time()
    vis_prep_time = end_vis_prep - start_vis_prep
    total_time = end_vis_prep - start_total
    
    # Store the server_id from the response if available
    server_id = resp.get("server_id")
    state["urls"].append((share_url, server_id))

    final_status = (
        f"{len(filelist)} images @ {image_size} resolution\n"
        f"Image loading and cropping time: {load_time:.2f} sec\n"
        f"Model inference time: {model_forward_time:.2f} sec\n"
        f"Visualization preparation time: {vis_prep_time:.2f} sec\n"
        f"👇 Scroll down to view the 3D reconstruction"
    )
    
    # Build final visualization header and feedback prompt.
    final_vis_header = """
    <style>
        .vis-box {{
            background: linear-gradient(145deg, #f0f0f0, #ffffff);
            color: #333;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        @media (prefers-color-scheme: dark) {{
            .vis-box {{
                background: linear-gradient(145deg, #2a2a2a, #383838);
                color: #fff;
            }}
        }}
        .vis-title {{
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .vis-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .control-list {{
            background: rgba(0,0,0,0.05);
            padding: 12px 15px;
            border-radius: 8px;
        }}
        .control-list div {{
            margin: 4px 0;
        }}
        .url-list {{
            background: rgba(0,0,0,0.05);
            padding: 12px 15px;
            border-radius: 8px;
        }}
        .url-list p {{
            margin: 0 0 8px 0;
        }}
        .url-list ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .url-list a {{
            color: #2196F3;
            text-decoration: none;
        }}
        .url-list a:hover {{
            text-decoration: underline;
        }}
        @media (max-width: 768px) {{
            .vis-content {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    <div class="vis-box">
        <div class="vis-title">
            <span>🎨</span> Visualization Ready! <span>✨</span>
        </div>
        <div class="vis-content">
            <div class="control-list">
                <div>🖱️ Left click + drag to rotate</div>
                <div>👆 Right click + drag to pan</div>
                <div>⚙️ Scroll to zoom in/out</div>
                <div>🎛️ Click "Show Controls" to minimize control panel</div>
                <div>🔍 Adjust "Per-View Conf Percentile" under "Confidence Options" to reduce "floater" points</div>
            </div>
            <div class="url-list">
                <p>🔗 Share these URLs to view the visualization in any browser:</p>
                <ul>
                    {url_list}
                </ul>
            </div>
        </div>
    </div>
    """

    # Generate URL list HTML
    url_list_html = ""
    for i, (url, server_id) in enumerate(state["urls"]):
        if i == len(state["urls"]) - 1:
            url_list_html += f'<li style="margin: 4px 0;"><span>🆕</span> <a href="{url}" target="_blank">{url}</a> (latest)</li>'
        else:
            url_list_html += f'<li style="margin: 4px 0;"><a href="{url}" target="_blank">{url}</a></li>'

    # Format the final HTML with the URL list
    final_vis_header = final_vis_header.format(url_list=url_list_html)

    viser_iframe_html = (
        f"<iframe src='{share_url}' width='100%' height='800' frameborder='0' "
        "style='border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'></iframe>"
    )

    yield (
        gr.update(visible=False),                     # loading_html is now empty
        gr.update(value=final_vis_header, visible=True),       # vis_header_html: detailed visualization header
        gr.update(visible=True),         # feedback_column: now visible
        gr.update(value=viser_iframe_html, visible=True),      # iframe_html: the 3D viewer
        final_status,               # status_text: final status text
        state                           # updated state
    )


# -------------------------------
# delete_visers_callback
# -------------------------------
def delete_visers_callback(state):
    """
    Cleans up visualization servers when the session ends.
    """
    session_id = state.get("session_id")
    if session_id:
        try:
            # Generate a unique message ID for this request
            message_id = f"term_{int(time.time()*1000)}_{os.getpid()}_{id(state)}"
            
            # Terminate all servers for this session
            term_cmd = {
                "cmd": "terminate_session", 
                "session_id": session_id,
                "message_id": message_id
            }
            global_manager_req_queue.put(term_cmd)
            
            # Wait for response with matching message_id
            start_wait = time.time()
            timeout = 600
            while True:
                # Check for timeout first
                if time.time() - start_wait > timeout:
                    print(f"Timeout waiting for termination response for session {session_id}")
                    break
                
                try:
                    resp = global_manager_resp_queue.get(timeout=1)
                    if resp.get("message_id") == message_id:
                        print(f"Terminated servers for session {session_id}, Response: {resp}")
                        break
                    else:
                        # Put back responses meant for other sessions
                        global_manager_resp_queue.put(resp)
                except Exception:
                    # Just continue the loop if queue is empty
                    continue
            
        except Exception as e:
            print(f"Error terminating servers for session {session_id}: {e}")
    print(f"All viser servers for session {session_id} cleaned up.")

# -------------------------------
# handle_feedback
# -------------------------------
def handle_feedback(feedback_type, timestamp, output_dir):
    """
    Move data from no_feedback to the appropriate feedback folder.
    """
    print(f"Feedback type: {feedback_type} received for session: {timestamp}.")
    src_dir = os.path.join(output_dir, "no_feedback", timestamp)
    dst_dir = os.path.join(output_dir, feedback_type, timestamp)
    
    # Check if this is a regular upload (not an example)
    if os.path.exists(src_dir):
        # Move the directory to the appropriate feedback folder
        shutil.move(src_dir, dst_dir)
        
        # Update the metadata file
        metadata_path = os.path.join(dst_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['feedback_type'] = feedback_type
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        # print that the feedback was saved
        print(f"Feedback saved to {dst_dir}")
        return True
    
    # Check if this is an example scene
    example_dir = os.path.join(output_dir, "example_scenes", timestamp)
    if os.path.exists(example_dir):
        # For example scenes, just update the metadata without moving
        metadata_path = os.path.join(example_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['feedback_type'] = feedback_type
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        print(f"Feedback saved for example scene: {example_dir}")
        return True
        
    return False


# -------------------------------
# create_demo
# -------------------------------
def create_demo(checkpoint_dir, examples_dir, output_dir, device: torch.device, is_lightning_checkpoint=False):
    """
    Creates the Gradio demo interface.
    
    Layout:
      - Header with instructions.
      - Input row: Gallery and video input.
      - Output components (6 outputs):
            1. loading_message (HTML)
            2. vis_message (HTML)
            3. feedback_column (gr.Column) – contains the feedback prompt and buttons.
            4. viser_iframe (HTML)
            5. status_box (Textbox)
            6. state (State)
    """
    global global_manager_req_queue, global_manager_resp_queue
    
    global_manager_req_queue, global_manager_resp_queue, manager_process = start_manager()

    model, lit_module = load_model(checkpoint_dir, device=device, is_lightning_checkpoint=is_lightning_checkpoint)

    # Load examples
    # examples = []
    # example_filenames = set()  # Track example filenames
    # video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # if os.path.exists(examples_dir):
    #     print(f"Loading examples from: {examples_dir}")
    #     # Iterate through subdirectories
    #     for example_name in sorted(os.listdir(examples_dir)):
    #         example_path = os.path.join(examples_dir, example_name)
    #         if os.path.isdir(example_path):
    #             # Look for videos in this subdirectory
    #             for ext in video_extensions:
    #                 pattern = os.path.join(example_path, f"*{ext}")
    #                 found_videos = glob.glob(pattern)
    #                 if found_videos:
    #                     video_path = found_videos[0]  # Take the first video file
    #                     print(f"Adding example: {video_path}")
    #                     examples.append([None, video_path])
    #                     example_filenames.add(os.path.basename(video_path))
    #                     break  # Only take one video per subdirectory
    
    # if not examples:
    #     print("Warning: No example videos found in the examples directory!")
    # else:
    #     print(f"Successfully loaded {len(examples)} examples")

    
    examples = [
        "https://fast3r-3d.github.io/demo_examples/teddybear.mp4",
        "https://fast3r-3d.github.io/demo_examples/kitchen.mp4",
        "https://fast3r-3d.github.io/demo_examples/redkitchen.mp4",
        "https://fast3r-3d.github.io/demo_examples/family.mp4",
        "https://fast3r-3d.github.io/demo_examples/lighthouse.mp4",
    ]
    example_filenames = [os.path.basename(example) for example in examples]

    custom_css = """
    <style>
        .thumbs-up { background-color: #90ee90 !important; }
        .thumbs-down { background-color: red !important; color: white !important; }
        .feedback-button {
            border: none !important;
            background: rgba(33, 150, 243, 0.1) !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            transition: all 0.3s ease !important;
            margin: 0 5px !important;
        }
        .feedback-button:hover {
            transform: translateY(-2px) !important;
            background: rgba(33, 150, 243, 0.2) !important;
        }
        .feedback-button.positive:hover {
            background: rgba(76, 175, 80, 0.2) !important;
        }
        .feedback-button.negative:hover {
            background: rgba(244, 67, 54, 0.2) !important;
        }
        .floating-box {
            background: linear-gradient(145deg, #f0f0f0, #ffffff);
            color: #333;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            padding: 10px;
            border-radius: 12px;
            margin: 5px 0;
            animation: float 3s ease-in-out infinite;
        }
        @media (prefers-color-scheme: dark) {
            .floating-box {
                background: linear-gradient(145deg, #2a2a2a, #383838);
                color: #fff;
            }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
    </style>
    """
    video_input = gr.Video(label="Or Upload a Video", height="270px")
    with gr.Blocks(title="Fast3R 3D Reconstruction Demo", ) as demo:
        header_html = f"""
        <style>
            .header-box {{
                background: linear-gradient(145deg, #f0f0f0, #ffffff);
                color: #333;
                box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            @media (prefers-color-scheme: dark) {{
                .header-box {{
                    background: linear-gradient(145deg, #2a2a2a, #383838);
                    color: #fff;
                }}
            }}
            .header-title {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 15px;
            }}
            .header-links {{
                display: flex;
                gap: 15px;
                margin-bottom: 10px;
            }}
            .header-links a {{
                color: #2196F3;
                text-decoration: none;
            }}
            .header-links a:hover {{
                text-decoration: underline;
            }}
            .header-content {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 15px;
            }}
            @media (min-width: 768px) {{
                .header-content {{
                    grid-template-columns: 1fr 1fr;
                }}
            }}
        </style>
        <style>
            .loading-box {{
                background: linear-gradient(145deg, #f0f0f0, #ffffff);
                color: #333;
                box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
                padding: 15px;
                border-radius: 12px;
                margin: 20px auto;
                text-align: center;
                max-width: 600px;
                animation: float 3s ease-in-out infinite;
            }}
            @media (prefers-color-scheme: dark) {{
                .loading-box {{
                    background: linear-gradient(145deg, #2a2a2a, #383838);
                    color: #fff;
                }}
            }}
            .loading-title {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 15px;
                animation: bounce 2s ease-in-out infinite;
            }}
            .loading-subtitle {{
                font-size: 18px;
                margin: 10px 0;
                animation: bounce 2s ease-in-out infinite;
                animation-delay: 0.3s;
            }}
            .loading-emoji {{
                font-size: 24px;
                margin: 15px 0;
                display: inline-block;
                animation: bounce 1s infinite;
            }}
            .loading-emoji:nth-child(2) {{
                animation-delay: 0.2s;
            }}
            .loading-emoji:nth-child(3) {{
                animation-delay: 0.4s;
            }}
            @keyframes float {{
                0%, 100% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-10px); }}
            }}
            @keyframes bounce {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-8px); }}
            }}
        </style>
        {custom_css}
        <div class="header-box">
            <div class="header-title">⚡ Fast3R 3D Reconstruction Demo ⚡</div>
            <div class="header-links">
                <a href="https://fast3r-3d.github.io/">Website</a> |
                <a href="https://arxiv.org/abs/2501.13928">Paper</a> |
                <a href="https://github.com/facebookresearch/fast3r">Code</a>
            </div>
            <div class="header-content">
                <div>
                    Upload unordered images of a scene, and Fast3R predicts 3D reconstructions 
                    and camera poses in one forward pass. Works with mixed camera types 
                    (e.g., iPhone + DSLR) and aspect ratios.
                </div>
                <div>
                    <b>How to use:</b><br>
                    3D from images: Select and upload multiple images of a scene (Ctrl/Shift + click to select multiple)<br>
                    3D from video: Upload a video (auto-samples at 1 FPS)<br>
                    • Click 'Submit' to start reconstruction
                </div>
            </div>
        </div>
        """
        state = gr.State(value={"session_id": "", "urls": []}, delete_callback=delete_visers_callback)

        gr.HTML(header_html)
        with gr.Row():
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Upload Images of a Scene", columns=6, height="150px", show_download_button=True)
                video_input.render()
            with gr.Column(scale=1):
                # Add resolution radio button
                image_resolution = gr.Radio(
                    ["512", "224"], 
                    label="Image Resolution", 
                    value="512", 
                    info="Lower resolution (224) gives super fast speed with a small trade-off in quality"
                )
                submit_button = gr.Button("Submit", variant="primary", size="lg")
                status_box = gr.Textbox(label="Processing Speed", interactive=False, lines=5, visible=True)
                gr.Examples(
                    examples=examples,
                    inputs=video_input,
                    label="Example Scenes",
                    examples_per_page=6
                )

        # Define 7 output components.
        loading_message = gr.HTML("")          # loading animation (full row during loading)
        with gr.Row():
            with gr.Column(scale=2):
                vis_message = gr.HTML(visible=False)  # For visualization ready message
            with gr.Column(scale=1):
                with gr.Column(visible=False, elem_classes="floating-box") as feedback_column:
                    gr.HTML(
                        """
                        <div class="feedback-container" style="text-align: center;">
                            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">
                                <span>✨</span> Your Opinion Matters! <span>✨</span>
                            </div>
                            <div style="font-size: 14px; line-height: 1.3; margin-bottom: 5px;">
                                We're on a mission to make Fast3R even more amazing!<br> 
                                How did the reconstruction turn out?
                            </div>
                        </div>
                        """
                    )
                    thank_you = gr.HTML(
                        """
                        <div style="text-align: center; margin: 2px 0; color: #4CAF50; font-weight: bold;">
                            💝 Thank you for helping us improve!
                        </div>
                        """,
                        visible=False
                    )
                    with gr.Row():
                        thumbs_up = gr.Button("👍 Love it!", elem_classes=["feedback-button", "positive"])
                        thumbs_down = gr.Button("👎 Not quite there", elem_classes=["feedback-button", "negative"])

        viser_iframe = gr.HTML("")             # 3D viewer iframe (full row final)
        just_uploaded_video = gr.State(value=True)

        def update_gallery_upload(gallery_images, video, just_uploaded_video):
            if gallery_images:
                if video and just_uploaded_video:
                    return video, False
                else:
                    return None, just_uploaded_video
            else:
                return None, just_uploaded_video

        def update_video_upload(gallery_images, video, just_uploaded_video, state):
            if video:
                # Check if this is an example video by comparing basename
                video_filename = os.path.basename(video)
                if isinstance(video, str) and video_filename in example_filenames:
                    state["is_example"] = True
                else:
                    state["is_example"] = False
                
                # Always extract frames to a temporary directory
                temp_dir = os.path.join("temp_preview_frames", f"preview_{int(time.time()*1000)}")
                os.makedirs(temp_dir, exist_ok=True)
                frame_paths = extract_frames_from_video(video, temp_dir)
                
                return frame_paths, True, state
            else:
                state["is_example"] = False
                return [], False, state


        gallery.change(fn=update_gallery_upload, inputs=[gallery, video_input, just_uploaded_video],
                       outputs=[video_input, just_uploaded_video])
        video_input.change(fn=update_video_upload, 
                          inputs=[gallery, video_input, just_uploaded_video, state],
                          outputs=[gallery, just_uploaded_video, state])

        def handle_feedback_click(feedback_type, state):
            """Handle feedback button clicks"""
            if "current_timestamp" in state:
                timestamp = state["current_timestamp"]
                success = handle_feedback(feedback_type, timestamp, output_dir)
                if success:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True)
                    )
            raise gr.Error("Failed to handle feedback")

        def process_images_wrapper(uploaded_files, video_file, state, image_resolution):
            # Reset is_example flag for direct uploads
            if not state.get("is_example", False):
                state = state.copy()
                state["is_example"] = False
            
            # Convert image_resolution from string to integer
            image_size = int(image_resolution)

            generator = process_images(uploaded_files, video_file, state,
                                     model, lit_module, device,
                                     global_manager_req_queue, global_manager_resp_queue, 
                                     output_dir, examples_dir,
                                     image_size=image_size)
            
            for output in generator:
                loading_message, vis_message, feedback_column, viser_iframe, status_box, state = output
                yield (loading_message, vis_message, feedback_column, viser_iframe, status_box, state,
                       gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))  # make feedback buttons visible and thank you message invisible

        # Register feedback button clicks.
        thumbs_up.click(
            fn=lambda state: handle_feedback_click("good", state),
            inputs=[state],
            outputs=[thumbs_up, thumbs_down, thank_you]
        )
        thumbs_down.click(
            fn=lambda state: handle_feedback_click("bad", state),
            inputs=[state],
            outputs=[thumbs_up, thumbs_down, thank_you]
        )

        submit_button.click(
            fn=process_images_wrapper,
            inputs=[gallery, video_input, state, image_resolution],
            outputs=[loading_message, vis_message, feedback_column, viser_iframe, status_box, state, thumbs_up, thumbs_down, thank_you],
        )

    return demo

def main():
    os.environ["GRADIO_TEMP_DIR"] = "./gradio_tmp_dir"
    parser = argparse.ArgumentParser(description="Fast3R 3D Reconstruction Demo...")
    parser.add_argument("--checkpoint_dir", type=str, default="jedyang97/Fast3R_ViT_Large_512")
    parser.add_argument("--examples_dir", type=str, default="./demo_examples")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs",
                        help="Directory to store processed scenes with feedback")
    parser.add_argument("--is_lightning_checkpoint", action="store_true", default=False,
                        help="Whether the checkpoint is from Lightning training (default: False)")
    args = parser.parse_args()

    for folder in ['good', 'bad', 'no_feedback', 'example_scenes']:
        os.makedirs(os.path.join(args.output_dir, folder), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo = create_demo(args.checkpoint_dir, args.examples_dir, args.output_dir, device=device, 
                       is_lightning_checkpoint=args.is_lightning_checkpoint)
    demo.queue(default_concurrency_limit=2)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
