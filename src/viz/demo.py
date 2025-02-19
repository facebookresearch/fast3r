#!/usr/bin/env python
"""
⚡ Fast3R 3D Reconstruction Demo ⚡
===================================
Upload multiple unordered images of a scene, and Fast3R predicts the 3D reconstructions and camera poses in one forward pass. 
The images do not need to come from the same camera (e.g., one iPhone, one DSLR) and can be in different aspect ratios (e.g., landscape and portrait).
"""

import os
import sys
import time
import shutil
import socket
import struct
import pickle
import torch
import numpy as np
import trimesh
import hydra
from omegaconf import OmegaConf
import open3d as o3d
import rootutils
import gradio as gr
import multiprocessing as mp
from multiprocessing.connection import Listener, Client
from rich.console import Console
from functools import partial
import contextlib
import argparse

# Set up project root so that relative imports work correctly.
rootutils.setup_root(__file__ + "/..", indicator=".project-root", pythonpath=True)

from src.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from src.dust3r.inference_multiview import inference
from src.dust3r.utils.image import load_images, rgb
from src.viz.viser_visualizer import start_visualization


# -------------------------------
# Helper function: Run Viser Server in a Separate Process
# -------------------------------
def run_viser_server(output, min_conf_thr_percentile, global_conf_thr_value_to_drop_view, pipe_conn):
    """
    This function is run in a separate process. It starts the visualization server,
    obtains its share URL, sends it back via a pipe, and then remains alive.
    """
    try:
        server = start_visualization(
            output=output,
            min_conf_thr_percentile=min_conf_thr_percentile,
            global_conf_thr_value_to_drop_view=global_conf_thr_value_to_drop_view,
        )
        share_url = server.request_share_url()
        # Send the share_url back to the parent process.
        pipe_conn.send({"share_url": share_url})
        pipe_conn.close()
        # Keep the process alive so that the server remains accessible.
        while True:
            time.sleep(3600)
    except Exception as e:
        try:
            pipe_conn.send({"error": str(e)})
        except Exception:
            pass
        pipe_conn.close()


# -------------------------------
# ViserServerManager Class
# -------------------------------
class ViserServerManager:
    def __init__(self, req_queue, resp_queue):
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        # servers maps server_id to a dict with keys: "process" and "share_url"
        self.servers = {}
        self.console = Console()
        self.next_id = 1

    def run(self):
        self.console.log("[bold green]ViserServerManager started[/bold green]")
        while True:
            try:
                cmd = self.req_queue.get(timeout=1)
            except Exception:
                continue

            if cmd["cmd"] == "launch":
                sid = self.next_id
                self.next_id += 1
                self.console.log(f"Launching viser server with id {sid}")
                try:
                    # Retrieve the inference output directly from the command.
                    output = cmd["output"]

                    # Create a Pipe to receive the share URL from the new process.
                    parent_conn, child_conn = mp.Pipe()
                    # Launch viser server in a new process.
                    p = mp.Process(
                        target=run_viser_server,
                        args=(
                            output,
                            cmd.get("min_conf_thr_percentile", 10),
                            cmd.get("global_conf_thr_value_to_drop_view", 1.5),
                            child_conn,
                        )
                    )
                    p.start()
                    child_conn.close()  # Close child end in the manager process.

                    # Wait for the share URL from the child process.
                    result = parent_conn.recv()
                    if "error" in result:
                        self.console.log(f"[red]Error launching server: {result['error']}[/red]")
                        self.resp_queue.put({"cmd": "launch", "error": result["error"]})
                        p.terminate()
                        p.join(timeout=5)
                    else:
                        share_url = result["share_url"]
                        self.servers[sid] = {"share_url": share_url, "process": p}
                        self.console.log(f"Server {sid} launched with URL {share_url} (pid: {p.pid})")
                        self.resp_queue.put({"cmd": "launch", "id": sid, "share_url": share_url})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.console.log(f"[red]Error launching server: {e}[/red]")
                    self.resp_queue.put({"cmd": "launch", "error": str(e)})

            elif cmd["cmd"] == "terminate":
                sid = cmd["id"]
                if sid in self.servers:
                    server_info = self.servers[sid]
                    process = server_info.get("process")
                    self.console.log(f"Terminating server with id {sid} (pid: {process.pid if process else 'N/A'})")
                    try:
                        if process is not None:
                            process.kill()
                            process.join(timeout=10)
                        del self.servers[sid]
                        self.resp_queue.put({"cmd": "terminate", "id": sid, "status": "terminated"})
                    except Exception as e:
                        self.console.log(f"[red]Error terminating server {sid}: {e}[/red]")
                        self.resp_queue.put({"cmd": "terminate", "id": sid, "error": str(e)})
                else:
                    self.resp_queue.put({"cmd": "terminate", "id": sid, "error": "ID not found"})
            else:
                self.console.log(f"Unknown command: {cmd}")
                self.resp_queue.put({"cmd": "error", "error": "Unknown command"})


# -------------------------------
# Manager Startup Function
# -------------------------------
def start_manager():
    req_queue = mp.Queue()
    resp_queue = mp.Queue()
    manager_process = mp.Process(target=ViserServerManager(req_queue, resp_queue).run)
    manager_process.start()
    return req_queue, resp_queue, manager_process


# -------------------------------
# Inference Pipeline Function
# -------------------------------
def get_reconstructed_scene(model, device, silent, image_size, filelist,
                            profiling=False, dtype=torch.float32,
                            rotate_clockwise_90_for_hyperscape=False,
                            crop_to_landscape_for_hyperscape=False):
    imgs = load_images(
        filelist,
        size=image_size,
        verbose=not silent,
        rotate_clockwise_90_for_hyperscape=rotate_clockwise_90_for_hyperscape,
        crop_to_landscape_for_hyperscape=crop_to_landscape_for_hyperscape,
    )
    start_time = time.time()
    output = inference(
        imgs,
        model,
        device,
        dtype=dtype,
        verbose=not silent,
        profiling=profiling,
    )
    end_time = time.time()
    print(f"Inference time elapsed: {end_time - start_time:.2f} seconds")
    return output


# -------------------------------
# Model Loading Function
# -------------------------------
def load_model(checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    def replace_dust3r_in_config(cfg_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                replace_dust3r_in_config(value)
            elif isinstance(value, str) and "dust3r." in value and "src.dust3r." not in value:
                cfg_dict[key] = value.replace("dust3r.", "src.dust3r.")
        return cfg_dict

    cfg.model.net = replace_dust3r_in_config(cfg.model.net)
    if "encoder_args" in cfg.model.net:
        cfg.model.net.encoder_args.patch_embed_cls = "PatchEmbedDust3R"
        cfg.model.net.head_args.landscape_only = False
    else:
        cfg.model.net.patch_embed_cls = "PatchEmbedDust3R"
        cfg.model.net.landscape_only = False

    cfg.model.net.decoder_args.random_image_idx_embedding = True
    cfg.model.net.decoder_args.attn_bias_for_inference_enabled = False
    lit_module = hydra.utils.instantiate(cfg.model, train_criterion=None, validation_criterion=None)
    ckpt_last_path = os.path.join(checkpoint_dir, "checkpoints", "last.ckpt")
    if os.path.isdir(ckpt_last_path):
        CKPT_PATH = os.path.join(checkpoint_dir, "checkpoints", "last_aggregated.ckpt")
        if not os.path.exists(CKPT_PATH):
            from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
            convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_dir=ckpt_last_path,
                output_file=CKPT_PATH, tag=None
            )
    else:
        CKPT_PATH = os.path.join(checkpoint_dir, "checkpoints", "last.ckpt")
    lit_module = MultiViewDUSt3RLitModule.load_from_checkpoint(
        checkpoint_path=CKPT_PATH,
        net=lit_module.net,
        train_criterion=lit_module.train_criterion,
        validation_criterion=lit_module.validation_criterion,
    )
    lit_module.eval()
    model = lit_module.net.to(device)
    return model, lit_module, device


# -------------------------------
# Gallery Update Utility
# -------------------------------
def update_gallery(files):
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
# Main Processing Function
# -------------------------------
def process_images(uploaded_files, state, request, model, lit_module, device, manager_req_queue, manager_resp_queue):
    """
    Processes uploaded images:
      - Saves images.
      - Runs inference and aligns local points.
      - Sends the inference output via a pipe.
      - Sends a "launch" command to the ViserServerManager via the request queue.
      - Waits for a response from the manager (which contains the share URL and a server id).
      - Updates the session state with the new server info.
    Returns:
      - An HTML snippet with a clickable URL message listing all active server URLs (latest highlighted) and an embedded iframe.
      - A status message with processing times.
      - The updated session state (a dict with keys "session_id" and "urls").
    """
    if not uploaded_files:
        return (
            "<div style='color: red;'>Error: Please upload at least one image.</div>",
            "Error: No images uploaded.",
            state
        )

    start_total = time.time()

    # Save images.
    temp_dir = "temp_uploaded_images"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    filelist = []
    for i, file_obj in enumerate(uploaded_files):
        if isinstance(file_obj, str):
            file_path = file_obj
        else:
            file_path = os.path.join(temp_dir, f"image_{i}.jpg")
            with open(file_path, "wb") as f:
                f.write(file_obj.read())
        filelist.append(file_path)
    end_image = time.time()
    image_prep_time = end_image - start_total

    # Run inference and set up the output.
    start_inference = time.time()
    output = get_reconstructed_scene(model, device, silent=False, image_size=512, filelist=filelist,
                                     profiling=True, dtype=torch.float32)
    end_inference = time.time()
    model_forward_time = end_inference - start_inference

    # Align local points.
    lit_module.align_local_pts3d_to_global(
        preds=output['preds'],
        views=output['views'],
        min_conf_thr_percentile=85
    )

    # Send "launch" command to the manager with the output via the pipe.
    cmd = {
        "cmd": "launch",
        "output": output,
        "min_conf_thr_percentile": 10,
        "global_conf_thr_value_to_drop_view": 1.5
    }
    manager_req_queue.put(cmd)
    # Wait for response.
    try:
        resp = manager_resp_queue.get(timeout=30)
        if "error" in resp:
            share_url = f"ERROR: {resp['error']}"
            server_id = None
        else:
            share_url = resp["share_url"]
            server_id = resp["id"]
    except Exception as e:
        share_url = f"ERROR: {str(e)}"
        server_id = None

    total_time = time.time() - start_total
    vis_prep_time = total_time - (image_prep_time + model_forward_time)

    # Update session state.
    session_id = state.get("session_id", "")
    if not session_id:
        session_id = request.session_hash if hasattr(request, "session_hash") else "session_" + str(time.time())
        state["session_id"] = session_id
        state["urls"] = []
    state["urls"].append((share_url, server_id))
    updated_state = state

    # Generate HTML for visualization area.
    html_all = f"""
    <style>
      @media (prefers-color-scheme: dark) {{
        .vis-box {{
          background-color: #333;
          color: #fff;
          border: 1px solid #888;
        }}
      }}
      @media (prefers-color-scheme: light) {{
        .vis-box {{
          background-color: #f7f7f7;
          color: #000;
          border: 1px solid #ccc;
        }}
      }}
    </style>
    <div class="vis-box" style="padding: 10px; border-radius: 5px; margin-bottom: 20px;">
      <strong>Visualization</strong><br>
      Open or share these URLs to view the visualization in any browser:
      <ul>
    """
    for i, (url, _) in enumerate(updated_state["urls"]):
        if i == len(updated_state["urls"]) - 1:
            html_all += f"<li style='color: red; font-weight: bold;'><a href='{url}' target='_blank'>{url}</a> (latest)</li>"
        else:
            html_all += f"<li><a href='{url}' target='_blank'>{url}</a></li>"
    html_all += "</ul></div>"
    iframe_html = f'<iframe src="{share_url}" width="100%" height="600" frameborder="0"></iframe>'
    html_output = html_all + iframe_html

    status = (
        f"Image preparation time: {image_prep_time:.2f} sec\n"
        f"Model inference time: {model_forward_time:.2f} sec\n"
        f"Visualization preparation time: {vis_prep_time:.2f} sec\n"
        f"Total processing time: {total_time:.2f} sec"
    )
    
    return html_output, status, updated_state


# -------------------------------
# Delete Callback for Session State
# -------------------------------
def delete_visers_callback(state):
    session_id = state.get("session_id")
    if session_id and "urls" in state:
        for url, sid in state["urls"]:
            try:
                # Send a termination command to the manager.
                term_cmd = {"cmd": "terminate", "id": sid}
                global_manager_req_queue.put(term_cmd)
                resp = global_manager_resp_queue.get(timeout=60)
                print(f"Terminated server with URL: {url}, Response: {resp}")
            except Exception as e:
                print(f"Error terminating server with URL {url}: {e}")
    print("All viser servers for session cleaned up.")


# -------------------------------
# Main Function and Gradio Interface
# -------------------------------
def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Fast3R 3D Reconstruction Demo. Upload multiple unordered images of a scene, and Fast3R predicts the 3D reconstructions and camera poses in one forward pass. The images do not need to come from the same camera (e.g., one iPhone, one DSLR) and can be in different aspect ratios (e.g., landscape and portrait).")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory for Fast3R")
    args = parser.parse_args()

    # Start the manager.
    global global_manager_req_queue, global_manager_resp_queue
    global_manager_req_queue, global_manager_resp_queue, _ = start_manager()

    # Load the model using the provided checkpoint_dir.
    model, lit_module, device = load_model(args.checkpoint_dir)

    # Define a simple wrapper that takes the two Gradio inputs plus the request.
    def process_images_wrapper(uploaded_files, state, request):
        return process_images(uploaded_files, state, request,
                              model, lit_module, device,
                              global_manager_req_queue, global_manager_resp_queue)

    # Build the Gradio interface.
    with gr.Blocks() as demo:
        gr.Markdown("## ⚡ Fast3R 3D Reconstruction Demo ⚡")
        gr.Markdown(
            "[Website](https://fast3r-3d.github.io/) | [Paper](https://arxiv.org/abs/2501.13928) | [Code](https://github.com/facebookresearch/fast3r)\n\n"
            "Upload multiple unordered images of a scene, and Fast3R predicts the 3D reconstructions and camera poses in one forward pass. "
            "The images do not need to come from the same camera (e.g., one iPhone, one DSLR) and can be in different aspect ratios (e.g., landscape and portrait)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(file_count="multiple", label="Upload Your Images")
                gallery = gr.Gallery(label="Preview", columns=4, height="250px")
                submit_button = gr.Button("Submit", variant="primary")
                status_box = gr.Textbox(label="Processing Status", interactive=False, lines=4)
            with gr.Column(scale=2):
                output_html = gr.HTML(label="Visualization")

        init_state = {"session_id": "", "urls": []}
        state = gr.State(value=init_state, delete_callback=delete_visers_callback)

        file_input.change(fn=update_gallery, inputs=file_input, outputs=gallery)
        # The wrapper now accepts three arguments: uploaded_files, state, and the request.
        submit_button.click(fn=process_images_wrapper, inputs=[file_input, state],
                            outputs=[output_html, status_box, state])

    demo.launch(share=True)


if __name__ == "__main__":
    main()
