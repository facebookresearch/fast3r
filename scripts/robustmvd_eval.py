# fast3r_rmvd_wrapper.py

import os
import time
import PIL
import hydra
import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import inv
import datetime

# robustmvd / rmvd import
import rmvd

# Hydra + DUSt3R imports
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


import rootutils
rootutils.setup_root(
    "/home/jianingy/research/fast3r/src",
    indicator=".project-root",  # or remove if not needed
    pythonpath=True
)
from src.models.fast3r import Fast3R
from src.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from src.dust3r.inference_multiview import inference

from src.dust3r.datasets.utils.transforms import ImgNorm


########################
# 1) Utility: fix old references in config
########################
def replace_dust3r_in_config(cfg):
    for key, value in cfg.items():
        if isinstance(value, dict):
            replace_dust3r_in_config(value)
        elif isinstance(value, str):
            if "dust3r." in value and "src.dust3r." not in value:
                cfg[key] = value.replace("dust3r.", "src.dust3r.")
    return cfg



class Fast3RWrapperModel(nn.Module):
    """
    A custom wrapper so we can use Fast3R (DUSt3R) inside the rmvd framework.

    REQUIRED by rmvd:
      - input_adapter(images, keyview_idx, poses, intrinsics, depth_range)
      - forward(...)
      - output_adapter(...)

    Then we typically do:
       model = rmvd.prepare_custom_model(Fast3RWrapperModel())
    so that we get a .run(...) method suitable for rmvd's evaluation or inference pipelines.

    This version uses the local head output ('pts3d_local') for depth.
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.name = "Fast3R"
        self.model = None  # will be set outside by loading logic

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        """
        rmvd doc says: if images is a list, each item might be shape (1,3,H,W).
        We'll build DUSt3R 'list_of_views' from that list, ignoring poses/intrinsics.

        Because there's no "batch dimension" > 1, we produce an output with no batch dimension either.
        """
        device = next(self.model.parameters()).device
        
        if not isinstance(images, list):
            raise TypeError("Expected `images` to be a list of arrays [ (1,3,H,W), ... ].")

        list_of_views = []
        for arr in images:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Each item in images must be np.ndarray, got {type(arr)}")
            if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] != 3:
                raise ValueError(f"Expected shape (1,3,H,W), got {arr.shape}")

            # use ImgNorm to normalize, images original are float32 and values from 0 to 255
            pil_image = PIL.Image.fromarray(arr[0].astype(np.uint8).transpose(1,2,0))
            arr_t = ImgNorm(pil_image).unsqueeze(0).to(device)  # shape (3,H,W), convert to tensor

            # Build single view
            view_dict = {
                "img": arr_t,  # DUSt3R expects (B=1,3,H,W) here
                "dataset": ["rmvd"]
            }
            list_of_views.append(view_dict)

        # We'll store keyview_idx if needed
        sample = {
            "list_of_views": list_of_views,
            "keyview_idx": keyview_idx if isinstance(keyview_idx, int) else 0
        }
        return sample

    def forward(self, list_of_views=None, keyview_idx=0):
        """
        Actually run DUSt3R. We'll do a single call to 'inference' with all views.
        Then we store 'keyview_idx' in the output to pick the reference in output_adapter.
        """
        device = next(self.model.parameters()).device
        
        if list_of_views is None:
            return {}
        out = inference(
            list_of_views,
            model=self.model,
            dtype=torch.float32,
            device=device,
            profiling=False,
            verbose=False
        )
        out["keyview_idx"] = keyview_idx
        return out


    def output_adapter(self, model_output):
        """
        We'll produce a single reference depth => shape(1,H,W).
         - local: 'pts3d_local' => shape(1,H,W,3)
         - local: 'conf_local' => shape(1,H,W) => depth_uncertainty = 1 - conf_local
        fallback to global => 'pts3d_in_other_view' / 'conf'.

        We'll save debug images of depth & uncertainty to 'depth_viz'.
        """
        if not model_output or "preds" not in model_output:
            return {}, {}

        preds_list = model_output["preds"]
        keyidx = model_output.get("keyview_idx", 0)
        if keyidx >= len(preds_list):
            keyidx = 0

        ref_pred = preds_list[keyidx]
        # Extract 3D points
        # pts3d = ref_pred.get("pts3d_local", None)
        # conf_map = ref_pred.get("conf_local", None)

        pts3d = ref_pred.get("pts3d_in_other_view", None)
        conf_map = ref_pred.get("conf", None)

        # shape => (1,H,W,3)
        # Instead of  depth_tensor = pts3d_tensor[..., 2]
        # do the L2 distance from origin => sqrt(x^2 + y^2 + z^2)
        # depth_tensor = torch.sqrt(
        #     pts3d[..., 0] ** 2
        #     + pts3d[..., 1] ** 2
        #     + pts3d[..., 2] ** 2
        # ).unsqueeze(0)  # => shape (1,1,H,W)
        # depth_np = depth_tensor.cpu().numpy()  # => (1,1,H,W)
        
        # use the z-value
        depth_tensor = -pts3d[..., 2]  # => shape (1,H,W)
        depth_np = depth_tensor.cpu().numpy()  # => (1,H,W)
        depth_np = depth_np.squeeze(0)         # => (H,W)
        depth_np = depth_np[None,None, ...].astype(np.float32)  # => (1,1,H,W)

        depth_uncertainty_np = None
        if conf_map is not None:
            # shape(1,H,W)
            conf_tensor = conf_map
            conf_np = conf_tensor.cpu().numpy().astype(np.float32)  # => (1,H,W)
            conf_np = conf_np.squeeze(0)  # =>(H,W)
            unc_np = (1.0 - conf_np)      # =>(H,W)
            depth_uncertainty_np = unc_np[None, None,...]  # =>(1,1,H,W)

        # Optional debug save
        # folder = "/home/jianingy/research/fast3r/scripts/depth_viz"
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # import matplotlib.pyplot as plt

        # # Depth
        # depth_path = os.path.join(folder, f"depth_{timestamp}.png")
        # plt.imshow(depth_np.squeeze(), cmap='turbo')
        # plt.colorbar()
        # plt.title("Depth Map (Z=negative => bigger=closer)")
        # plt.savefig(depth_path)
        # plt.close()

        # Uncertainty
        # if depth_uncertainty_np is not None:
        #     unc_path = os.path.join(folder, f"depth_unc_{timestamp}.png")
        #     plt.imshow(depth_uncertainty_np.squeeze(), cmap='gray')
        #     plt.colorbar()
        #     plt.title("Depth Uncertainty (1-conf)")
        #     plt.savefig(unc_path)
        #     plt.close()

        pred = {"depth": depth_np}
        if depth_uncertainty_np is not None:
            pred["depth_uncertainty"] = depth_uncertainty_np

        aux = {}
        return pred, aux




if __name__ == "__main__":


    # -------------------------------------------
    # Load the lightning module from checkpoint
    # -------------------------------------------
    # Checkpoint
    checkpoint_dir = "/data/jianingy/dust3r_data/fast3r_checkpoints/super_long_training_5175604"
    possible_dir = os.path.join(checkpoint_dir, "checkpoints", "last.ckpt")
    if os.path.isdir(possible_dir):
        # Convert zero checkpoint
        ckpt_agg = os.path.join(checkpoint_dir, "checkpoints", "last_aggregated.ckpt")
        if not os.path.exists(ckpt_agg):
            convert_zero_checkpoint_to_fp32_state_dict(possible_dir, ckpt_agg, tag=None)
        CKPT_PATH = ckpt_agg
    else:
        CKPT_PATH = os.path.join(checkpoint_dir, "checkpoints", "last.ckpt")

    cfg_path = os.path.join(os.path.dirname(CKPT_PATH), "../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.model.net = replace_dust3r_in_config(cfg.model.net)

    if "encoder_args" in cfg.model.net:
        cfg.model.net.encoder_args.patch_embed_cls = "PatchEmbedDust3R"
        cfg.model.net.head_args.landscape_only = False
    else:
        cfg.model.net.patch_embed_cls = "PatchEmbedDust3R"
        cfg.model.net.landscape_only = False

    cfg.model.net.decoder_args.random_image_idx_embedding = True
    cfg.model.net.decoder_args.attn_bias_for_inference_enabled = False

    # cfg.model.net._target_ = "scripts.robustmvd_eval.Fast3RWrapperModel"  # replace with wrapper

    lit_module = hydra.utils.instantiate(
        cfg.model, train_criterion=None, validation_criterion=None
    )

    lit_module = MultiViewDUSt3RLitModule.load_from_checkpoint(
        checkpoint_path=CKPT_PATH,
        net=lit_module.net,
        train_criterion=lit_module.train_criterion,
        validation_criterion=lit_module.validation_criterion
    )
    lit_module.eval()
    
    model_wrapper = Fast3RWrapperModel()
    model_wrapper.model = lit_module.net

    model = rmvd.prepare_custom_model(model_wrapper)  # adds the .run(...) method

    # dataset = rmvd.create_dataset(dataset_name_or_path="eth3d", dataset_type="mvd", split="robustmvd", input_size=(384, 512))  # explicitly specify the split
    # dataset = rmvd.create_dataset(dataset_name_or_path="dtu", dataset_type="mvd", split="robustmvd", input_size=(384, 512))
    dataset = rmvd.create_dataset(dataset_name_or_path="scannet", dataset_type="mvd", split="robustmvd", input_size=(384, 512))
    # dataset = rmvd.create_dataset(dataset_name_or_path="tanks_and_temples", dataset_type="mvd", split="robustmvd", input_size=(384, 512))
    evaluation = rmvd.create_evaluation(evaluation_type="mvd", out_dir="/home/jianingy/research/fast3r/scripts/rmvd_eval_output_scannet_global_head",
                                        inputs=["intrinsics", "poses"], eval_uncertainty=False, alignment="median")
    results = evaluation(dataset=dataset, model=model)
    print("Eval results:", results)
