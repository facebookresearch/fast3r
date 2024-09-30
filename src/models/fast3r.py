import os
from copy import deepcopy
import huggingface_hub
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from src.dust3r.heads.postprocess import postprocess
from src.dust3r.heads.dpt_head import PixelwiseTaskWithDPT
from src.croco.models.blocks import Block
from src.croco.models.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from packaging import version
from functools import partial

from src.dust3r.patch_embed import get_patch_embed

from src.dust3r.utils.misc import (
    freeze_all_params,
    transpose_to_landscape,
)
import torch.autograd.profiler as profiler

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse(
    "0.22.0"
), "Outdated huggingface_hub version, please reinstall requirements.txt"


class Fast3R(nn.Module,
             huggingface_hub.PyTorchModelHubMixin,
             library_name="fast3r",
             repo_url="https://github.com/jedyang97/fast3r",
             tags=["image-to-3d"]
             ):
    def __init__(
        self,
        encoder_args: dict,
        decoder_args: dict,
        head_args: dict,
        freeze="none",
    ):
        super(Fast3R, self).__init__()

        self.encoder_args = encoder_args
        self.build_encoder(encoder_args)

        self.decoder_args = decoder_args
        self.build_decoder(decoder_args)

        self.head_args = head_args
        self.build_head(head_args)

        self.set_freeze(freeze)

    def build_encoder(self, encoder_args: dict):
        # Initialize the encoder based on the encoder type
        if encoder_args["encoder_type"] == "croco":
            # drop the encoder_type key
            encoder_args = deepcopy(encoder_args)
            encoder_args.pop("encoder_type")
            self.encoder = CroCoEncoder(**encoder_args)
        elif encoder_args["encoder_type"] == "dino_v2":
            # Load the pretrained dino_v2 model
            self.encoder = torch.hub.load('facebookresearch/dino', 'dino_v2_vitb16')
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_args['encoder_type']}")

    def build_decoder(self, decoder_args: dict):
        self.decoder = Fast3RDecoder(**decoder_args)

    def build_head(
        self,
        head_args: dict,
    ):
        assert (
            head_args['img_size'] % head_args['patch_size'] == 0
        ), f"{head_args['img_size']=} must be multiple of {head_args['patch_size']=}"
        self.output_mode = head_args['output_mode']
        self.head_type = head_args['head_type']
        self.depth_mode = head_args['depth_mode']
        self.conf_mode = head_args['conf_mode']
        # allocate head
        self.downstream_head = self.head_factory(
            head_args['head_type'], head_args['output_mode'], has_conf=bool(head_args['conf_mode'])
        )
        # magic wrapper
        self.head = transpose_to_landscape(
            self.downstream_head, activate=head_args['landscape_only']
        )

    def head_factory(self, head_type, output_mode, has_conf=False):
        """ " build a prediction head for the decoder"""
        if head_type == "dpt" and output_mode == "pts3d":
            assert self.decoder_args.depth > 9
            l2 = self.decoder_args.depth
            feature_dim = 256
            last_dim = feature_dim // 2
            out_nchan = 3
            ed = self.encoder_args.embed_dim
            dd = self.decoder_args.embed_dim
            return PixelwiseTaskWithDPT(
                num_channels=out_nchan + has_conf,
                feature_dim=feature_dim,
                last_dim=last_dim,
                hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                dim_tokens=[ed, dd, dd, dd],
                postprocess=postprocess,
                depth_mode=self.head_args.depth_mode,
                conf_mode=self.head_args.conf_mode,
                head_type="regression",
            )
        else:
            raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            return super(Fast3R, cls).from_pretrained(
                pretrained_model_name_or_path, **kw
            )

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)

    def load_from_dust3r_checkpoint(self, dust3r_checkpoint_path: str):
        """Load a Dust3R checkpoint into the model.
        Only load the patch_embed, enc_blocks, enc_norm, and downstream_head1 components from the checkpoint.

        Args:
            dust3r_checkpoint_path (str): Path to the Dust3R checkpoint.
        """
        # Load the checkpoint
        checkpoint = torch.load(dust3r_checkpoint_path)['model']

        # Initialize state dictionaries for different components
        encoder_state_dict = {}
        downstream_head_state_dict = {}

        # Prepare logging
        loaded_keys = set()
        not_loaded_keys = set()

        # Split the checkpoint into encoder and downstream head
        for key, value in checkpoint.items():
            if key.startswith("patch_embed") or key.startswith("enc_blocks") or key.startswith("enc_norm"):
                # These are encoder components
                if isinstance(self.encoder, CroCoEncoder):
                    new_key = key.replace("patch_embed", "encoder.patch_embed") \
                                .replace("enc_blocks", "encoder.enc_blocks") \
                                .replace("enc_norm", "encoder.enc_norm")
                    encoder_state_dict[new_key] = value
                    loaded_keys.add(key)  # Store full key
                else:
                    not_loaded_keys.add(key)
            elif key.startswith("downstream_head1"):
                # These are for downstream_head
                new_key = key.replace("downstream_head1", "downstream_head")
                downstream_head_state_dict[new_key] = value
                loaded_keys.add(key)  # Store full key
            else:
                not_loaded_keys.add(key)

        # Load the encoder part into the model if it is an instance of CroCoEncoder
        if isinstance(self.encoder, CroCoEncoder):
            self.encoder.load_state_dict(encoder_state_dict, strict=False)

        # Load the downstream head part into the model
        self.downstream_head.load_state_dict(downstream_head_state_dict, strict=False)

        del checkpoint

        # Process keys to log only first-level names
        loaded_first_level_keys = {key.split('.')[0] for key in loaded_keys}
        not_loaded_first_level_keys = {key.split('.')[0] for key in not_loaded_keys}

        # Log unique first-level keys
        log.info(f"Loaded first-level keys: {sorted(loaded_first_level_keys)}")
        log.info(f"First-level keys not loaded: {sorted(not_loaded_first_level_keys)}")

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "encoder": [self.encoder],
            "sandwich": [self.encoder, self.downstream_head],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _encode_images(self, views):
        B = views[0]["img"].shape[0]
        encoded_feats, positions, shapes = [], [], []

        # TODO: Batchify this
        for view in views:
            img = view["img"]
            true_shape = view.get(
                "true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1)
            )
            feat, pos = self.encoder(img, true_shape)
            encoded_feats.append(feat)
            positions.append(pos)
            shapes.append(true_shape)

        return encoded_feats, positions, shapes

    def forward(self, views):
        """
        Args:
            views (list[dict]): a list of views, each view is a dict of tensors, the tensors are batched

        Returns:
            list[dict]: a list of results for each view
        """
        # encode the images --> B,S,D
        encoded_feats, positions, shapes = self._encode_images(views)

        # Create image IDs for each patch
        num_images = len(views)
        B, _, _ = encoded_feats[0].shape

        different_resolution_across_views = not all(encoded_feats[0].shape[1] == encoded_feat.shape[1] for encoded_feat in encoded_feats)

        # Initialize an empty list to collect image IDs for each patch.
        # Note that at inference time, different views may have different number of patches.
        image_ids = []

        # Loop through each encoded feature to get the actual number of patches
        for i, encoded_feat in enumerate(encoded_feats):
            num_patches = encoded_feat.shape[1]  # Get the number of patches for this image
            # Extend the image_ids list with the current image ID repeated num_patches times
            image_ids.extend([i] * num_patches)

        # Repeat the image_ids list B times and reshape it to match the expected shape
        image_ids = torch.tensor(image_ids * B).reshape(B, -1).to(encoded_feats[0].device)

        # combine all ref images into object-centric representation
        dec_output = self.decoder(encoded_feats, positions, image_ids)

        ################## Forward pass through the head ##################
        # TODO: optimize this

        # Initialize the final results list
        final_results = [{} for _ in range(num_images)]

        with profiler.record_function("head: gathered outputs"):
            # Prepare the gathered outputs for each layer
            gathered_outputs_list = []
            if different_resolution_across_views:  # If the views have different resolutions, gathered_outputs_list is a list of lists, the outer list is for different views, and the inner list is for different layers
                for img_id in range(num_images):
                    gathered_outputs_per_view = []
                    for layer_output in dec_output:
                        B, P, D = layer_output.shape
                        mask = (image_ids == img_id)
                        gathered_output = layer_output[mask].view(B, -1, D)
                        gathered_outputs_per_view.append(gathered_output)
                    gathered_outputs_list.append(gathered_outputs_per_view)
            else:  # If the views have the same resolution, gathered_outputs_list is a list of tensors, each tensor is for a different layer
                for layer_output in dec_output:
                    B, P, D = layer_output.shape
                    gathered_outputs_per_view = []
                    for img_id in range(num_images):
                        mask = (image_ids == img_id)
                        gathered_output = layer_output[mask].view(B, -1, D)
                        gathered_outputs_per_view.append(gathered_output)
                    gathered_outputs_list.append(torch.cat(gathered_outputs_per_view, dim=0))  # fold the view dimension into batch dimension

        with profiler.record_function("head: forward pass"):
            if different_resolution_across_views:
                # Forward pass for each view separately
                final_results = [{} for _ in range(num_images)]
                for img_id in range(num_images):
                    img_result = self.head(gathered_outputs_list[img_id], shapes[img_id])
                    # Re-map the results back to the original batch and image order
                    for key in img_result.keys():
                        if key == 'pts3d':
                            final_results[img_id]['pts3d_in_other_view'] = img_result[key]
                        else:
                            final_results[img_id][key] = img_result[key]
            else:
                # Concatenate shapes
                concatenated_shapes = torch.cat(shapes, dim=0)

                # Forward pass through self.head()
                result = self.head(gathered_outputs_list, concatenated_shapes)

                # Initialize the final results list
                final_results = [{} for _ in range(num_images)]

                # Re-map the results back to the original batch and image order
                for key in result.keys():
                    for img_id in range(num_images):
                        img_result = result[key][img_id * B:(img_id + 1) * B]
                        if key == 'pts3d':
                            final_results[img_id]['pts3d_in_other_view'] = img_result
                        else:
                            final_results[img_id][key] = img_result

        return final_results

class CroCoEncoder(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        patch_embed_cls="ManyAR_PatchEmbed",
        embed_dim=768,
        num_heads=12,
        depth=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_embed="RoPE100",
        attn_implementation="pytorch_naive",
    ):
        super(CroCoEncoder, self).__init__()

        # patch embeddings  (with initialization done as in MAE)
        self.patch_embed_cls = patch_embed_cls
        self._set_patch_embed(img_size, patch_size, embed_dim)

        # Positional embedding
        self.pos_embed = pos_embed
        if pos_embed.startswith("RoPE"):  # eg RoPE100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)

        # Transformer blocks
        self.enc_blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer,
                  rope=self.rope,
                  attn_implementation=attn_implementation)
            for _ in range(depth)
        ])
        self.enc_norm = norm_layer(embed_dim)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def forward(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # Apply encoder blocks
        for blk in self.enc_blocks:
            x = blk(x, pos)

        # Apply final normalization
        x = self.enc_norm(x)
        return x, pos

class Fast3RDecoder(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        enc_embed_dim: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_implementation: str = "pytorch_naive",
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(Fast3RDecoder, self).__init__()

        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)

        self.dec_blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop= attn_drop,
                norm_layer=nn.LayerNorm,
                attn_implementation=attn_implementation,
            ) for _ in range(depth)
        ])

        # initialize the positional embedding for the decoder
        self.random_image_idx_embedding = random_image_idx_embedding
        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(1000))
            ).float(),
            persistent=False,
        )

        # final norm layer
        self.dec_norm = norm_layer(embed_dim)

    def _generate_per_rank_generator(self):
        # this way, the randperm will be different for each rank, but deterministic given a fixed number of forward passes (tracked by self.random_generator)
        # and to ensure determinism when resuming from a checkpoint, we only need to save self.random_generator to state_dict
        # generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        """
        Generates non-repeating random image indices for each sample, retrieves corresponding
        positional embeddings for each view, and concatenates them.

        Args:
            encoded_feats (list of tensors): Encoded features for each view.
            batch_size (int): Number of samples in the batch.
            num_views (int): Number of views per sample.
            max_image_idx (int): Maximum image index for embedding.
            device (torch.device): Device to move data to.

        Returns:
            Tensor: Concatenated positional embeddings for the entire batch.
        """
        # Generate random non-repeating image IDs (on CPU)
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)

        # First view is always 0 for all samples
        image_ids[:, 0] = 0

        # Get a generator that is unique to each rank, while also being deterministic based on the global across numbers of forward passes
        per_rank_generator = self._generate_per_rank_generator()

        # Generate random non-repeating IDs for the remaining views using the generator
        for b in range(batch_size):
            # Use the torch.Generator for randomness to ensure randomness between forward passes
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
            image_ids[b, 1:] = random_ids

        # Move the image IDs to the correct device
        image_ids = image_ids.to(device)

        # Initialize list to store positional embeddings for all views
        image_pos_list = []

        for i in range(num_views):
            # Retrieve the number of patches for this view
            num_patches = encoded_feats[i].shape[1]

            # Gather the positional embeddings for the entire batch based on the random image IDs
            image_pos_for_view = self.image_idx_emb[image_ids[:, i]]  # (B, D)

            # Expand the positional embeddings to match the number of patches
            image_pos_for_view = image_pos_for_view.unsqueeze(1).repeat(1, num_patches, 1)

            image_pos_list.append(image_pos_for_view)

        # Concatenate positional embeddings for all views along the patch dimension
        image_pos = torch.cat(image_pos_list, dim=1)  # (B, Npatches_total, D)

        return image_pos

    def forward(self, encoded_feats, positions, image_ids):
        """ Forward pass through the decoder.

        Args:
            encoded_feats (list of tensors): Encoded features for each view. Shape: B x Npatches x D
            positions (list of tensors): Positional embeddings for each view. Shape: B x Npatches x 2
            image_ids (tensor): Image IDs for each patch. Shape: B x Npatches
        """
        x = torch.cat(encoded_feats, dim=1)  # concate along the patch dimension
        pos = torch.cat(positions, dim=1)

        final_output = [x]  # before projection

        # project to decoder dim
        x = self.decoder_embed(x)

        # Add positional embedding based on image IDs
        if self.random_image_idx_embedding:
            # Generate random positional embeddings for all views and samples
            image_pos = self._get_random_image_pos(encoded_feats=encoded_feats,
                                                   batch_size=encoded_feats[0].shape[0],
                                                   num_views=len(encoded_feats),
                                                   max_image_idx=self.image_idx_emb.shape[0] - 1,
                                                   device=x.device)
        else:
            # Use default image IDs from input
            num_images = (torch.max(image_ids) + 1).cpu().item()
            image_idx_emb = self.image_idx_emb[:num_images]
            image_pos = image_idx_emb[image_ids]

        # Apply positional embedding based on image IDs and positions
        x += image_pos  # x has size B x Npatches x D, image_pos has size Npatches x D, so this is broadcasting

        for blk in self.dec_blocks:
            x = blk(x, pos)
            final_output.append(x)

        x = self.dec_norm(x)
        final_output[-1] = x

        return final_output
