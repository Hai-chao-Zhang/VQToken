#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from scipy.ndimage import median_filter
from matplotlib.colors import Normalize
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2
import numpy as np
import os
import random

import os,random
import numpy as np
import cv2

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

from VQToken.vq_token import kmeans_clustering_tokens_torch, adaptive_kmeans_clustering_tokens_torch
from VQToken.vq_attn import VQAttn

import torch.nn.functional as F

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
        self.cross_attention = VQAttn(query_dim=729, context_dim=896, num_heads=8)

        # # Apply cross-attention to get weighted clusters
        # weighted_clusters = self.cross_attention.cross_attention_weighted_clusters(query_tensor, context_tensor)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def token_dynamics(self, image_feature, vis=False, adaptive=''):
        # rank0_print("Token Dynamics")
        # image_feature = image_feature.permute(2, 0, 1).contiguous()
        # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        # image_feature = image_feature.permute(1, 2, 0).contiguous()
        # return image_feature
        if not adaptive:
            cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=32)
            # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=12)
            # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=24)
        else:
            # cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=12, method=adaptive) # , self.config.num_iters) #image
            # cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=128, method=adaptive) # , self.config.num_iters) #image
            cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=32, method=adaptive) # , self.config.num_iters) #image
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=12) # , self.config.num_iters) #image
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=int(0.3* image_feature.shape[0]*image_feature.shape[1])) # , self.config.num_iters) #image
        cluster_indices = cluster_indices.type(image_feature.dtype)
        print(f"################ ######## #### ## # Cluster indices shape: {clusters.shape[0]}")
        # Apply cross-attention to get weighted clusters

        # cluster_indices = torch.randn_like(cluster_indices)
        # clusters = torch.randn_like(clusters)
        
        weighted_clusters = self.get_model().cross_attention.cross_attention_weighted_clusters(cluster_indices, clusters)
        #### TODO: Check if I add parameters in att)func into training optimizer.
        #### TODO: check the implementation of crs_attn_func and the kmeans_clustering_tokens_torch, check if the layer is correct.
        if vis:
            return weighted_clusters, cluster_indices
        return weighted_clusters
        # out_image_features.append(torch.cat([weighted_clusters, clusters])) #.to(torch.bfloat16)

    def token_dynamics_random_base(self, image_feature, vis=False, adaptive=''):
        # rank0_print("Token Dynamics")
        # image_feature = image_feature.permute(2, 0, 1).contiguous()
        # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        # image_feature = image_feature.permute(1, 2, 0).contiguous()
        # return image_feature
        if not adaptive:
            cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=32)
        else:
            # cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=12, method=adaptive) # , self.config.num_iters) #image
            cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=32, method=adaptive) # , self.config.num_iters) #image
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=12) # , self.config.num_iters) #image
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=int(0.3* image_feature.shape[0]*image_feature.shape[1])) # , self.config.num_iters) #image
        cluster_indices = cluster_indices.type(image_feature.dtype)
        print(f"################ ######## #### ## # Cluster indices shape: {clusters.shape[0]}")
        # Apply cross-attention to get weighted clusters

        # cluster_indices = torch.randn_like(cluster_indices)
        clusters = torch.randn_like(clusters)
        
        weighted_clusters = self.get_model().cross_attention.cross_attention_weighted_clusters(cluster_indices, clusters)
        #### TODO: Check if I add parameters in att)func into training optimizer.
        #### TODO: check the implementation of crs_attn_func and the kmeans_clustering_tokens_torch, check if the layer is correct.
        if vis:
            return weighted_clusters, cluster_indices
        return weighted_clusters
        # out_image_features.append(torch.cat([weighted_clusters, clusters])) #.to(torch.bfloat16)

    def token_dynamics_only_base(self, image_feature, vis=False, adaptive=''):
        # rank0_print("Token Dynamics")
        # image_feature = image_feature.permute(2, 0, 1).contiguous()
        # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        # image_feature = image_feature.permute(1, 2, 0).contiguous()
        # return image_feature
        if not adaptive:
            # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=12)
            cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=32)
            # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=64)
        else:
            # cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=12, method=adaptive) # , self.config.num_iters) #image
            # cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=32, method=adaptive) # , self.config.num_iters) #image
            cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=32, method=adaptive) 
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=12) # , self.config.num_iters) #image
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=int(0.3* image_feature.shape[0]*image_feature.shape[1])) # , self.config.num_iters) #image
        cluster_indices = cluster_indices.type(image_feature.dtype)
        print(f"################ ######## #### ## # Cluster indices shape: {clusters.shape[0]}")
        # Apply cross-attention to get weighted clusters

        # cluster_indices = torch.randn_like(cluster_indices)
        # clusters = torch.randn_like(clusters)
        
        # weighted_clusters = self.get_model().cross_attention.cross_attention_weighted_clusters(cluster_indices, clusters)
        #### TODO: Check if I add parameters in att)func into training optimizer.
        #### TODO: check the implementation of crs_attn_func and the kmeans_clustering_tokens_torch, check if the layer is correct.
        if vis:
            return clusters, cluster_indices
        return clusters
        # out_image_features.append(torch.cat([weighted_clusters, clusters])) #.to(torch.bfloat16)

    def token_dynamics_random_map(self, image_feature, vis=False, adaptive=''):
        # rank0_print("Token Dynamics")
        # image_feature = image_feature.permute(2, 0, 1).contiguous()
        # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        # image_feature = image_feature.permute(1, 2, 0).contiguous()
        # return image_feature
        if not adaptive:
            cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=32)
        else:
            # cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=12, method=adaptive) # , self.config.num_iters) #image
            cluster_indices, clusters = adaptive_kmeans_clustering_tokens_torch(image_feature, max_K=32, method=adaptive) # , self.config.num_iters) #image
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=12) # , self.config.num_iters) #image
        # cluster_indices, clusters = kmeans_clustering_tokens_torch(image_feature, K=int(0.3* image_feature.shape[0]*image_feature.shape[1])) # , self.config.num_iters) #image
        cluster_indices = cluster_indices.type(image_feature.dtype)
        print(f"################ ######## #### ## # Cluster indices shape: {clusters.shape[0]}")
        # Apply cross-attention to get weighted clusters

        cluster_indices = torch.randn_like(cluster_indices)
        # clusters = torch.randn_like(clusters)
        
        weighted_clusters = self.get_model().cross_attention.cross_attention_weighted_clusters(cluster_indices, clusters)
        #### TODO: Check if I add parameters in att)func into training optimizer.
        #### TODO: check the implementation of crs_attn_func and the kmeans_clustering_tokens_torch, check if the layer is correct.
        if vis:
            return weighted_clusters, cluster_indices
        return weighted_clusters
        # out_image_features.append(torch.cat([weighted_clusters, clusters])) #.to(torch.bfloat16)

    def token_pruning_simple(self, image_feature, vis=False, adaptive="", keep_ratio=0.10):#0.0014
        """
        Perform Token Pruning on each frame by keeping the most important tokens based on their norm.
        
        Optimized for lower memory usage by processing in chunks.

        Args:
        - image_feature (torch.Tensor): Tokenized image feature with shape [num_frames, token_num_per_frame, token_dim].
        - keep_ratio (float): Ratio of tokens to keep.
        - chunk_size (int): Number of tokens processed at a time to reduce memory footprint.

        Returns:
        - pruned_features (torch.Tensor): Pruned token sequence for each frame, shape [num_frames, new_token_num, token_dim].
        """
        chunk_size=256
        num_frames, token_num_per_frame, token_dim = image_feature.shape
        # keep_token_num = max(1, int(token_num_per_frame * keep_ratio))  # Ensure at least one token remains
        keep_token_num = 12
        
        pruned_features = []

        for frame in range(num_frames):
            frame_feature = image_feature[frame]  # Shape: [token_num_per_frame, token_dim]

            # **Step 1: Compute token norms in chunks to save memory**
            token_norms = torch.zeros(token_num_per_frame, device=image_feature.device)
            for start in range(0, token_num_per_frame, chunk_size):
                end = min(start + chunk_size, token_num_per_frame)
                token_norms[start:end] = torch.norm(frame_feature[start:end], dim=-1)  # Compute norms only for this chunk

            # **Step 2: Select top `keep_token_num` tokens with highest norms**
            top_indices = torch.topk(token_norms, keep_token_num, sorted=False).indices  # Shape: [keep_token_num]

            # **Step 3: Keep only the selected tokens**
            pruned_tokens = frame_feature[top_indices]  # Shape: [keep_token_num, token_dim]
            pruned_features.append(pruned_tokens)

        # **Ensure all frames have the same number of tokens**
        min_token_num = min(frame.shape[0] for frame in pruned_features)
        pruned_feature = torch.stack([frame[:min_token_num] for frame in pruned_features], dim=0)  # Shape: [num_frames, min_token_num, token_dim]

        return pruned_feature

        # """
        # Perform simple Token Pruning on each frame by keeping the most important tokens based on their norm.
        
        # Args:
        # - image_feature (torch.Tensor): Tokenized image feature with shape [num_frames, token_num_per_frame, token_dim].
        # - keep_ratio (float): Ratio of tokens to keep. For example, 0.5 means keeping half of the tokens.

        # Returns:
        # - pruned_features (torch.Tensor): Pruned token sequence for each frame, shape [num_frames, new_token_num, token_dim].
        # """
        # num_frames, token_num_per_frame, token_dim = image_feature.shape
        # keep_token_num = int(token_num_per_frame * keep_ratio)
        
        # pruned_features = []
        
        # # Apply Token Pruning independently for each frame
        # for frame in range(num_frames):
        #     frame_feature = image_feature[frame]  # Shape: [token_num_per_frame, token_dim]
            
        #     # Step 1: Compute the norm of each token
        #     token_norms = torch.norm(frame_feature, dim=-1)  # Shape: [token_num_per_frame]
            
        #     # Step 2: Find the top `keep_token_num` tokens based on norm
        #     top_indices = torch.topk(token_norms, keep_token_num).indices  # Shape: [keep_token_num]
            
        #     # Step 3: Keep only the selected tokens
        #     pruned_tokens = frame_feature[top_indices]  # Shape: [keep_token_num, token_dim]
        #     pruned_features.append(pruned_tokens)
        
        # # Stack the pruned tokens across all frames
        # pruned_features = torch.stack(pruned_features)  # Shape: [num_frames, keep_token_num, token_dim]
        # return pruned_features



    def token_merging_tome_vid(self, image_feature, vis=False, adaptive="", merge_ratio=0.90):#merge_ratio=0.9986
        """
        Perform Token Merging (ToMe) on image tokens to reduce the number of tokens.
        
        Args:
        - image_feature (torch.Tensor): Tokenized image feature with shape [num_frames, token_num_per_frame, token_dim].
        - merge_ratio (float): Ratio of tokens to merge. For example, 0.5 means reducing by half.

        Returns:
        - merged_feature (torch.Tensor): Compressed token sequence after ToMe, shape [num_frames, new_token_num, token_dim].
        """
        chunk_size=64
        num_frames, token_num_per_frame, token_dim = image_feature.shape
        # target_token_num = max(1, int(token_num_per_frame * (1 - merge_ratio)))  # Ensure at least one token remains
        target_token_num = 12
        
        merged_features = []

        for frame in range(num_frames):
            tokens = image_feature[frame]  # Shape: [token_num_per_frame, token_dim]

            # **Step 1: Compute pairwise cosine similarity in blocks**
            similarity_matrix = torch.full((token_num_per_frame, token_num_per_frame), -float('inf'), device=image_feature.device)

            for start in range(0, token_num_per_frame, chunk_size):
                end = min(start + chunk_size, token_num_per_frame)
                chunk = tokens[start:end]  # Shape: [chunk_size, token_dim]

                # Compute cosine similarity for the chunk
                sim_chunk = F.cosine_similarity(chunk.unsqueeze(1), tokens.unsqueeze(0), dim=-1)

                # Store computed values in the similarity matrix
                similarity_matrix[start:end] = sim_chunk

            similarity_matrix.fill_diagonal_(-float('inf'))  # Avoid self-similarity
            similarity_matrix = torch.triu(similarity_matrix)  # Upper triangle to avoid duplicates

            # **Step 2: Identify pairs with highest similarity**
            flat_indices = similarity_matrix.view(-1).topk(target_token_num, sorted=False).indices
            row_indices, col_indices = flat_indices // similarity_matrix.size(1), flat_indices % similarity_matrix.size(1)
            
            selected_pairs = torch.stack((row_indices, col_indices))

            # **Step 3: Merge selected tokens**
            merged_tokens = []
            used_indices = set()
            for i, j in zip(selected_pairs[0], selected_pairs[1]):
                if i.item() in used_indices or j.item() in used_indices:
                    continue
                merged_token = (tokens[i] + tokens[j]) / 2  # Merge by averaging
                merged_tokens.append(merged_token)
                used_indices.update([i.item(), j.item()])

            # **Step 4: Select unmerged tokens**
            mask = torch.ones(token_num_per_frame, dtype=torch.bool, device=image_feature.device)
            mask[row_indices] = False
            mask[col_indices] = False
            remaining_tokens = tokens[mask]

            # **Step 5: Concatenate merged and remaining tokens, ensuring correct output size**
            compressed_frame = torch.cat((torch.stack(merged_tokens), remaining_tokens[:target_token_num - len(merged_tokens)]), dim=0)

            merged_features.append(compressed_frame)

        # **Ensure all frames have the same number of tokens**
        min_token_num = min(frame.shape[0] for frame in merged_features)
        merged_feature = torch.stack([frame[:min_token_num] for frame in merged_features], dim=0)  # Shape: [num_frames, min_token_num, token_dim]

        return merged_feature


    def token_merging_tome(self, image_feature,  vis=False, adaptive="", merge_ratio=0.90): #merge_ratio=0.9986
        """
        Perform Token Merging (ToMe) on each frame with memory-efficient similarity computation.

        Args:
        - image_feature (torch.Tensor): Tokenized image feature [num_frames, token_num_per_frame, token_dim].
        - merge_ratio (float): Ratio of tokens to merge.
        - chunk_size (int): Number of tokens to process at once (for similarity calculation).

        Returns:
        - merged_features (torch.Tensor): Compressed token sequence [num_frames, new_token_num, token_dim].
        """
        num_frames, token_num_per_frame, token_dim = image_feature.shape
        # target_token_num = int(token_num_per_frame * (1 - merge_ratio))
        target_token_num = 12

        merged_features = []
        chunk_size=64

        for frame in range(num_frames):
            frame_feature = image_feature[frame]  # Shape: [token_num_per_frame, token_dim]

            # **Step 1: Compute similarity in chunks to reduce memory usage**
            similarity_matrix = torch.full((token_num_per_frame, token_num_per_frame), -float('inf'), device=image_feature.device)

            for start in range(0, token_num_per_frame, chunk_size):
                end = min(start + chunk_size, token_num_per_frame)
                chunk = frame_feature[start:end]  # Shape: [chunk_size, token_dim]

                # Compute cosine similarity for the current chunk
                sim_chunk = F.cosine_similarity(chunk.unsqueeze(1), frame_feature.unsqueeze(0), dim=-1)

                # Store computed values in the full similarity matrix
                similarity_matrix[start:end] = sim_chunk

            similarity_matrix.fill_diagonal_(-float('inf'))  # Exclude self-similarity
            similarity_matrix = torch.triu(similarity_matrix)  # Keep upper triangle to avoid duplicates

            # **Step 2: Find top pairs to merge**
            flat_indices = similarity_matrix.view(-1).topk(target_token_num, sorted=False).indices
            i_indices, j_indices = torch.div(flat_indices, token_num_per_frame, rounding_mode='floor'), flat_indices % token_num_per_frame

            # **Step 3: Merge selected token pairs**
            merged_tokens = (frame_feature[i_indices] + frame_feature[j_indices]) / 2

            # **Step 4: Select remaining unmerged tokens**
            mask = torch.ones(token_num_per_frame, dtype=torch.bool, device=image_feature.device)
            mask[i_indices] = False  # Mark merged tokens
            mask[j_indices] = False

            remaining_tokens = frame_feature[mask]

            # **Step 5: Concatenate merged & remaining tokens, trimming to target size**
            compressed_frame = torch.cat((merged_tokens, remaining_tokens[:target_token_num - merged_tokens.size(0)]), dim=0)

            merged_features.append(compressed_frame)

        # Find the minimum token count across all frames (ensures uniform shape)
        min_token_num = min(frame.shape[0] for frame in merged_features)

        # Trim each frame's tokens to the minimum length and stack
        merged_features = torch.stack([frame[:min_token_num] for frame in merged_features], dim=0)

        return merged_features  # Shape: [num_frames, min_token_num, token_dim]
        # """
        # Perform Token Merging (ToMe) on each frame individually to reduce the number of tokens per frame.
        
        # Args:
        # - image_feature (torch.Tensor): Tokenized image feature with shape [num_frames, token_num_per_frame, token_dim].
        # - merge_ratio (float): Ratio of tokens to merge. For example, 0.5 means reducing tokens by half within each frame.

        # Returns:
        # - merged_features (torch.Tensor): Compressed token sequence for each frame, shape [num_frames, new_token_num, token_dim].
        # """
        # num_frames, token_num_per_frame, token_dim = image_feature.shape
        # target_token_num = int(token_num_per_frame * (1 - merge_ratio))
        
        # merged_features = []
        
        # # Apply ToMe independently for each frame
        # for frame in range(num_frames):
        #     frame_feature = image_feature[frame]  # Shape: [token_num_per_frame, token_dim]
            
        #     # Step 1: Compute similarity within the frame
        #     similarity_matrix = F.cosine_similarity(frame_feature.unsqueeze(0), frame_feature.unsqueeze(1), dim=-1)
        #     similarity_matrix = torch.triu(similarity_matrix, diagonal=1)

        #     # Step 2: Find the pairs with highest similarity and merge them
        #     flat_indices = similarity_matrix.view(-1).topk(target_token_num).indices
        #     selected_pairs = np.unravel_index(flat_indices.cpu().numpy(), similarity_matrix.shape)
            
        #     # Convert selected pairs back to torch tensors
        #     selected_pairs = [torch.tensor(arr).to(image_feature.device) for arr in selected_pairs]

        #     # Step 3: Merge pairs
        #     merged_tokens = []
        #     used_indices = set()
        #     for i, j in zip(selected_pairs[0], selected_pairs[1]):
        #         if i.item() in used_indices or j.item() in used_indices:
        #             continue
        #         merged_token = (frame_feature[i, :] + frame_feature[j, :]) / 2
        #         merged_tokens.append(merged_token)
        #         used_indices.update([i.item(), j.item()])
            
        #     # Add unused tokens as-is
        #     remaining_tokens = [frame_feature[k, :] for k in range(token_num_per_frame) if k not in used_indices]
        #     merged_tokens.extend(remaining_tokens[:target_token_num])  # Ensure target size

        #     # Stack to form the new token set for this frame
        #     merged_tokens = torch.stack(merged_tokens, dim=0)  # Shape: [target_token_num, token_dim]
        #     merged_features.append(merged_tokens)

        # # Stack the merged tokens across all frames
        # # merged_features = torch.stack(merged_features)  # Shape: [num_frames, target_token_num, token_dim]
        # min_token_num = min([frame.shape[0] for frame in merged_features])

        # # Trim each frame's tokens to the minimum length
        # merged_features_trimmed = [frame[:min_token_num, :] for frame in merged_features]

        # # Stack trimmed frames along the new dimension
        # merged_features = torch.stack(merged_features_trimmed, dim=0)  # Shape: [num_frames, min_token_num, token_dim]
        # return merged_features

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature


    def smooth_cluster_indices(self, cluster_indices, frame_shape):
        """
        Smooth cluster indices to ensure spatial continuity.

        Args:
        - cluster_indices (np.ndarray): Shape (num_frames, num_tokens), original cluster assignments.
        - frame_shape (tuple): Tuple (num_frames, channels, height, width), representing the frame size.

        Returns:
        - smoothed_indices (np.ndarray): Shape (num_frames, num_tokens), smoothed cluster assignments.
        """
        num_frames, num_tokens = cluster_indices.shape
        height, width = frame_shape[2], frame_shape[3]  # Extract height and width of frames

        # Compute token grid dimensions dynamically
        token_grid_height = int(np.sqrt(num_tokens * (height / width)))  # Adjust for aspect ratio
        token_grid_width = num_tokens // token_grid_height  # Ensure correct total number

        if token_grid_height * token_grid_width != num_tokens:
            raise ValueError(f"Invalid token reshaping: Expected {num_tokens}, got {token_grid_height * token_grid_width}")

        smoothed_indices = np.zeros_like(cluster_indices)

        for i in range(num_frames):
            # Reshape to correct token grid dimensions
            frame_clusters = cluster_indices[i].reshape(token_grid_height, token_grid_width).astype(np.int32)  

            # Apply median filtering for smoothing
            smoothed_frame_clusters = median_filter(frame_clusters, size=3)  # 3x3 neighborhood

            # Flatten back to 1D
            smoothed_indices[i] = smoothed_frame_clusters.flatten()

        return smoothed_indices

    def save_cluster_heatmaps_with_frames(self, cluster_indices, frames, save_dir):
        """
        Save heatmaps of cluster_indices and the corresponding frames side by side,
        with a custom gradient-based heatmap.

        Args:
        - cluster_indices (torch.Tensor): Cluster indices with shape [num_frames, num_tokens].
        - frames (torch.Tensor): Corresponding frames with shape [num_frames, channels, height, width].
        - save_dir (str): Directory to save the heatmaps and frames.
        """
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        cluster_indices = cluster_indices.cpu().numpy()  # Convert cluster indices to NumPy
        frames = frames.cpu().numpy()  # Convert frames to NumPy

        # Apply smoothing
        cluster_indices = self.smooth_cluster_indices(cluster_indices, frames.shape)

        num_frames = cluster_indices.shape[0]
        heatmap_size = int(np.sqrt(cluster_indices.shape[1]))  # Assuming square token layout
        height, width = frames.shape[2], frames.shape[3]  # Get frame dimensions

        combined_images = []  # To store all combined images

        # Define a custom colormap
        colormap = plt.get_cmap("jet")  # Use a smooth colormap (e.g., Jet or Viridis)

        for frame_idx in range(num_frames):
            # Reshape cluster indices to the spatial grid
            frame_clusters = cluster_indices[frame_idx].reshape(heatmap_size, heatmap_size).astype(np.float32)

            # Normalize cluster indices for smooth gradient mapping
            norm = Normalize(vmin=frame_clusters.min(), vmax=frame_clusters.max())
            normalized_clusters = norm(frame_clusters)  # Normalize to range [0, 1]

            # Map normalized values to colormap
            heatmap = (colormap(normalized_clusters)[:, :, :3] * 255).astype(np.uint8)  # RGB format

            # Resize the heatmap to match the frame size
            heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)

            # Convert the frame to HWC format for OpenCV and scale to [0, 255]
            frame = frames[frame_idx].transpose(1, 2, 0)  # Convert CHW -> HWC
            frame = (frame * 255).clip(0, 255).astype(np.uint8)  # Ensure values are valid for uint8
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a transparent overlay of the heatmap on the frame
            overlay = cv2.addWeighted(frame, 0.7, heatmap_resized, 0.3, 0)

            # Combine frame, heatmap, and overlay into a single "row"
            combined_row = np.vstack((frame, heatmap_resized, overlay))
            combined_images.append(combined_row)

        # Concatenate all rows into a single large image
        big_picture = np.hstack(combined_images)

        # Save the final big picture
        save_path = os.path.join(save_dir, f"{str(random.randint(10000, 99999))}.png")
        cv2.imwrite(save_path, big_picture)







    # def save_cluster_heatmaps_with_frames(self, cluster_indices, frames, save_dir):
    #     """
    #     Save heatmaps of cluster_indices and the corresponding frames side by side,
    #     with a custom gradient-based heatmap.

    #     Args:
    #     - cluster_indices (torch.Tensor): Cluster indices with shape [num_frames, num_tokens].
    #     - frames (torch.Tensor): Corresponding frames with shape [num_frames, channels, height, width].
    #     - save_dir (str): Directory to save the heatmaps and frames.
    #     """
    #     os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    #     cluster_indices = cluster_indices.cpu().numpy()  # Convert cluster indices to NumPy
    #     frames = frames.cpu().numpy()  # Convert frames to NumPy

    #     num_frames = cluster_indices.shape[0]
    #     heatmap_size = int(np.sqrt(cluster_indices.shape[1]))
    #     height, width = frames.shape[2], frames.shape[3]  # Get frame dimensions

    #     combined_images = []  # To store all combined images

    #     # Define a custom colormap
    #     colormap = plt.get_cmap("jet")  # Use a smooth colormap (e.g., Jet or Viridis)

    #     for frame_idx in range(num_frames):
    #         # Reshape cluster indices to the spatial grid
    #         frame_clusters = cluster_indices[frame_idx].reshape(heatmap_size, heatmap_size).astype(np.float32)

    #         # Normalize cluster indices for smooth gradient mapping
    #         norm = Normalize(vmin=frame_clusters.min(), vmax=frame_clusters.max())
    #         normalized_clusters = norm(frame_clusters)  # Normalize to range [0, 1]

    #         # Map normalized values to colormap
    #         heatmap = (colormap(normalized_clusters)[:, :, :3] * 255).astype(np.uint8)  # RGB format

    #         # Resize the heatmap to match the frame size
    #         heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)

    #         # Convert the frame to HWC format for OpenCV and scale to [0, 255]
    #         frame = frames[frame_idx].transpose(1, 2, 0)  # Convert CHW -> HWC
    #         frame = (frame * 255).clip(0, 255).astype(np.uint8)  # Ensure values are valid for uint8

    #         # Create a transparent overlay of the heatmap on the frame
    #         overlay = cv2.addWeighted(frame, 0.7, heatmap_resized, 0.3, 0)

    #         # Combine frame, heatmap, and overlay into a single "row"
    #         combined_row = np.vstack((frame, heatmap_resized, overlay))
    #         combined_images.append(combined_row)

    #     # Concatenate all rows into a single large image
    #     big_picture = np.hstack(combined_images)

    #     # Save the final big picture
    #     save_path = os.path.join(save_dir, f"{str(random.randint(10000, 99999))}.png")
    #     cv2.imwrite(save_path, big_picture)


    # def save_cluster_heatmaps_with_frames(self, cluster_indices, frames, save_dir):
    #     """
    #     Save heatmaps of cluster_indices and the corresponding frames side by side.

    #     Args:
    #     - cluster_indices (torch.Tensor): Cluster indices with shape [num_frames, num_tokens].
    #     - frames (torch.Tensor): Corresponding frames with shape [num_frames, channels, height, width].
    #     - save_dir (str): Directory to save the heatmaps and frames.
    #     """
    #     os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    #     cluster_indices = cluster_indices.cpu().numpy()  # Convert cluster indices to NumPy
    #     frames = frames.cpu().numpy()  # Convert frames to NumPy

    #     num_frames = cluster_indices.shape[0]
    #     heatmap_size = int(np.sqrt(cluster_indices.shape[1]))
    #     height, width = frames.shape[2], frames.shape[3]  # Get frame dimensions

    #     combined_images = []  # To store all combined images

    #     for frame_idx in range(num_frames):
    #         # Reshape cluster indices to the spatial grid
    #         frame_clusters = cluster_indices[frame_idx].reshape(heatmap_size, heatmap_size).astype(np.uint8)

    #         # Normalize and apply colormap to the cluster indices
    #         normalized_clusters = cv2.normalize(frame_clusters, None, 0, 255, cv2.NORM_MINMAX)
    #         heatmap = cv2.applyColorMap(normalized_clusters, cv2.COLORMAP_JET)

    #         # Resize the heatmap to match the frame size
    #         heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)

    #         # Convert the frame to HWC format for OpenCV and scale to [0, 255]
    #         frame = frames[frame_idx].transpose(1, 2, 0)  # Convert CHW -> HWC
    #         frame = (frame * 255).clip(0, 255).astype(np.uint8)  # Ensure values are valid for uint8

    #         # Create a transparent overlay of the heatmap on the frame
    #         overlay = cv2.addWeighted(frame, 0.7, heatmap_resized, 0.3, 0)

    #         # Combine frame, heatmap, and overlay into a single "row"
    #         combined_row = np.vstack((frame, heatmap_resized, overlay))
    #         combined_images.append(combined_row)

    #     # Concatenate all rows into a single large image
    #     big_picture = np.hstack(combined_images)

    #     # Save the final big picture
    #     save_path = os.path.join(save_dir, f"{str(random.randint(10000, 99999))}.png")
    #     cv2.imwrite(save_path, big_picture)

        
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, vis=False):
        text_token_count = input_ids.shape[1]

        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images) #### image encoder TODO: ecoding
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []


            total_tokens_before = 0
            total_tokens_after = 0
            for idx, image_feat in enumerate(encoded_image_features):
                num_tokens_before = image_feat.shape[0] if len(image_feat.shape)==2 else image_feat.shape[0] * image_feat.shape[1]   # Original token count
                total_tokens_before += num_tokens_before  # Track total before compression


                if idx in video_idx_in_batch:

                    # compression_method = self.token_dynamics
                    
                    # compression_method = self.token_dynamics_random_map
                    # compression_method = self.token_dynamics_random_base
                    
                    compression_method = self.token_dynamics_only_base
                    # compression_method = self.token_merging_tome_vid
                    # compression_method = self.token_merging_tome
                    # compression_method = self.token_pruning_simple

                    # if compression_method == "2dPool":
                    #     compression_method = self.get_2dPool
                    # elif compression_method == "token_dynamics":
                    #     compression_method = self.token_dynamics
                    
                    
                    # adaptive='' # ''  or "elbow" or "silhouette"
                    adaptive='elbow' # ''  or "elbow" or "silhouette"
                    vis=False

                    if vis==True:
                        image_feat, cluster_indices = compression_method(image_feat, vis=True, adaptive=adaptive)
                        save_dir = "./tmp/LLaVa-Video/sv_dir"
                        # Visualize the cluster_indices and add it to visualized_images
                        self.save_cluster_heatmaps_with_frames(
                            cluster_indices=cluster_indices,
                            frames=images_list[idx],
                            save_dir=os.path.join(save_dir, f"video_{idx}")
                        )
                    else:
                        image_feat = compression_method(image_feat,adaptive=adaptive)

                    #    if compression_method != self.token_dynamics:
                    #         _, _, token_dim = image_feat.shape
                    if image_feat.dim() == 3:
                        image_feat = image_feat.view(-1, image_feat.shape[2])  # Reshape to [num_frames * target_token_num, token_dim]   

                    # if compression_method != self.token_dynamics:
                    #     _, _, token_dim = image_feat.shape
                    #     if image_feat.dim() == 3:
                    #         image_feat = image_feat.view(-1, token_dim)  # Reshape to [num_frames * target_token_num, token_dim]      


                    image_features.append((image_feat)) 

                    num_tokens_after = image_feat.shape[0] if len(image_feat.shape)==2 else image_feat.shape[0] * image_feat.shape[1]  # Token count after compression only count token number for 
                    total_tokens_after += num_tokens_after  # Track total after compression

                    #TODO: the image_feat is the token sequences
                    #TODO: get_2dPool is to pool the token sequences to a single token
                    #TODO: which is the most naive compression for the visual token sequences
                    #TODO: replace or add other compression methods here
                    # vide_idx_in_batch will trigger if the input is video

                else:
                    image_features.append(image_feat)

                    num_tokens_after = num_tokens_before  # No compression applied
                    total_tokens_after += num_tokens_after
                print(f"Video {idx}: Tokens Before: {num_tokens_before}, Tokens After: {num_tokens_after}")

            print(f"📉 Total Visual Tokens Before Compression: {total_tokens_before}")
            print(f"📈 Total Visual Tokens After Compression: {total_tokens_after}")


            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            # image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")

        # Count text tokens
        
        total_tokens_llm = text_token_count + total_tokens_after
        print(f"📝 Text Tokens: {text_token_count}")
        print(f"🔢 Total Tokens Input to LLM: {total_tokens_llm}")

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
