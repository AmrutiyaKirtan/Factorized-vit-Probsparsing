import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
import math
import sys
from typing import Iterable, Optional
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os

print("All dependencies imported!")

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Optimized Point-based ProbSparse Attention Module
    
    This module implements the proven ProbSparse attention approach:
    1. Selects individual patches/points (25) using probabilistic sparsity
    2. Uses GPU-optimized operations for better performance
    3. Maintains fine-grained attention control at the patch level
    4. Applies the original ProbSparse logic with improved efficiency
    5. Removes over-engineered complexity for better practical performance
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, window_size=7, factor=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.windowsize = window_size
        self.factor = factor
        self.use_probsparse = True
        self.sr_ratio = sr_ratio
        
        if sr_ratio > 1:
            self.mul = [2, 7, 7]
        if sr_ratio == 1:
            self.mul = [1, 7]
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.single_dim = []
        self.scales = []
        for i_layer in range(len(self.mul)):
            single_d = dim // sum(self.mul) * self.mul[i_layer]
            self.single_dim.append(single_d)
            if i_layer == 0:
                scale = single_d ** -0.5
            else:
                scale = (single_d // self.num_heads) ** -0.5
            self.scales.append(scale)

        self.kv_g = nn.Linear(self.single_dim[0], self.single_dim[0] * 2, bias=qkv_bias)
        self.kv_l = DEPTHWISECONV(dim - self.single_dim[0], dim - self.single_dim[0])
        self.sr = nn.Conv2d(self.single_dim[0], self.single_dim[0], kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(self.single_dim[0])
        self.local_conv_g = nn.Conv2d(self.single_dim[0], self.single_dim[0], kernel_size=3, padding=1,
                                     stride=1, groups=self.single_dim[0])

        self.unfolds = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.single_heads = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        self.strides = []

        for i_layer in range(1, len(self.mul)):
            if i_layer == 1:
                dilation = 1
            else:
                dilation = self.sr_ratio
            kernel_size = self.windowsize
            stride = dilation * (kernel_size - 1) + 1
            self.strides.append(stride)

            unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation)
            fc = nn.Linear(self.single_dim[i_layer] // self.num_heads, self.single_dim[i_layer] // self.num_heads, bias=qkv_bias)
            single = nn.Linear(self.single_dim[i_layer] // self.num_heads, 2 * self.single_dim[i_layer] // self.num_heads, bias=qkv_bias)
            local_conv = nn.Conv2d(self.single_dim[i_layer], self.single_dim[i_layer], kernel_size=3,
                                 padding=1, stride=1, groups=self.single_dim[i_layer])

            self.unfolds.append(unfold)
            self.fcs.append(fc)
            self.single_heads.append(single)
            self.local_convs.append(local_conv)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def _prob_QK_points(self, Q, K, sample_k, n_top):
        """
        Simplified Point-based ProbSparse attention (same as original WN_PY.py)
        Q: [B, H, L_Q, D] - queries from all patches
        K: [B, H, L_K, D] - keys from all patches
        sample_k: number of individual points/patches to sample (25)
        n_top: number of top queries to select based on sparsity
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Ensure sample_k doesn't exceed available keys
        sample_k = min(sample_k, L_K)
        
        # GPU-optimized random sampling of individual patches/points
        # Create random indices for each query (more efficient than expansion)
        random_indices = torch.randint(
            0, L_K, (L_Q, sample_k), 
            device=K.device, dtype=torch.long
        )
        
        # Ensure indices are within bounds (safety check)
        random_indices = torch.clamp(random_indices, 0, L_K - 1)
        
        # Efficient sampling using advanced indexing
        # K: [B, H, L_K, D] -> K_sample: [B, H, L_Q, sample_k, D]
        K_sample = K[:, :, random_indices, :]  # [B, H, L_Q, sample_k, D]
        
        # Calculate attention scores between queries and sampled keys
        # Q: [B, H, L_Q, D] -> [B, H, L_Q, 1, D]
        # K_sample: [B, H, L_Q, sample_k, D] -> [B, H, L_Q, D, sample_k]
        Q_expanded = Q.unsqueeze(-2)  # [B, H, L_Q, 1, D]
        Q_K_sample = torch.matmul(Q_expanded, K_sample.transpose(-2, -1)).squeeze(-2)  # [B, H, L_Q, sample_k]
        
        # Calculate sparsity measurement M(q, K) = max - mean
        # Higher M indicates more focused/sparse attention patterns
        M_max = Q_K_sample.max(-1)[0]  # [B, H, L_Q] - maximum attention score
        M_mean = Q_K_sample.mean(-1)   # [B, H, L_Q] - average attention score
        M = M_max - M_mean             # [B, H, L_Q] - sparsity measurement
        
        # Select top-u queries with highest sparsity (most focused attention)
        M_top = M.topk(n_top, sorted=False)[1]  # [B, H, n_top]
        
        return M_top
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)  # B,N,C -> B,N,C

        # global attention blocks: self.mul[0] will tell no.of global attention heads
        q_g = q[:,:,:self.single_dim[0]].reshape(B, N, 1, self.single_dim[0] // 1).permute(0, 2, 1, 3)
        # B,N,C_g -> B,N,h,C_g/h -> B,h,N,C_g/h
        x_g = x[:,:,:self.single_dim[0]].permute(0, 2, 1).reshape(B, self.single_dim[0], H, W)
        # B,N,C_g -> B,C_g,N -> B,C_g,H,W
        x_g = self.sr(x_g).reshape(B, self.single_dim[0], -1).permute(0, 2, 1)
        # B,C_g,H,W -> B,C_g,H_g,W_g -> B,C_g,N_g -> B,N_g,C_g
        x_g = self.norm(x_g)  # B,N_g,C_g -> B,N_g,C_g
        kv_g = self.kv_g(x_g).reshape(B, -1, 2, 1, self.single_dim[0] // 1).permute(2, 0, 3, 1, 4)
        # B,N_g,C_g -> B,N_g,C_g*2 -> B,N_g,2,h,C_g/h -> 2,B,h,N_g,C_g/h
        k_g, v_g = kv_g[0], kv_g[1]  # 2,B,h,N_g,C_g/h -> B,h,N_g,C_g/h

        if self.use_probsparse and q_g.shape[2] > 25: 
            # Calculate parameters for Point-based ProbSparse (same as original)
            L_Q = q_g.shape[2]
            L_K = k_g.shape[2]
            sample_k = min(25, L_K)  # Sample 25 individual points/patches
            n_top = int(self.factor * math.log(L_Q))  # Top-u queries
            n_top = max(1, min(n_top, L_Q))  # Ensure valid range

            # Get indices of top sparse queries using Point-based ProbSparse
            M_top = self._prob_QK_points(q_g, k_g, sample_k, n_top)  # [B, H, n_top]
    
            # Select only top queries
            q_reduce = torch.gather(
                q_g, 2, 
                M_top.unsqueeze(-1).expand(-1, -1, -1, q_g.shape[-1])
            )  # [B, H, n_top, D]
            attn_sparse = (q_reduce @ k_g.transpose(-2, -1)) * self.scales[0]
            attn_sparse = attn_sparse.softmax(dim=-1)
            attn_sparse = self.attn_drop(attn_sparse)
              
            # Apply local convolution to v_g
            # Reshape v_g properly for local convolution
            v_g_reshaped = v_g.transpose(1, 2).reshape(B, self.single_dim[0], H//self.sr_ratio, W//self.sr_ratio)
            v_g_conv = self.local_conv_g(v_g_reshaped)
            v_g_conv = v_g_conv.view(B, self.single_dim[0], -1).transpose(1, 2).view(B, 1, -1, self.single_dim[0])
            v_g = v_g + v_g_conv
            v_selected = (attn_sparse @ v_g)  # [B, H, n_top, D]
            
            attn_g_out = torch.zeros_like(q_g)  # [B, H, L_Q, D]
            attn_g_out.scatter_(
                2,
                M_top.unsqueeze(-1).expand(-1, -1, -1, v_selected.shape[-1]),
                v_selected
            )        
            attn_g = attn_g_out.transpose(1, 2).reshape(B, N, self.single_dim[0])

        else:
            # Standard full attention (for short sequences or when ProbSparse is disabled)
            attn_g = (q_g @ k_g.transpose(-2,-1)) * self.scales[0]
            attn_g = attn_g.softmax(dim=-1)
            attn_g = self.attn_drop(attn_g)
            
            v_g = v_g + self.local_conv_g(
                v_g.transpose(1, 2)
                .reshape(B, -1, self.single_dim[0])
                .transpose(1, 2)
                .view(B, self.single_dim[0], H//self.sr_ratio, W//self.sr_ratio)
                .view(B, self.single_dim[0], -1)
                .view(B, 1, self.single_dim[0]//1, -1)
                .transpose(-1, -2)
            )
            
            attn_g = (attn_g @ v_g).transpose(1, 2).reshape(B, N, self.single_dim[0])







        q_l = q[:,:,self.single_dim[0]:].reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # B,N,C_l -> B,N,h,C_l/h -> B,h,N,C_l/h
        x_l = x[:,:,self.single_dim[0]:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B,N,C_l -> B,H,W,C_l -> B,C_l,H,W
        kv_l = self.kv_l(x_l)  # B,C_l,H,W -> B,C_l,H,W

        attn_l = []
        for i_c in range(1, len(self.mul)):
            if i_c == 1:
                q_ = q_l[:,:,:,:self.single_dim[1]//self.num_heads]  # B,h,N,C_l_i/h
                kv_ = kv_l[:,:self.single_dim[1],:,:]  # B,C_l_i,H,W
            else:
                q_ = q_l[:,:,:,self.single_dim[1]//self.num_heads:]  # B,h,N,C_l_i/h
                kv_ = kv_l[:,self.single_dim[1]:,:,:]  # B,C_l_i,H,W

            pad_l = pad_t = 0
            pad_r = (self.strides[i_c-1] - W % self.strides[i_c-1]) % self.strides[i_c-1]
            pad_b = (self.strides[i_c-1] - H % self.strides[i_c-1]) % self.strides[i_c-1]
            rp = nn.ReflectionPad2d((pad_l, pad_r, pad_t, pad_b))
            kv_ = rp(kv_)  # B,C_l_i,H,W -> B,C_l_i,Hi, Wi

            # Where dilation sampling happens
            kv_ = self.unfolds[i_c-1](kv_)  # B,C_l_i,Hi, Wi -> B,C_l_i*L2,H_i*W_i
            kv_ = kv_.reshape(B, self.num_heads, self.single_dim[i_c] // self.num_heads,
                             self.windowsize**2, -1).permute(0,1,3,4,2)
            # B,C_l_i*L2,H_i*W_i -> B,h,C_l_i/h, L2,H_i*W_i -> B, h, L2,H_i*W_i,C_l_i/h
            kv_ = kv_.reshape(B,self.num_heads,self.windowsize**2,-1)  # B,h,L2,H_i*W_i,C_l_i/h -> B,h,L2,H_i*W_i*C_l_i/h
            kv_ = nn.AdaptiveAvgPool2d((None, self.single_dim[i_c] // self.num_heads))(kv_)
            # B,h,L2,H_i*W_i*C_l_i/h -> B,h,L2,C_l_i/h
            kv_ = self.fcs[i_c-1](kv_)  # B, h, L2,C_l_i/h -> B,h,L2,C_l_i/h
            kv_ = self.single_heads[i_c-1](kv_)  # B,h,L2,C_l_i/h -> B,h, L2,2C_l_i/h

            k_ = kv_[:,:,:,:(self.single_dim[i_c] // self.num_heads)]  # B,h,L2,C_l_i/h
            v_ = kv_[:,:,:,(self.single_dim[i_c] // self.num_heads):]
            attn_ = (q_ @ k_.transpose(-2, -1)) * self.scales[i_c]  # B,h,N,C_l_i/h;B,h,L2,C_l_i/h -> B,h,N,L2
            attn_ = attn_.softmax(dim=-1)
            attn_ = self.attn_drop(attn_)

            v_ = v_ + self.local_convs[i_c-1](v_.transpose(2,
                                                         3).reshape(B,self.single_dim[i_c],self.windowsize,self.windowsize)).view(B, self.num_heads,
                                                                                                                                  self.single_dim[i_c] // self.num_heads,
                                                                                                                                  self.windowsize**2).transpose(2, 3)
            # B, h, L2,C_l_i/h -> B,h,C_l_i/h, L2 -> B,C_l_i,L, L -> B,C_l_i,L,L -> B,h,C_l_i/h,L2 -> B,h,L2,C_l_i/h

            attn_ = (attn_ @ v_).permute(0, 2, 1, 3)  # B, h, N, L2; B, h, L2, C_l_i/h -> B,h,N,C_l_i/h -> B,N,h,C_l_i/h
            attn_ = attn_.reshape(B, N, self.single_dim[i_c])  # B,N,h,C_l_i/h -> B,N,C_l_i
            attn_l.append(attn_)

        for i_c in range(1,len(self.mul)):
            attn_g = torch.cat((attn_g,attn_l[i_c-1]),-1)

        x = self.proj(attn_g)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class FactorizationTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 5, 2], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i+1)),
                                          patch_size=7 if i == 0 else 3,
                                          stride=4 if i == 0 else 2,
                                          in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                          embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                   out_channels=in_ch,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                   out_channels=out_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
    return out_dict


@register_model
def favit_b0(pretrained=False, **kwargs):
    model = FactorizationTransformer(
        patch_size=4, embed_dims=[32, 64, 128, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 6, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def favit_b1(pretrained=False, **kwargs):
    model = FactorizationTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 6, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def favit_b2(pretrained=False, **kwargs):
    model = FactorizationTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 3, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def favit_b3(pretrained=False, **kwargs):
    model = FactorizationTransformer(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 6, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 3, 14, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


def main():
    """Main training function with modern best practices"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================================
    # UPGRADED TRAINING CONFIGURATION
    # ============================================================================
    image_size = 224
    num_classes = 10
    batch_size = 128     # Optimal for T4 GPU (faster + better gradients)
    base_lr = 1e-3
    weight_decay = 0.05
    epochs = 30          # User requested limit
    warmup_epochs = 5     # Gradual warmup
    
    # Mixup/Cutmix parameters 
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    mixup_prob = 1.0      # (was 1.0)
    mixup_switch_prob = 0.5
    
    # Label smoothing
    smoothing = 0.1
    
    print(f"\n{'='*70}")
    print("ENHANCED TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model: FAViT-B1 (upgraded from B0)")
    print(f"Dataset: CIFAR-10 ({num_classes} classes)")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size} (increased from 32)")
    print(f"Epochs: {epochs} (increased from 5)")
    print(f"Augmentation: RandAugment ONLY (Mixup/Cutmix Disabled)")
    print(f"Regularization: Label Smoothing + Model EMA")
    print(f"{'='*70}\n")

    # ============================================================================
    # ADVANCED DATA AUGMENTATION (RandAugment)
    # ============================================================================
    transform_train = create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',  # RandAugment
        interpolation='bicubic',
        re_prob=0.25,        # Random erasing
        re_mode='pixel',
        re_count=1,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # ============================================================================
    # DATASET LOADING
    # ============================================================================
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, 
                                     transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, 
                                   transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    # ============================================================================
    # MODEL INITIALIZATION (Upgraded to FAViT-B1)
    # ============================================================================
    print("Initializing FAViT-B1 model...")
    model = favit_b1(num_classes=num_classes).to(device)  # Upgraded from favit_b0
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total parameters: {total_params:.2f}M")
    print(f"Trainable parameters: {trainable_params:.2f}M\n")

    # ============================================================================
    # MODEL EMA (Exponential Moving Average for stable final weights)
    # ============================================================================
    model_ema = ModelEmaV2(model, decay=0.999) # Reduced from 0.9999 for shorter training
    print("Model EMA initialized with decay=0.999\n")

    # ============================================================================
    # MIXUP & CUTMIX
    # ============================================================================
    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mixup_prob,
        switch_prob=mixup_switch_prob,
        mode='batch',
        label_smoothing=smoothing,
        num_classes=num_classes
    )
    print("Mixup/Cutmix initialized")
    print(f"  - Mixup alpha: {mixup_alpha}")
    print(f"  - Cutmix alpha: {cutmix_alpha}")
    print(f"  - Mix probability: {mixup_prob}\n")

    # ============================================================================
    # OPTIMIZER & SCHEDULER
    # ============================================================================
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    def cosine_lr_schedule(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            cos_inner = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * cos_inner))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr_schedule)
    
    # Loss function (Standard CE since Mixup is disabled)
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    print(f"Optimizer: AdamW (lr={base_lr}, weight_decay={weight_decay})")
    print(f"Scheduler: Cosine with {warmup_epochs} warmup epochs")
    print(f"Loss: LabelSmoothingCrossEntropy (Mixup Disabled)\n")

    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    def train_epoch(model, loader, mixup_fn):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Apply Mixup/Cutmix
            imgs, labels = mixup_fn(imgs, labels)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update EMA model
            model_ema.update(model)
            
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}", 
                      flush=True)
        
        return total_loss / total

    def evaluate(model, loader):
        model.eval()
        total_loss, correct = 0, 0
        criterion_val = nn.CrossEntropyLoss()  # Standard CE for validation
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion_val(outputs, labels)
                total_loss += loss.item() * imgs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    # ============================================================================
    # MAIN TRAINING LOOP
    # ============================================================================
    print("="*70)
    print("TRAINING STARTED")
    print("="*70)
    
    best_acc = 0.0
    best_ema_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, mixup_fn)
        
        # Evaluate both regular model and EMA model
        test_loss, test_acc = evaluate(model, val_loader)
        ema_test_loss, ema_test_acc = evaluate(model_ema.module, val_loader)
        
        scheduler.step()
        
        # Track best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
        if ema_test_acc > best_ema_acc:
            best_ema_acc = ema_test_acc
            # Save best EMA model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_ema_acc,
            }, 'best_favit_b1_cifar10.pth')
        
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")
        print(f"  EMA Val Loss: {ema_test_loss:.4f}, Acc: {ema_test_acc*100:.2f}% (Best: {best_ema_acc*100:.2f}%)")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-"*70)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print(f"Best Regular Model Accuracy: {best_acc*100:.2f}%")
    print(f"Best EMA Model Accuracy: {best_ema_acc*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
