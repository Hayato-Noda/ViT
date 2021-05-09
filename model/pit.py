from math import sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.layers.depthwiseconv2d import Pool
from model.layers.pittransformer import Transformer

# helpers

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# main class

class PiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth, tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        
        heads = cast_tuple(heads, len(depth))
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = patch_size // 2),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, dim)
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            layers.append(Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout))

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(
            *layers)
            
        self.to_latent = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.layers(x)
        x = x[:, 0]

        return self.to_latent(x)
