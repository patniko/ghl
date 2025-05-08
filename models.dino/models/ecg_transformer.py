import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class PatchEmbedding(nn.Module):
    """Convert ECG signal into patch embeddings."""
    def __init__(self, in_channels=1, embed_dim=768, patch_size=32, overlap=8):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=self.stride)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        x = self.proj(x)     # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class MetadataEmbedding(nn.Module):
    """Embed metadata information."""
    def __init__(self, metadata_dim=64):
        super().__init__()
        # Age: normalized to [0,1]
        self.age_embed = nn.Linear(1, metadata_dim)
        # Sex: binary (0/1)
        self.sex_embed = nn.Embedding(2, metadata_dim)
        # Height/Weight: normalized
        self.height_embed = nn.Linear(1, metadata_dim)
        self.weight_embed = nn.Linear(1, metadata_dim)
        # Combine all metadata
        self.combine = nn.Linear(4 * metadata_dim, metadata_dim)
        
    def forward(self, metadata):
        # metadata should be dict with keys: age, sex, height, weight
        age_emb = self.age_embed(metadata['age'].unsqueeze(-1))
        sex_emb = self.sex_embed(metadata['sex'].long())
        height_emb = self.height_embed(metadata['height'].unsqueeze(-1))
        weight_emb = self.weight_embed(metadata['weight'].unsqueeze(-1))
        
        combined = torch.cat([age_emb, sex_emb, height_emb, weight_emb], dim=-1)
        return self.combine(combined)

class ECGTransformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., patch_size=32, overlap=8, 
                 num_patches=128, dropout=0.1, use_metadata=False, metadata_dim=64):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size, overlap)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Metadata handling
        self.use_metadata = use_metadata
        if use_metadata:
            self.metadata_embed = MetadataEmbedding(metadata_dim)
            self.metadata_proj = nn.Linear(metadata_dim, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.init_weights()
        
    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x, metadata=None):
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add metadata if available
        if self.use_metadata and metadata is not None:
            metadata_emb = self.metadata_embed(metadata)
            metadata_emb = self.metadata_proj(metadata_emb).unsqueeze(1)  # (batch_size, 1, embed_dim)
            x = x + metadata_emb
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]  # Return only cls token


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x):
        x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        x, _ = self.attn(x, x, x)
        return x.transpose(0, 1)

    def _ff_block(self, x):
        return self.mlp(x)