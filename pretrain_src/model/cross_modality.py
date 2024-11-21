import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU '0' or '1'
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import Blip2Processor, Blip2Model
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
from accelerate import Accelerator
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
# accelerator = Accelerator()
# device = accelerator.device


# model_blip2 = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl",
#                                          cache_dir="/media/mlr_lab/325C37DE7879ABF2/prarabda/HF_Datasets")
# model_blip2.language_model = None  # Remove the language model head

# model_dinov2 = Dinov2Model.from_pretrained('facebook/dinov2-large',
#                                           cache_dir="/media/mlr_lab/325C37DE7879ABF2/prarabda/HF_Datasets")


# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                      std=[0.2470, 0.2435, 0.2616]),
# ])


# # Freeze the parameters of blip2 and dinov2 models
# for param in model_blip2.parameters():
#     param.requires_grad = False

# for param in model_dinov2.parameters():
#     param.requires_grad = False

# # Define the fully connected layer
# num_classes = 100
# fc = nn.Sequential(
#     nn.Linear(1024, 100),  # 512 input dimension, 100 output dimension (for CIFAR-100)
#     nn.LayerNorm(100)  # Apply LayerNorm after the fully connected layer
# )



# class SlotAttention(nn.Module):
#     def __init__(self, num_slots=5, dim=768, iters=5, eps=1e-8, hidden_dim=257):

#         super(SlotAttention, self).__init__()
#         self.num_slots = num_slots
#         self.dim = dim
#         self.iters = iters
#         self.eps = eps

#         # Slot initialization (learnable)
#         self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
#         self.slot_sigma = nn.Parameter(torch.randn(1, 1, dim))

#         # Linear layers for input transformation
#         self.to_q = nn.Linear(dim, dim)  # Query
#         self.to_k = nn.Linear(dim, dim)  # Key
#         self.to_v = nn.Linear(dim, dim)  # Value

#         # GRU for slot updates
#         self.gru = nn.GRUCell(dim, dim)

#         # MLP for slot refinement
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim)
#         )

#         # Slot attention projection
#         self.norm_slots = nn.LayerNorm(dim)
#         self.norm_mlp = nn.LayerNorm(dim)
#         self.norm_inputs = nn.LayerNorm(dim)

#     def forward(self, inputs):

#         batch_size, num_inputs, dim = inputs.shape

#         # Initialize slots from a normal distribution
#         slots = self.slot_mu + torch.randn(batch_size, self.num_slots, dim, device=inputs.device) * self.slot_sigma

#         # Apply layer normalization to inputs
#         inputs = self.norm_inputs(inputs)

#         for _ in range(self.iters):
#             # Step 1: Compute attention
#             slots_prev = slots
#             slots = self.norm_slots(slots)

#             # Compute queries, keys, and values
#             q = self.to_q(slots)  # (batch_size, num_slots, dim)
#             k = self.to_k(inputs)  # (batch_size, num_inputs, dim)
#             v = self.to_v(inputs)  # (batch_size, num_inputs, dim)

#             # Scale queries by sqrt(dim)
#             q = q / (dim ** 0.5)

#             # Attention logits: (batch_size, num_slots, num_inputs)
#             attn_logits = torch.einsum('bnd,bmd->bnm', q, k)

#             # Attention weights (softmax over inputs)
#             attn = F.softmax(attn_logits, dim=-1) + self.eps  # (batch_size, num_slots, num_inputs)

#             # Normalize attention across slots
#             attn = attn / attn.sum(dim=1, keepdim=True)

#             # Step 2: Compute updates for slots
#             updates = torch.einsum('bnm,bmd->bnd', attn, v)  # (batch_size, num_slots, dim)

#             # Step 3: Slot updates with GRU and MLP
#             slots = self.gru(
#                 updates.view(-1, dim), slots_prev.view(-1, dim)
#             ).view(batch_size, self.num_slots, dim)

#             slots = slots + self.mlp(self.norm_mlp(slots))

#         return slots

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

        # Initialize using Xavier
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        if self.w_q.bias is not None:
            nn.init.zeros_(self.w_q.bias)
            nn.init.zeros_(self.w_k.bias)
            nn.init.zeros_(self.w_v.bias)
            nn.init.zeros_(self.w_o.bias)

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout,mask=None):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout,mask=None)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # Using GELU instead of ReLU
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout=0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.layer_norm_self = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardBlock(d_model,dropout)
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.attention(x, x, x)  # Include mask if needed
        x = self.dropout(x)
        x = self.layer_norm_self(x + residual)

        residual=x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x=self.layer_norm_ff(x+residual)

        return x




class CrossModalityBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout=0.1) -> None:
        super().__init__()
        self.cross_attention_1 = MultiHeadAttentionBlock(d_model, h, dropout)
        self.cross_attention_2 = MultiHeadAttentionBlock(d_model, h, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.self_attention_1=SelfAttentionBlock(d_model,h,dropout)
        self.self_attention_2=SelfAttentionBlock(d_model,h,dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x_1, x_2, mask=None):

        residual_1= x_1
        residual_2=x_2  # Store the original query for residual connection
        x_1= self.cross_attention_1(x_1, x_2, x_2)  # Cross-attention: query attends to key-value
        x_1 = self.dropout(x_1)  # Apply dropout
        x_1 = self.layer_norm_1(x_1 + residual_1)  # Residual connection + Layer Norm

        x_2 = self.cross_attention_2(x_2 , residual_1,residual_1)
        x_2 = self.dropout(x_2)
        x_2 = self.layer_norm_2(x_2 + residual_2)   

        # Self-Attention Block with Residual Connection and Layer Norm
        x_1=self.self_attention_1(x_1)
        x_2=self.self_attention_2(x_2)
        
        return x_1, x_2
    
class AttentionFusion(nn.Module):
    def __init__(self, h_dim=1024,dropout=0.1,num_blocks=2,num_heads=8):
        super().__init__()
        self.emb1_self_attention=nn.ModuleList(
            [SelfAttentionBlock(d_model=h_dim, h=num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.emb2_self_attention=nn.ModuleList(
            [SelfAttentionBlock(d_model=h_dim, h=num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.emb_cross_attention=nn.ModuleList(
            [CrossModalityBlock(d_model=h_dim, h=num_heads, dropout=dropout) for _ in range(num_blocks)]
        )

        self.projection=nn.LazyLinear(h_dim)   
        self.layer_norm_combined = nn.LayerNorm(h_dim)


    def forward(self, emb1, emb2):
        # Self-attention
        for block in self.emb1_self_attention:
            emb1 = block(emb1)
        for block in self.emb2_self_attention:
            emb2 = block(emb2)
        
        # Cross-attention

        for block in self.emb_cross_attention:
            emb1,emb2 = block(emb1, emb2)

        # Combine and project
        combined_emb = torch.cat((emb1, emb2), dim=-1)
        combined_emb = self.projection(combined_emb)
        combined_emb=self.layer_norm_combined(combined_emb)

        return combined_emb

class CrossModality(nn.Module):
    def __init__(self,feature1_dim,feature2_dim,img_feature_dim):
        super().__init__()
        self.proj_blip2 = nn.Sequential(
            nn.Linear(feature1_dim, img_feature_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(img_feature_dim * 2, img_feature_dim),
            nn.LayerNorm(img_feature_dim)
        )

        self.proj_dinov2 = nn.Sequential(
            nn.Linear(feature2_dim, img_feature_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(img_feature_dim * 2, img_feature_dim),
            nn.LayerNorm(img_feature_dim)
        )
        self.cross_modal=AttentionFusion(h_dim=img_feature_dim)
        # self.classification_head = nn.Sequential(
        # nn.Linear(1024, 100),  # 512 input dimension, 100 output dimension (for CIFAR-100)
        # nn.LayerNorm(100)  # Apply LayerNorm after the fully connected layer
        # )


    def forward(self,blip2_emb,dino_emb):
        blip2_feature=self.proj_blip2(blip2_emb)
        dino_feature=self.proj_dinov2(dino_emb)

        fused_emb=self.cross_modal(blip2_feature,dino_feature)
        
        # fused_mean=torch.mean(fused_emb,dim=1)

        # logits =self.classification_head(fused_mean)

        return fused_emb
        
        
# model=CrossModality()

