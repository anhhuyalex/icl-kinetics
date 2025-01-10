import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, x_dim, mlp_dim) -> None:
        super().__init__()   
       
        self.mlp = nn.Sequential(
            nn.Linear(x_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.to_embedding(x) # shape: (B, D)
        
        # if self.positional_encoding: x += self.pos_embedding
        output = self.mlp(x)
        # print ("output", output.shape, output)
        return output

class CausalTransformer(nn.Module):
    def __init__(self, x_dim, model_size, num_classes, mlp_dim,  device=None, dtype=None) -> None:

        super().__init__()   
        self.to_embedding = nn.Sequential(
            # nn.LayerNorm(x_dim),
            nn.Linear(x_dim, model_size),
            nn.LayerNorm(model_size),
        )
        self.qkv = nn.Linear(model_size, model_size * 3)

        self.mlp_dim = mlp_dim 
        if self.mlp_dim < 0:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Sequential(
                # nn.LayerNorm(model_size),
                nn.Linear(model_size, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, num_classes),
            )
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_embedding(x) # shape: (B, N, D)
        
        # if self.positional_encoding: x += self.pos_embedding
        q, k, v = self.qkv(x).chunk(3, dim=-1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        sa = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        x = x + sa # shape: (B, N, D)
        output = self.mlp(x[:, -1, :])
        # print ("output", output.shape, output)
        return output, q, k
 
    
class MultilayerCausalTransformer(nn.Module):
    def __init__(self, x_dim, 
                 num_attn_layers,
                 model_size, num_classes, mlp_dim, 
                 is_layer_norm = "True",
                 device=None, dtype=None) -> None:

        super().__init__()    
        if is_layer_norm == "True":
            self.to_embedding = nn.Sequential(
                nn.LayerNorm(x_dim),
                # nn.Linear(x_dim, model_size),
                # nn.LayerNorm(model_size),
            )
        else:
            self.to_embedding = nn.Identity()
        self.num_attn_layers = num_attn_layers
        self.qkv = nn.ModuleList([nn.Linear(x_dim, x_dim * 3, bias = False) for _ in range(num_attn_layers)])

        self.mlp_dim = mlp_dim 
        if self.mlp_dim < 0:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        # nn.LayerNorm(x_dim),
                        nn.Linear(x_dim, mlp_dim),
                        nn.ReLU(),
                        nn.Linear(mlp_dim, mlp_dim),
                        nn.ReLU(),
                        nn.Linear(mlp_dim, x_dim),
                        ) 
                    for _ in range(num_attn_layers-1)
                ]
            )
            self.mlpoutput =  nn.Sequential(
                # nn.LayerNorm(x_dim),
                nn.Linear(x_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_embedding(x) # shape: (B, N, D)
        # if self.positional_encoding: x += self.pos_embedding
        for _ in range(self.num_attn_layers):
            q, k, v = self.qkv[_](x).chunk(3, dim=-1)
            sa = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0) 
            x = x + sa # shape: (B, N, D)
            if _ < self.num_attn_layers - 1:
                x = self.mlp[_](x)
        # output = self.mlp(x[:, -1, :])
        # output = x[:, -1, :]
        output = self.mlpoutput(x[:, -1, :])
        # print ("output", output.shape, output)
        return output, output, sa

class CausalTransformer(nn.Module):
    def __init__(self, x_dim, model_size, num_classes, mlp_dim, 
                 is_layer_norm = "True",
                 device=None, dtype=None) -> None:

        super().__init__()    
        if is_layer_norm == "True":
            self.to_embedding = nn.Sequential(
                nn.LayerNorm(x_dim),
                # nn.Linear(x_dim, model_size),
                # nn.LayerNorm(model_size),
            )
        else:
            self.to_embedding = nn.Identity()
        # self.qkv = nn.Linear(model_size, model_size * 3)
        self.qkv = nn.Linear(x_dim, x_dim * 3, bias = False)

        self.mlp_dim = mlp_dim 
        if self.mlp_dim < 0:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Sequential(
                # nn.LayerNorm(x_dim),
                nn.Linear(x_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, 1),
            )
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_embedding(x) # shape: (B, N, D)
        # if self.positional_encoding: x += self.pos_embedding
        q, k, v = self.qkv(x).chunk(3, dim=-1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        sa = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0) 
        # x = x + sa # shape: (B, N, D)
        output = self.mlp(x[:, -1, :] + sa[:, -1, :])
        # print ("output", output.shape, output)
        return output, output, sa 