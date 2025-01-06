import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # split into heads

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        
        attn = self.attend(dots) # attn = softmax(Q @ K^T / sqrt(d))
        out = torch.matmul(attn, v) # out = attn @ V
        out = rearrange(out, 'b h n d -> b n (h d)') # concat heads
        return self.to_out(out)
    
class TransformerEncoder(nn.Module):
    def __init__(self, model_size, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(model_size)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(model_size, heads = heads, dim_head = dim_head),
                FeedForward(model_size, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class Transformer(nn.Module):
    def __init__(self, x_dim, model_size, len_context, num_classes, depth, heads, mlp_dim, dim_head = 64, positional_encoding=False,
                 transformer_encoder = "pytorch"):
        """
        Adapting implementation from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
        """
        super().__init__()   
        self.to_x_embedding = nn.Sequential(
            nn.LayerNorm(x_dim),
            nn.Linear(x_dim, model_size),
            nn.LayerNorm(model_size),
        )
        # self.to_y_embedding = nn.Embedding(num_classes, model_size)
        # self.len_context = 2 * len_context + 1 # x,y from support set and x from query set

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
 
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.len_context, model_size)) # learnable positional embedding
        
        self.transformer_encoder = TransformerEncoder(model_size, depth, heads, dim_head, mlp_dim)
        self.linear_head = nn.Linear(model_size, num_classes)

    def forward(self, x):
        """
        :param x: (B, N, D) tensor of support set -- sequences of randomly projected images,
            where N is the sequence length, and D is the dimensionality of each token
        """
        seq = self.to_x_embedding(x)
        if self.positional_encoding: seq += self.pos_embedding

        output = self.transformer_encoder(seq)
        output = output.mean(dim = 1)
        output = self.linear_head(output)
        return output
       
class PytorchTransformer(nn.Module):
    def __init__(self, x_dim, model_size, len_context, num_classes, depth, nheads, mlp_dim, dim_head = 64, positional_encoding=False,
                 dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=True, norm_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()   
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}") 
        print ("x_dim", x_dim, "nheads", nheads)
        self.to_embedding = nn.Sequential(
            nn.LayerNorm(x_dim),
            nn.Linear(x_dim, model_size),
            nn.LayerNorm(model_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nheads, dim_feedforward=mlp_dim, dropout=dropout,
                                                    activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, 
                                                    norm_first=norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(model_size, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, norm=encoder_norm)

        self.linear_head = nn.Linear(model_size, num_classes)

        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, len_context, model_size)) # learnable positional embedding
        self._reset_parameters()

        self.x_dim = x_dim
        self.model_size = model_size 
        self.len_context = len_context 
        self.num_classes = num_classes 
        self.depth = depth 
        self.nheads = nheads
        self.mlp_dim = mlp_dim 
        self.dim_head = dim_head 
        self.batch_first = batch_first

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_embedding(x)
        if self.positional_encoding: x += self.pos_embedding
        memory = self.encoder(x)
        output = memory.mean(dim=1)
        output = self.linear_head(output)
        return output

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

class CausalTransformerOneMinusOne(nn.Module):
    def __init__(self, x_dim, model_size, num_classes, mlp_dim,  device=None, dtype=None) -> None:

        super().__init__()    
        self.qkv = nn.Linear(x_dim, x_dim * 3)

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
         
        # if self.positional_encoding: x += self.pos_embedding
        q, k, v = self.qkv(x).chunk(3, dim=-1) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x
        sa = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0) 
        # x = x + sa # shape: (B, N, D)
        output = self.mlp(x[:, -1, :] + sa[:, -1, :])
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

class CausalTransformerOneMinusOneEmbed(nn.Module):
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

class M2CausalTransformer(nn.Module):
    def __init__(self, x_dim, D, model_size, temperature, is_temperature_fixed, num_classes, mlp_dim,  device=None, dtype=None) -> None:

        super(M2CausalTransformer, self).__init__()
        self.D = D
        # self.to_embedding = nn.Sequential(
            # nn.LayerNorm(x_dim),
            # nn.Linear(x_dim, model_size),
            # nn.LayerNorm(model_size),
        # )
        # self.qkv = nn.Linear(model_size, model_size * 3)

        self.value_matrix = nn.Linear(x_dim, x_dim, bias=False)

        self.mlp_dim = mlp_dim 
        if self.mlp_dim < 0:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Sequential(
                nn.LayerNorm(x_dim),
                nn.Linear(x_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, num_classes),
            )

        # make a nn.Parameter called self.temperature
        if is_temperature_fixed == "False":
            self.invtemperature = nn.Parameter(torch.tensor(float(1.0 / temperature)))
        else:
            self.invtemperature = 1/temperature
         
 
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.to_embedding(x) # shape: (B, N, D+L)
        query = x[:, :, :self.D] # shape: (B, N, D)
        key = x[:, :, :self.D] # shape: (B, N, D)
        # if self.positional_encoding: x += self.pos_embedding
        v = self.value_matrix(x) # for head: Q = W_q @ x, K = W_k @ x, V = W_v @ x 

        L, S = query.size(-2), key.size(-2) # length of sequence of keys and queries
        # get mask 
        attn_bias = torch.zeros(L, S, dtype=x.dtype, device=x.device) 
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(x.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias=attn_bias.to(x.dtype) 
        scale_factor = 1.0 / (query.size(-1) ** 0.5) # scale factor for attention, defined as 1/sqrt(d_k)

        # compute attention
        query_last_token = query[:, [-1], :] # shape: (B, 1, D)
        key_T = key.transpose(-2, -1) # shape: (B, D, N)
        # print ("query_last_token", query_last_token.shape, "key_T", key_T.shape)
        attn_weight  = torch.softmax(query_last_token @ key_T * scale_factor * self.invtemperature + attn_bias[-1], dim=-1) # shape: (B, 1, N)
        # print("query_last_token @ key_T * scale_factor",query_last_token @ key_T * scale_factor)
        # attn_weight_einsum  = torch.softmax(einsum(query_last_token, key_T,'b m d, b d n -> b m n') * scale_factor * self.temperature + attn_bias[-1], dim=-1)
        sa = attn_weight @ v # shape: (B, 1, D) 
        # print ("sa", sa.shape, "x", x.shape)
        output = self.mlp(x[:, -1, :] + sa.squeeze(1))
        # output = self.mlp(sa.squeeze(1))
        # print ("output", output.shape, output[0])
        return output, v, attn_weight

class PhenomenologicalTransformer(nn.Module):
    def __init__(self, D, temperature, is_temperature_fixed, num_classes, mlp_dim) -> None:

        super(PhenomenologicalTransformer, self).__init__()
        self.D = D 

        self.phi = nn.Sequential(
            # nn.LayerNorm(D),
            nn.Linear(D, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
            # nn.Linear(mlp_dim, num_classes),
        )
        # self.final_linear = nn.Linear(1 + num_classes, num_classes)
        # self.Wx = nn.Linear(num_classes, num_classes)
        # self.Wl = nn.Linear(num_classes, num_classes)
        # self.w = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.tensor(float(0.0)))
        # make a nn.Parameter called self.temperature
        if is_temperature_fixed == "False":
            # self.invtemperature = nn.Parameter(torch.tensor(float(1.0 / temperature)))
            self.invtemperature = nn.Parameter(torch.randn(1)) 
        else:
            self.invtemperature = 1.0/temperature
         
 
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.to_embedding(x) # shape: (B, N, D+L)
        query = x[:, :, :self.D] # shape: (B, N, D)
        xt = query[:, -1, :] # shape: (B, D)
        phi_xt = self.phi(xt) # shape: (B, 1)
         

        # L, S = query.size(-2), key.size(-2) # length of sequence of keys and queries
        # get mask 
        # attn_bias = torch.zeros(L, S, dtype=x.dtype, device=x.device) 
        # temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(x.device)
        # attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        # attn_bias=attn_bias.to(x.dtype) 
        scale_factor = 1.0 # / (query.size(-1) ** 0.5) # scale factor for attention, defined as 1/sqrt(d_k)

        # compute attention
        key = x[:, :, :self.D] # shape: (B, N, D)
        query_last_token = query[:, [-1], :] # shape: (B, 1, D)
        key_T = key.transpose(-2, -1) # shape: (B, D, N)
        # print ("query_last_token", query_last_token.shape, "key_T", key_T.shape)
        attn_weight  = torch.softmax(query_last_token @ key_T * scale_factor * self.invtemperature, dim=-1) # shape: (B, 1, N)
        # print("query_last_token @ key_T * scale_factor",query_last_token @ key_T * scale_factor)
        # attn_weight_einsum  = torch.softmax(einsum(query_last_token, key_T,'b m d, b d n -> b m n') * scale_factor * self.temperature + attn_bias[-1], dim=-1)
        label = x[:, :, self.D:] # shape: (B, N, L)
        
        sa = attn_weight @ (self.w * label) # shape: (B, 1, L) 
        # print ("sa", sa.shape, "x", x.shape)
        # print ("label", label.shape, "sa", sa.shape)
        # phi_xt_sa = torch.cat((phi_xt, sa.squeeze(1)), dim=-1) # shape: (B, L+1)
        # output = self.final_linear(phi_xt_sa)
        # output = self.Wx(phi_xt) + self.Wl(sa.squeeze(1))
        output = phi_xt + sa.squeeze(1)
        return output, phi_xt, attn_weight