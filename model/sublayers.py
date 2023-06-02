import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['AttentionBlock', "ResnetBlock", "DownSample", "UpSample", "SinusoidalPositionEmbeddings"]


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Inputs:
            q: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k)
            k: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k) or (sentence_num, n_head, enc_token_num, d_k)
            v: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k) or (sentence_num, n_head, enc_token_num, d_k)
            mask: (sentence_num, 1, enc_token_num, 1) or (sentence_num, 1, dec_token_num, dec_token_num) or ()
        Ouputs:
            output: (sentence_num, n_head, enc_token_num, d_v) or (sentence_num, n_head, dec_token_num, d_v)
            attn: (sentence_num, n_head, enc_token_num, enc_token_num) or (sentence_num, n_head, dec_token_num, dec_token_num) or (sentence_num, n_head, dec_token_num, enc_token_num)
        """
        #    (sentence_num, n_head, enc_token_num, d_k) & (sentence_num, n_head, d_k, enc_token_num) -> (sentence_num, n_head, enc_token_num, enc_token_num)
        # or (sentence_num, n_head, dec_token_num, d_k) & (sentence_num, n_head, d_k, dec_token_num) -> (sentence_num, n_head, dec_token_num, dec_token_num)
        # or (sentence_num, n_head, dec_token_num, d_k) & (sentence_num, n_head, d_k, enc_token_num) -> (sentence_num, n_head, dec_token_num, enc_token_num)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        #    (sentence_num, n_head, enc_token_num, enc_token_num) & (sentence_num, n_head, enc_token_num, d_v) -> (sentence_num, n_head, enc_token_num, d_v)
        # or (sentence_num, n_head, dec_token_num, dec_token_num) & (sentence_num, n_head, dec_token_num, d_v) -> (sentence_num, n_head, dec_token_num, d_v)
        # or (sentence_num, n_head, dec_token_num, enc_token_num) & (sentence_num, n_head, enc_token_num, d_v) -> (sentence_num, n_head, dec_token_num, d_v)
        output = torch.matmul(attn, v)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        Inputs:
            n_head: 注意力头的个数
            d_model: 等于embedding_size
            d_k: key laten dimension
            d_v: value laten dimension
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc   = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        """
        Inputs:
            q: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size) 
            k: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size) 
            v: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size)  
            mask: Encoder-> (sentence_num, enc_token_num, 1) or Decoder-> (sentence_num, dec_token_num, dec_token_num)
        Outputs:
            q: (sentence_num, enc_token_num, embedding_size) or (sentence_num, dec_token_num, embedding_size)
            attn: (sentence_num, n_head, enc_token_num, enc_token_num) 
                or (sentence_num, n_head, dec_token_num, dec_token_num) 
                or (sentence_num, n_head, dec_token_num, enc_token_num)
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q: (sentence_num, enc_token_num, embedding_size) -> (sentence_num, enc_token_num, n_head * d_k) -> (sentence_num, enc_token_num, n_head, d_k)
        # or (sentence_num, dec_token_num, embedding_size) -> (sentence_num, dec_token_num, n_head * d_k) -> (sentence_num, dec_token_num, n_head, d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # k: (sentence_num, enc_token_num, embedding_size) -> (sentence_num, enc_token_num, n_head * d_k) -> (sentence_num, enc_token_num, n_head, d_k)
        # or (sentence_num, dec_token_num, embedding_size) -> (sentence_num, dec_token_num, n_head * d_k) -> (sentence_num, dec_token_num, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) 
        # v: (sentence_num, enc_token_num, embedding_size) -> (sentence_num, enc_token_num, n_head * d_v) -> (sentence_num, enc_token_num, n_head, d_v)
        # or (sentence_num, dec_token_num, embedding_size) -> (sentence_num, dec_token_num, n_head * d_v) -> (sentence_num, dec_token_num, n_head, d_v)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) 

        # Transpose for attention dot product: b x n x lq x dv
        # q: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k)
        # k: (sentence_num, n_head, enc_token_num, d_k) or (sentence_num, n_head, dec_token_num, d_k)
        # v: (sentence_num, n_head, enc_token_num, d_v) or (sentence_num, n_head, dec_token_num, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # Encoder: (sentence_num, enc_token_num, 1) -> (sentence_num, 1, enc_token_num, 1)
            # or 
            # Decoder: (sentence_num, dec_token_num, dec_token_num) -> (sentence_num, 1, dec_token_num, dec_token_num)
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # q:    (sentence_num, n_head, enc_token_num, d_v) or (sentence_num, n_head, dec_token_num, d_v)
        # attn: (sentence_num, n_head, enc_token_num, enc_token_num) or (sentence_num, n_head, dec_token_num, enc_token_num)
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # q: (sentence_num, n_head, enc_token_num, d_v) -> (sentence_num, enc_token_num, n_head, d_v) -> (sentence_num, enc_token_num, n_head*d_v)
        # or q: (sentence_num, n_head, dec_token_num, d_v) -> (sentence_num, dec_token_num, n_head, d_v) -> (sentence_num, dec_token_num, n_head*d_v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q: (sentence_num, token_num, n_head*d_v) -> (sentence_num, token_num, embedding_size)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.mhsa = MultiHeadAttention(n_head=4, d_model=self.channels, d_k=self.channels//4, d_v=self.channels//4)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]
        return x + h
    

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        # add in timestep embedding
        h += self.dense_1(self.act_fn(t))[:, :, None, None]

        # group 3
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + self.match_input(x)
        h = self.attention(h)

        return h
    

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)