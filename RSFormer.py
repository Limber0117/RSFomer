import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, hidden_size, trainable=True):
        super().__init__()
        self.embedding = nn.Embedding(max_length, hidden_size)
        self.trainable = trainable

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        if not self.trainable:
            with torch.no_grad():
                return self.embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return self.embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):

    def forward(self, q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1, seq_length=5000):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0

        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.attention = Attention()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        x, attn = self.attention(q, k, v, mask=None, dropout=self.dropout)
        attn = torch.mean(attn, dim=1)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.head_dim)
        return self.output_proj(x), attn

class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        result = sublayer(x)
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))  # layer_norm
        else:
            if isinstance(result, tuple):
                value1, value2 = result
                return self.norm(x + self.dropout(self.a * value1)), value2  # layer_norm
            else:
                value = result
            return self.norm(x + self.dropout(self.a * value))  # layer_norm


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, data_len, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout, data_len)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.res_conn1 = ResidualConnection(d_model, enable_res_parameter, dropout)
        self.res_conn2 = ResidualConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x):
        x, attn = self.res_conn1(x, lambda _x: self.attn.forward(_x))
        # x = self.skipconnect1(x, lambda _x: self.attn.forward(_x))
        x = self.res_conn2(x, self.ffn)
        return x, attn

class Mask():
    def __init__(self, encs, device, masking_ratio, ratio_highest_attention, data_shape, d_model):
        super(Mask, self).__init__()
        self.encs = encs
        self.device = device
        self.masking_ratio = masking_ratio
        self.ratio_highest_attention = ratio_highest_attention
        self.data_shape = data_shape
        self.d_model = d_model
        self.trunk_net = nn.Sequential(
            nn.Linear(self.data_shape[1], self.d_model),
            nn.LayerNorm(self.d_model),
        ).to(self.device)
    def attention_sampled_masking_heuristic(self, X, masking_ratio, ratio_highest_attention, instance_weights):
        # attention_weights = attention_weights.to('cpu')
        # instance_weights = torch.sum(attention_weights, axis = 1)
        res, index = instance_weights.topk(int(math.ceil(ratio_highest_attention * X.shape[1])))
        index = index.cpu().data.tolist()
        index2 = [random.sample(index[i], int(math.ceil(masking_ratio * X.shape[1]))) for i in range(X.shape[0])]
        return np.array(index2)

    def random_instance_masking(self, X, masking_ratio, ratio_highest_attention, instance_weights):
        indices = self.attention_sampled_masking_heuristic(X, masking_ratio, ratio_highest_attention, instance_weights)
        boolean_indices = torch.zeros(X.shape[0], X.shape[1], dtype=torch.bool, device=X.device)
        for i, index in enumerate(indices):
            boolean_indices[i, index] = True
        boolean_indices_masked = boolean_indices.unsqueeze(2).expand(-1, -1, X.shape[2])
        boolean_indices_unmasked = ~boolean_indices_masked
        X = torch.where(boolean_indices_unmasked, X, torch.tensor(0.0, device=X.device))
        X = X.float()
        return X

    def forward(self, x):
        with torch.no_grad():
            x_ = self.trunk_net(x)
            _, attn = self.encs(x_, pass_slicing=False)
            attn = torch.sum(attn, axis=1) - torch.diagonal(attn, offset=0, dim1=1, dim2=2)
        x = self.random_instance_masking(x, self.masking_ratio, self.ratio_highest_attention, attn)
        return x

class Encoder(nn.Module):
    """
    encoder in RSFormer
    """

    def __init__(self, slice_size, data_shape, d_encoder, attn_heads, enable_res_parameter, device,
                 stride, TENlayers, position_location, position_type):
        super(Encoder, self).__init__()
        self.stride = (stride, data_shape[1])
        self.slice_size = slice_size
        self.data_shape = data_shape
        self.device = device
        self.max_len = self.data_shape[0]
        self.position_location = position_location
        self.position_type = position_type

        self.temporal_slicing = nn.Conv1d(self.slice_size[1], d_encoder, kernel_size=self.slice_size[0],
                                          stride=self.stride[0])
        self.input_norm = nn.LayerNorm(d_encoder)
        if position_type == 'cond' or position_type == 'conv_static':
            self.position = nn.Conv1d(d_encoder, d_encoder, kernel_size=5, padding='same')
            self.a = nn.Parameter(torch.tensor(1.))
        elif position_type == 'relative':
            self.position = PositionalEncoding(self.max_len, d_encoder)
        else:
            self.position = PositionalEncoding(self.max_len, d_encoder, trainable=False)

        self.TRMs = nn.ModuleList([
            TransformerBlock(d_encoder, attn_heads, 4 * d_encoder, enable_res_parameter, data_shape[0]) for i in
            range(TENlayers)
        ])

    def forward(self, x, pass_slicing=True):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        if pass_slicing:
            x = self.temporal_slicing(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_norm(x)
        B, N, C = x.shape
        attn = torch.zeros(B, N, N, device=self.device)
        if self.position_location == 'top':
            if self.position_type == 'cond' or self.position_type == 'conv_static':
                x = x.transpose(2, 1)
                if self.position_type == 'cond':
                    x = x + self.position(x)
                else:
                    with torch.no_grad():
                        x = x + self.position(x)
                x = x.transpose(2, 1)
            elif self.position_type != 'none':
                x += self.position(x)
        for index, TRM in enumerate(self.TRMs):
            x, attn = TRM(x)
            if index == 1 and self.position_location == 'middle':
                if self.position_type == 'cond':
                    x = x.transpose(2, 1)
                    x = x + self.position(x)
                    x = x.transpose(2, 1)
                elif self.position_type != 'none':
                    x += self.position(x)
        return x, attn


class RSFormer(nn.Module):
    """
    RSFormer model
    """

    def __init__(self, args):
        super(RSFormer, self).__init__()
        attn_heads = args.attn_heads
        enable_res_parameter = args.enable_res_parameter
        num_class = args.num_class

        self.layers = args.layer
        self.device = args.device
        self.position = args.position_location
        self.pooling_type = args.pooling_type
        self.data_shape = args.data_shape
        self.d_encoder = args.hidden_size_per_layer
        self.slice_sizes = [(i, j) for i, j in zip(args.slice_per_layer, [self.data_shape[1]] + self.d_encoder)]
        self.stride = args.stride_per_layer
        self.TENlayer_per_layer = args.TENlayer_per_layer
        self.masking_ratio = args.masking_ratio
        self.ratio_highest_attention = args.ratio_highest_attention
        self.criterion_rec = torch.nn.MSELoss()
        self._form_data_shape()
        self.positionencoder = nn.Conv1d(self.d_encoder[0], self.d_encoder[0], kernel_size=5, padding='same')
        self.trunk_net = nn.Sequential(
            nn.Linear(self.data_shape[1], self.d_encoder[0]),
            nn.LayerNorm(self.d_encoder[0]),
        )
        self.encs = nn.ModuleList([
            Encoder(slice_size=self.slice_sizes[i], data_shape=self.data_shapes[i], d_encoder=self.d_encoder[i],
                       attn_heads=attn_heads, device=self.device, enable_res_parameter=enable_res_parameter,
                       stride=self.stride[i], TENlayers=self.TENlayer_per_layer[i],
                       position_location=self.position, position_type=args.position_type)
            for i in range(self.layers)
        ])
        self.mask = Mask(encs=self.encs[0], device=self.device, masking_ratio=self.masking_ratio,
                         ratio_highest_attention=self.ratio_highest_attention,
                         data_shape=self.data_shape, d_model=self.d_encoder[0])
        self.output = nn.Sequential(
            nn.Linear(self.data_shapes[-1][0] * self.d_encoder[-1], num_class),
            # nn.Sigmoid()
        ) if self.pooling_type == 'cat' else nn.Sequential(
            nn.Linear(self.d_encoder[-1], num_class),
            # nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)


    def _form_data_shape(self):
        self.data_shapes = []
        for i in range(self.layers):
            if not i:
                data_shape_pre = self.data_shape
            else:
                data_shape_pre = self.data_shapes[-1]
            len_raw = (data_shape_pre[0] - self.slice_sizes[i][0]) // self.stride[i] + 1
            self.data_shapes.append(
                (len_raw, self.d_encoder[i]))
        print(self.data_shapes)

    def forward(self, x, mode='train'):
        if mode == 'train':
            x = self.mask.forward(x)
        for Encs in self.encs:
            x, _ = Encs(x)

        if self.pooling_type == 'last_token':
            return self.output(x[:, -1, :])
        elif self.pooling_type == 'mean':
            return self.output(torch.mean(x, dim=1))
        elif self.pooling_type == 'cat':
            return self.output(x.view(x.shape[0], -1))
        else:
            return self.output(torch.max(x, dim=1)[0])

    def encode(self, x):
        for Encs in self.encs:
            x = Encs(x)
        if self.pooling_type == 'last_token':
            return x[:, -1, :]
        elif self.pooling_type == 'mean':
            return torch.mean(x, dim=1)
        elif self.pooling_type == 'cat':
            return x.view(x.shape[0], -1)
        else:
            return torch.max(x, dim=1)[0]
