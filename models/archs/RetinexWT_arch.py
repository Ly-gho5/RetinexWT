import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from thop import profile
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import numpy as np
from WTtransform import WTtransform

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # self.depth_conv = nn.Conv2d(
        #     n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.depth_conv = WTtransform(
            n_fea_middle, n_fea_middle, kernel_size=3, bias=True, wt_levels=3)

        self.conv3 = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=1)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        print('111')
        print(x_1.shape)
        illu_fea = self.depth_conv(x_1)
        print('222')
        print(illu_fea.shape)
        illu_fea = self.conv3(illu_fea)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class MDTA(nn.Module):
    def __init__(self,dim,dim_head,heads):
        '''
        Restormer: https://github.com/swz30/Restormer
        Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang;
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5728-5739
        '''
        super(MDTA,self).__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads,1,1))

        self.qkv = nn.Conv2d(dim,dim_head*heads*3,1,bias=False)
        self.qkv_dwconv = nn.Conv2d(dim_head*heads*3,dim_head*heads*3,3,1,1,groups=dim_head*heads*3,
                                    bias=False)
        self.project_out = nn.Conv2d(dim_head*heads,dim,1,bias=False)



    def forward(self,x_in,illu_fea_trans):
        '''

        Args:
            x_in: [b,c,h,w]
            illu_fea_trans: [b,c,h,w]

        Returns:[b,c,h,w]

        '''
        b,c,h,w = x_in.shape
        qkv = self.qkv_dwconv(self.qkv(x_in))
        q,k,v = qkv.chunk(3,dim=1)
        #[b,h,d,n]
        q,k,v,illu_fea_trans = map(lambda t:rearrange(t,'b (n d) h w -> b n d (h w)',n=self.num_heads),(q,k,v,illu_fea_trans))
        q = F.normalize(q,dim=-1)
        k = F.normalize(k,dim=-1)

        attn = (q @ k.transpose(-2,-1))*self.temperature#[b,h,d,d]
        attn = attn.softmax(dim=-1)

        v = v*illu_fea_trans

        out = attn @ v#[b,h,d,n]
        out = rearrange(out,'b n d (h w)->b (n d) h w',n=self.num_heads,h=h,w=w)
        out = self.project_out(out)#[b,c,h,w]

        return out

class BiasFree_LayerNorm(nn.Module):
    def __init__(self,normalized_shapes):
        super(BiasFree_LayerNorm,self).__init__()
        if isinstance(normalized_shapes,numbers.Integral):
            normalized_shapes = (normalized_shapes,)
        normalized_shapes = torch.Size(normalized_shapes)
        self.weight = nn.Parameter(torch.ones(normalized_shapes))
        self.normalized_shapes = normalized_shapes
    def forward(self,x):
        sigma = x.var(-1,keepdim=True,unbiased=False)
        return x/torch.sqrt(sigma+1e-5)*self.weight

class LayerNorm(nn.Module):
    def __init__(self,dim):
        super(LayerNorm,self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self,x):
        h,w = x.shape[-2:]
        x = rearrange(x,'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c->b c h w',h=h,w=w)
        return x


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        print(x_in.shape)
        print(illu_fea_trans.shape)
        x_in = x_in.permute(0, 2, 3, 1)  # [b,h,w,c]
        illu_fea_trans = illu_fea_trans.permute(0, 2, 3, 1)  # [b,h,w,c]
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        print(v_inp.shape)
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        print(x.shape)
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        print(out.shape)
        out = out.permute(0, 3, 1, 2)
        return out

class FDFN(nn.Module):
    def __init__(self,dim,ratio=2.66):
        super(FDFN,self).__init__()

        hidden_features = int(dim*ratio)
        self.project_in = nn.Conv2d(dim,hidden_features*2,1,bias=False)
        self.dwconv = nn.Conv2d(hidden_features*2,hidden_features*2,kernel_size=3,stride=1,padding=1,groups=hidden_features*2,bias=False)

        self.project_out = nn.Conv2d(hidden_features,dim,1,bias=False)

    def forward(self,x):
        '''

        Args:
            x: [b,c,h,w]

        Returns: [b,c,h,w]

        '''
        x  = self.project_in(x)
        x1,x2 = self.dwconv(x).chunk(2,dim=1)
        x = F.gelu(x1)*x2
        x = self.project_out(x)
        return x

class FDFN_1(nn.Module):
    def __init__(self, dim,ratio=2.66):
        super(FDFN_1, self).__init__()

        hidden_features = int(dim*ratio)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=False)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.groups = 12
        self.dim_conv = dim // (2*self.groups)
        self.dim_untouched = dim//self.groups - self.dim_conv 
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, groups=self.dim_conv, bias=False)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
                
        x = x.reshape(b * self.groups, -1, h, w)

        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = x.reshape(b, -1, h, w)

        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x = x1 * x2
        
        x = self.project_out(x)
        return x


# class FeedForward(nn.Module):
    # def __init__(self, dim, mult=4):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
    #         GELU(),
    #         nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
    #                   bias=False, groups=dim * mult),
    #         GELU(),
    #         nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
    #     )

    # def forward(self, x):
    #     """
    #     x: [b,h,w,c]
    #     return out: [b,h,w,c]
    #     """
    #     out = self.net(x.permute(0, 3, 1, 2))
    #     return out.permute(0, 2, 3, 1)

#===============================================================================================================================================================================================================

class PolaLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, dim_heads, heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 kernel_size=5, alpha=4):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.dim_heads = dim_heads
        self.num_heads = heads

        self.qkvg = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwc = nn.Conv2d(in_channels=self.dim_heads, out_channels=self.dim_heads, kernel_size=kernel_size,
                             groups=self.dim_heads, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.dim_heads)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))
        # print('Linear Attention window{} f{} kernel{}'.
        #       format(window_size, alpha, kernel_size))

    def forward(self, x_in, illu_fea, attn_mask):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, l, c = x_in.shape
        qkv = self.qkvg(x_in).reshape(b, l, 4, c).permute(2, 0, 1, 3) #[3, b, h, l, d]
        q, k, v, g = torch.unbind(qkv, dim=0) #[b,h,l,d]
        k = k + self.positional_encoding
        kernel_function = nn.ReLU()
        # illu_attn = rearrange(illu_fea, 'b l (h d)->b h l d', h=self.num_heads)
        v = v * illu_fea
        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(b, l, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(b, l, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(b, l, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (l ** -0.5)) @ (v1 * (l ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (l ** -0.5)) @ (v2 * (l ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(b, l, c)

        H = W = int(l ** 0.5)
        v = v.reshape(b * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(b, c, l).permute(0, 2, 1)

        x = x + v
        x = x * g
        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(x.shape)

        return x

def window_partion(x,window_size:int):#[b,h,w,c]->[N,ws,ws,c]
    B,H,W,C = x.shape
    x = x.view(B,H//window_size,window_size,W//window_size,window_size,C)# [b,h/win,win,w/win,win,c]
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows#[B*num_windows,mh,mw,c]

def window_reverse(windows,window_size,h,w):#[N,ws,ws,c]->[b,h,w,c]
    b = int(windows.shape[0]/(h*w/window_size/window_size))
    x = windows.view(b,h//window_size,w//window_size,window_size,window_size,-1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(b,h,w,-1)
    return x

class Windowattention(nn.Module):
    def __init__(self,dim,dim_heads,heads,window_size,qkv_bias,attn_drop,proj_drop):
        super(Windowattention,self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.dim_heads = dim_heads
        self.scale = dim_heads**-0.5
        self.num_heads = heads

        #相对位置查询表[(2M-1*2M-1),h]
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1),heads))
        #制作相对位置成对矩阵
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        relative_flatten = torch.flatten(coords, 1)
        relative_coords = relative_flatten[:, :, None] - relative_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)  # [m2,m2]

        self.qkv = nn.Linear(dim,dim_heads*heads*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_heads*heads,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x_in,illu_fea,attn_mask):
        '''

        Args:
            x_in: [N,ws*ws,c]
            illu_fea: [N,ws*ws,c]
            attn_mask: [N,s*s,s*s]

        Returns:[N,ws*ws,c]

        '''
        b,l,c =x_in.shape
        qkv = self.qkv(x_in).reshape(b, l, 3, self.num_heads, self.dim_heads).permute(2, 0, 3, 1, 4)#[3,_b,h,l,d]
        q, k, v = torch.unbind(qkv, dim=0)#[_b,h,l,d]
        q = q*self.scale
        attn = (q @ k.transpose(-2,-1))#[_b,h,l,l]
        illu_attn = rearrange(illu_fea,'b l (h d)->b h l d',h=self.num_heads)
        v = v*illu_attn

        #相对位置编码
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0]*self.window_size[1],self.window_size[0]*self.window_size[1],-1)
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous().unsqueeze(0)
        attn = attn+relative_position_bias

        #sw-msa掩码不连续像素块的注意力得分
        if attn_mask is not None:
            nw = attn_mask.shape[0]
            attn = attn.view(b//nw,nw,self.num_heads,l,l)+attn_mask.unsqueeze(1).unsqueeze(0)#[1,nw,1,l,l]
            attn = attn.view(-1, self.num_heads, l, l)#[nw,h,l,l]
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x= (attn @ v).transpose(1,2).reshape(b,l,c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features,act_layer,drop):
        super(Mlp,self).__init__()
        out_features = in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features,in_features)
        self.drop2 = nn.Dropout(drop)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwinTransformerBlock1(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, dim_head, num_heads, window_size, shift_size, mlp_ratio,
                 qkv_bias, drop, attn_drops, norm_layer=nn.LayerNorm, drop_path=0., act_layer=nn.GELU,
                 alpha=4, kernel_size=5):
        super(SwinTransformerBlock1, self).__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)

        # Use PolaLinearAttention exclusively
        self.attn = PolaLinearAttention(dim=dim, dim_heads=dim_head, window_size=(self.window_size,self.window_size), heads=num_heads,
                                        qkv_bias=qkv_bias, attn_drop=attn_drops, proj_drop=drop, kernel_size=kernel_size, alpha=alpha)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, illu_fea, attn_mask=None):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, h, w, c]
        short_cut = x.reshape(b, h * w, c)  # 将图像展平为 [b, h*w, c]
        # print(short_cut.shape)
        x = self.norm1(x)
        # print(x.shape)
        illu_fea = illu_fea.permute(0, 2, 3, 1)  # [b, h, w, c]

        # Padding to make dimensions divisible by window_size
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # Padding for x
        illu_fea = F.pad(illu_fea, (0, 0, pad_l, pad_r, pad_t, pad_b))  # Padding for illu_fea

        _, hp, wp, _ = x.shape  # Updated h, w after padding

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_illu_fea = torch.roll(illu_fea, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_illu_fea = illu_fea
            attn_mask = None  # No mask in non-shifted windows

        # Window partitioning for both x and illu_fea
        x_window = window_partion(shifted_x, self.window_size)  # [N, ws, ws, c]
        x_window = x_window.view(-1, self.window_size * self.window_size, c)  # [N, ws*ws, c]

        illu_fea_window = window_partion(shifted_illu_fea, self.window_size)  # [N, ws, ws, c]
        illu_fea_window = illu_fea_window.view(-1, self.window_size * self.window_size, c)  # [N, ws*ws, c]

        # Applying PolaLinearAttention
        attn_windows = self.attn(x_window, illu_fea_window, attn_mask=attn_mask)  # [N, ws*ws, c]

        # Reversing window partitioning
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [b, hp, wp, c]

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding to get back to original size
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()


        # print(x.shape)
        x = x.view(b, h * w, c)
        # print('111')
        # print(x.shape)
        x = short_cut + x  # Residual connection

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # Apply MLP with residual

        # Restore original shape
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)  # [b, c, h, w]

        return x

#===============================================================================================================================================================================================================

class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, stride=1):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - stride) // 2
        # self.to_q = SepConv2d(dim, inner_dim, kernel_size, stride, pad)
        # self.to_k = SepConv2d(dim, inner_dim, kernel_size, stride, pad)
        # self.to_v = SepConv2d(dim, inner_dim, kernel_size, stride, pad)
        self.to_q = nn.Conv2d(dim, inner_dim, kernel_size=kernel_size, stride=stride, padding=pad, groups=dim)
        self.to_k = nn.Conv2d(dim, inner_dim, kernel_size=kernel_size, stride=stride, padding=pad, groups=dim)
        self.to_v = nn.Conv2d(dim, inner_dim, kernel_size=kernel_size, stride=stride, padding=pad, groups=dim)

    def forward(self, x, illu_fea):
        b, n, c = x.shape
        h = self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_qk = illu_fea
        attn_qk = rearrange(attn_qk, 'b (l w) c -> b c l w', l=l, w=w)
        q = self.to_q(attn_qk)
        k = self.to_k(attn_qk)

        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)

        # print(attn_kv)
        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, illu_fea):
        B_, N, C = x.shape

        N_kv = illu_fea.size(1)
        v = self.to_v(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        qk = self.to_qk(illu_fea).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = v[0]

        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
            # self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, illu_fea, mask=None):
        B_, N, C = x.shape

        q, k, v = self.qkv(x, illu_fea)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H * W, H * W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num

        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}" % (flops / 1e9))
        return flops

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, use_eca=True):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer(dim) if use_eca else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)

        # spatial restore
        x = self.dwconv(x)
        x = self.conv2(x)

        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear',
                 token_mlp='leff'):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp

        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                    token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, illu_fea):
        B, C, H, W = x.shape
        # print(x.shape)
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        # shortcut = x.view(B, -1, C)
        x = x.permute(0, 2, 3, 1)  # b h w c
        # print(x.shape)
        shortcut = x

        illu_fea = illu_fea.permute(0, 2, 3, 1)
        # x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.win_size - W % self.win_size) % self.win_size
        pad_b = (self.win_size - H % self.win_size) % self.win_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # [B,H,W,C]
        illu_fea = F.pad(illu_fea, (0, 0, pad_l, pad_r, pad_t, pad_b))  # [B,H,W,C]

        attn_mask = None
        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H + pad_b, W + pad_r, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = shift_attn_mask

        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_zero = torch.roll(illu_fea, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_zero = illu_fea

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        zero_windows = window_partition(shifted_zero, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        zero_windows = zero_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, illu_fea=zero_windows)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H + pad_b, W + pad_r)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :H, :W, :]
        # x = x.contiguous().view(B, H * W, C)
        # x = x.permute(0, 3, 1, 2)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x.permute(0, 3, 1, 2) + self.drop_path(self.mlp(self.norm2(x)))
        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        del attn_mask, zero_windows
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops





class IGAB(nn.Module):
    def __init__(self,dim,dim_head,heads=8,num_blocks=2,window_size=7,qkv_bias=True,drop=0.1,mlp_ratio=4,ratio=2.66,
                 attn_drops=0.1,norm_layer=nn.LayerNorm, qk_scale=None, attn_drop=0, token_projection='linear', token_mlp='ffn'):
        super(IGAB,self).__init__()
        self.window_size = window_size
        self.shifted_size = window_size // 2
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                                                # MDTA(dim=dim,dim_head=dim_head,heads=heads),
                                              LayerNorm(dim),FDFN_1(dim),
                                            #   SwinTransformerBlock1(dim=dim, dim_head=dim_head, num_heads=heads,
                                            #                         window_size=window_size, mlp_ratio=mlp_ratio,
                                            #                         qkv_bias=qkv_bias, drop=drop, attn_drops=attn_drops,
                                            #                         norm_layer=norm_layer,
                                            #                         shift_size=0, alpha=4, kernel_size=5),
                                              SwinTransformerBlock1(dim=dim, dim_head=dim_head, num_heads=heads,
                                                                    window_size=window_size, mlp_ratio=mlp_ratio,
                                                                    qkv_bias=qkv_bias, drop=drop, attn_drops=attn_drops,
                                                                    norm_layer=norm_layer,
                                                                    shift_size=self.shifted_size, alpha=4, kernel_size=5),
                                                LeWinTransformerBlock(dim=dim, num_heads=heads, win_size=window_size,
                                                                    shift_size=0,
                                                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                                    qk_scale=qk_scale,
                                                                    drop=drop, attn_drop=attn_drop,
                                                                    drop_path=0.,
                                                                    norm_layer=norm_layer,
                                                                    token_projection=token_projection,
                                                                    token_mlp=token_mlp)                                                                  
                                              ]))

    def create_mask(self, x):
        '''
        仅在移位attention中被调用
        Args:
            x:[b,c,h,w]

        Returns: [N,w*w,w*w]

        '''
        b, h, w, c  = x.shape
        hp = int(np.ceil(h / self.window_size)) * self.window_size
        wp = int(np.ceil(w / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)  # [1,hp,wp,1]
        #
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shifted_size),
                    slice(-self.shifted_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shifted_size),
                    slice(-self.shifted_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_window = window_partion(img_mask, window_size=self.window_size)  # [(h/s)*(w/s),s,s,c]
        mask_window = mask_window.view(-1, self.window_size * self.window_size)  # [N,s*s]展平操作
        attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)  # [N,1,s*s]-[N,s*s,1]=[N,s*s,s*s]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, illu_fea):
        '''

        Args:
            x: [b,c,h,w]
            illu_fea: [b,c,h,w]

        Returns:[b,c,h,w]

        '''
        # 1.创建蒙版判别连续像素块
        b, c, h, w = x.shape
        attn_mask = self.create_mask(x.permute(0, 2, 3, 1))  # [N,s*s,s*s]
        for (ln,ff,swmsa, swmsa1) in self.blocks:
            # x = x+mdtn(x,illu_fea)
            # print(x.shape)
            x = x+ff(ln(x))
            print(x.shape)
            # x = wmsa(x, illu_fea, attn_mask=None)
            # print(x.shape)
            x = swmsa(x, illu_fea, attn_mask=attn_mask)
            print(x.shape)
            x = swmsa1(x, illu_fea)
            print(x.shape)
        return x


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class Retinexwt_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(Retinexwt_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        print(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img

    def forward(self, img):
        # img:        b,c=3,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


class Retinexwt(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 1, 1]):
        super(Retinexwt, self).__init__()
        self.stage = stage

        modules_body = [
            Retinexwt_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2,
                                       num_blocks=num_blocks)
            for _ in range(stage)]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out


