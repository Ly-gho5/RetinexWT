import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torchvision.utils import save_image
from .util import wavelet
import os
import torchvision.utils as vutils
import cv2


class WTtransform(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTtransform, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x, save_curr_x_ll=True, save_dir="saved_images"):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        # print(curr_x_ll.shape)
        # # 如果需要保存 input 张量为热力图
        # if save_input:
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #
        #     # 将 input 张量转换为热力图
        #     input_np = curr_x_ll.cpu().detach().numpy()
        #     for i in range(curr_x_ll.shape[0]):
        #         for j in range(curr_x_ll.shape[1]):
        #             heatmap = cv2.normalize(input_np[i, j], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
        #                                     dtype=cv2.CV_8U)
        #             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #             cv2.imwrite(os.path.join(save_dir, f"input_heatmap_batch_{i}_channel_{j}.png"), heatmap)




        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            #保存低频子带
            # if save_curr_x_ll:
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     for batch_idx in range(curr_x_ll.shape[0]):
            #         for channel_idx in range(curr_x_ll.shape[1]):
            #             # 提取单个通道的数据
            #             tensor_to_save = curr_x_ll[batch_idx, channel_idx].unsqueeze(0)
            #             # 归一化到 [0, 1] 范围
            #             tensor_to_save = (tensor_to_save - tensor_to_save.min()) / (
            #                     tensor_to_save.max() - tensor_to_save.min())
            #             save_path = os.path.join(save_dir,
            #                                      f"curr_x_ll_level_{i}_batch_{batch_idx}_channel_{channel_idx}.png")
            #             save_image(tensor_to_save, save_path)



            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

            # # 保存低频子带（LL）为灰度特征图
            # if save_curr_x_ll:
            #     for batch_idx in range(curr_x_tag.shape[0]):
            #         for channel_idx in range(curr_x_tag.shape[1]):
            #             # 提取低频子带的数据
            #             tensor_to_save = curr_x_tag[batch_idx, channel_idx, 0].unsqueeze(0)
            #             # 归一化到 [0, 1] 范围
            #             tensor_to_save = (tensor_to_save - tensor_to_save.min()) / (tensor_to_save.max() - tensor_to_save.min())
            #             save_path = os.path.join(save_dir, f"curr_x_LL_level_{i}_batch_{batch_idx}_channel_{channel_idx}.png")
            #             save_image(tensor_to_save, save_path)
            #
            # # 保存三个高频子带（LH、HL、HH）为灰度特征图
            # if save_curr_x_ll:
            #     for batch_idx in range(curr_x_tag.shape[0]):
            #         for channel_idx in range(curr_x_tag.shape[1]):
            #             for subband_idx, subband_name in enumerate(['LH', 'HL', 'HH']):
            #                 # 提取单个高频子带的数据
            #                 tensor_to_save = curr_x_tag[batch_idx, channel_idx, subband_idx+1].unsqueeze(0)
            #                 # 归一化到 [0, 1] 范围
            #                 tensor_to_save = (tensor_to_save - tensor_to_save.min()) / (tensor_to_save.max() - tensor_to_save.min())
            #                 save_path = os.path.join(save_dir, f"curr_x_{subband_name}_level_{i}_batch_{batch_idx}_channel_{channel_idx}.png")
            #                 save_image(tensor_to_save, save_path)


        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
            # # 保存逆小波变换的结果为灰度特征图
            # if save_curr_x_ll:
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     for batch_idx in range(next_x_ll.shape[0]):
            #         for channel_idx in range(next_x_ll.shape[1]):
            #             # 提取单个通道的数据
            #             tensor_to_save = next_x_ll[batch_idx, channel_idx].unsqueeze(0)
            #             # 归一化到 [0, 1] 范围
            #             tensor_to_save = (tensor_to_save - tensor_to_save.min()) / (tensor_to_save.max() - tensor_to_save.min())
            #             save_path = os.path.join(save_dir, f"inverse_wavelet_level_{i}_batch_{batch_idx}_channel_{channel_idx}.png")
            #             save_image(tensor_to_save, save_path)


        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))

        # if save_curr_x_ll:
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     for batch_idx in range(x.shape[0]):
        #         for channel_idx in range(x.shape[1]):
        #             # 提取单个通道的数据
        #             tensor_to_save = x[batch_idx, channel_idx].unsqueeze(0)
        #             # 归一化到 [0, 1] 范围
        #             tensor_to_save = (tensor_to_save - tensor_to_save.min()) / (tensor_to_save.max() - tensor_to_save.min())
        #             save_path = os.path.join(save_dir, f"base_conv_feature_batch_{batch_idx}_channel_{channel_idx}.png")
        #             save_image(tensor_to_save, save_path)

        x = x + x_tag

        # if save_curr_x_ll:
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     for batch_idx in range(x.shape[0]):
        #         for channel_idx in range(x.shape[1]):
        #             # 提取单个通道的数据
        #             tensor_to_save = x[batch_idx, channel_idx].unsqueeze(0)
        #             # 归一化到 [0, 1] 范围
        #             tensor_to_save = (tensor_to_save - tensor_to_save.min()) / (tensor_to_save.max() - tensor_to_save.min())
        #             save_path = os.path.join(save_dir, f"base_conv_feature_batch_{batch_idx}_channel_{channel_idx}.png")
        #             save_image(tensor_to_save, save_path)

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
