from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import torchvision.models as models


class CNNMRF(nn.Module):
    def __init__(self, style_image, content_image, device, content_weight=1, gpu_chunck_size=256, mrf_style_stride=2,
                 mrf_synthesis_stride=2):
        super(CNNMRF, self).__init__()
        # fine tune alpha_content to interpolate between the content and the style
        self.content_weight = content_weight
        # alpha_tv is fixed to 0.001
        self.tv_weight = 0.001
        self.patch_size = 3
        self.device = device
        self.gpu_chunck_size = gpu_chunck_size
        self.vgg = VGGNet().to(self.device).eval()
        self.mrf_style_stride = mrf_style_stride
        self.mrf_synthesis_stride = mrf_synthesis_stride
        self.mrf_layer_list = ['relu3_1', 'relu4_1']
        # style_patches, style_patches_norm, content_image_feature_map
        self.style_patches = dict()
        self.style_patches_norm = dict()
        self.content_image_feature_map = None
        self.update_params(style_image, content_image)

    def forward(self, synthesis):
        """
        calculate loss and return loss
        :param synthesis: synthesis image
        :return:
        """
        mrf_loss = 0
        for layer in self.mrf_layer_list:
            mrf_loss += self.cal_mrf_loss(synthesis, layer)

        content_loss = self.cal_content_loss(synthesis)
        tv_loss = self.cal_tv_loss(synthesis)
        loss = mrf_loss + self.content_weight * content_loss + self.tv_weight * tv_loss
        return loss

    def update_params(self, style_image, content_image):
        """
        extract style image patches and calculate norm of patches
        save content image feature map in vgg19's rule4_2 layer
        :param style_image:
        :param content_image:
        :return:
        """
        # style image patches
        for layer in self.mrf_layer_list:
            image_feature = self.vgg(style_image, layer=layer)
            self.style_patches[layer] = self.patches_sampling(image_feature, patch_size=self.patch_size, stride=self.mrf_style_stride)
            self.style_patches_norm[layer] = None

        "-----------------------"
        # content image feature map
        self.content_image_feature_map = self.vgg(content_image, layer='relu4_2')

    def cal_mrf_loss(self, synthesis, vgg_layer):
        """
        calculate mrf loss
        :param synthesis: synthesis image
        :param vgg_layer: vgg_layer name to extract feature map of image
        :return:
        """
        style_patches = self.style_patches[vgg_layer]
        style_patches_norm = self.style_patches_norm[vgg_layer]

        synthesis = self.vgg(synthesis, layer=vgg_layer)
        synthesis_patches = self.patches_sampling(synthesis, patch_size=self.patch_size, stride=self.mrf_synthesis_stride)
        max_response = []
        for i in range(0, style_patches.shape[0], self.gpu_chunck_size):
            i_start = i
            i_end = min(i+self.gpu_chunck_size, style_patches.shape[0])
            weight = style_patches[i_start:i_end, :, :, :]
            response = functional.conv2d(synthesis, weight, stride=self.mrf_synthesis_stride)
            max_response.append(response.squeeze())
        max_response = torch.cat(max_response, dim=0)

        if style_patches_norm is None:
            self.style_patches_norm[vgg_layer] = self.cal_style_patches_norm(vgg_layer, max_response.shape)
            style_patches_norm = self.style_patches_norm[vgg_layer]

        max_response = max_response * style_patches_norm
        max_response = torch.argmax(max_response, dim=0)
        max_response = torch.reshape(max_response, (1, -1)).squeeze()
        # loss
        loss = 0
        for i in range(0, len(max_response), self.gpu_chunck_size):
            i_start = i
            i_end = min(i+self.gpu_chunck_size, len(max_response))
            tp_ind = tuple(range(i_start, i_end))
            sp_ind = max_response[i_start:i_end]
            loss += torch.sum(torch.mean(torch.pow(synthesis_patches[tp_ind, :, :, :]-style_patches[sp_ind, :, :, :], 2), dim=[1, 2, 3]))
        loss = loss / len(max_response)
        return loss

    def cal_content_loss(self, synthesis):
        """
        calculate content loss
        :param synthesis:
        :return:
        """
        synthesis = self.vgg(synthesis, layer='relu4_2')
        loss = torch.mean(torch.pow(synthesis-self.content_image_feature_map, 2))
        return loss

    def cal_tv_loss(self, synthesis):
        """
        calculate tv loss
        :param synthesis:
        :return:
        """
        image = synthesis.squeeze().permute([1, 2, 0])
        r = image[:, :, 0]/0.229 + 0.485
        g = image[:, :, 1]/0.224 + 0.456
        b = image[:, :, 2]/0.225 + 0.406

        temp = torch.cat([r.unsqueeze(2), g.unsqueeze(2), b.unsqueeze(2)], dim=2)
        gx = torch.cat((temp[1:, :, :], temp[-1, :, :].unsqueeze(0)), dim=0)
        gx = gx - temp

        gy = torch.cat((temp[:, 1:, :], temp[:, -1, :].unsqueeze(1)), dim=1)
        gy = gy - temp

        loss = torch.mean(torch.pow(gx, 2)) + torch.mean(torch.pow(gy, 2))
        return loss

    def cal_style_patches_norm(self, layer, shape):
        """
        calculate norm of style image patches
        :param layer: relu3_1 or relu4_1
        :param shape: shape of norm array
        :return:
        """
        # norm of style image patches
        norm_array = torch.ones(shape)
        patches = self.style_patches[layer]
        for i in range(patches.shape[0]):
            norm_array[i] = norm_array[i] / self.cal_norm(patches[i])
        return norm_array.to(self.device)

    @staticmethod
    def cal_norm(x):
        """
        calculate norm of a tensor
        :param x:
        :return:
        """
        dim = tuple(np.arange(len(x.shape)))
        return torch.pow(torch.sum(torch.pow(x, 2), dim=dim), 0.5)

    def patches_sampling(self, image, patch_size, stride):
        """
        sampling patches form a image
        :param image:
        :param patch_size:
        :return:
        """
        h, w = image.shape[2:4]
        patches = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patches.append(image[:, :, i:i + patch_size, j:j + patch_size])
        patches = torch.cat(patches, dim=0).to(self.device)
        return patches


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x, layer):
        """
        extract specified layer output in vgg19 of a image
        :param x:
        :param layer:
        :return:
        """
        if layer == 'relu3_1':
            select = '11'
        elif layer == 'relu4_1':
            select = '20'
        elif layer == 'relu4_2':
            select = '22'
        else:
            raise ValueError('VGGNet forward: layer should be relu3_1 or relu4_1 or relu4_2')

        """Extract feature maps."""
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name == select:
                return x
