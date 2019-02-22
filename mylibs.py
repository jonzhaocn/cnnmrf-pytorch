import torch
import torch.nn as nn
import torch.nn.functional as functional


class ContentLoss(nn.Module):
    """
    content loss layer
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None

    def forward(self, input):
        self.loss = functional.mse_loss(input, self.target)
        return input

    def update(self, target):
        """
        update target of content loss
        :param target:
        :return:
        """
        self.target = target.detach()


class StyleLoss(nn.Module):
    """
    style loss layer
    """
    def __init__(self, target, patch_size, mrf_style_stride, mrf_synthesis_stride, gpu_chunck_size, device):
        super(StyleLoss, self).__init__()
        self.patch_size = patch_size
        self.mrf_style_stride = mrf_style_stride
        self.mrf_synthesis_stride = mrf_synthesis_stride
        self.gpu_chunck_size = gpu_chunck_size
        self.device = device
        self.loss = None

        self.style_patches = self.patches_sampling(target.detach(), patch_size=self.patch_size, stride=self.mrf_style_stride)
        self.style_patches_norm = self.cal_patches_norm()
        self.style_patches_norm = self.style_patches_norm.view(-1, 1, 1)

    def update(self, target):
        """
        update target of style loss
        :param target:
        :return:
        """
        self.style_patches = self.patches_sampling(target.detach(), patch_size=self.patch_size,
                                                   stride=self.mrf_style_stride)
        self.style_patches_norm = self.cal_patches_norm()
        self.style_patches_norm = self.style_patches_norm.view(-1, 1, 1)

    def forward(self, input):
        """
        calculate mrf loss
        :param input: synthesis image
        :return:
        """
        synthesis_patches = self.patches_sampling(input, patch_size=self.patch_size, stride=self.mrf_synthesis_stride)
        max_response = []
        for i in range(0, self.style_patches.shape[0], self.gpu_chunck_size):
            i_start = i
            i_end = min(i+self.gpu_chunck_size, self.style_patches.shape[0])
            weight = self.style_patches[i_start:i_end, :, :, :]
            response = functional.conv2d(input, weight, stride=self.mrf_synthesis_stride)
            max_response.append(response.squeeze(dim=0))
        max_response = torch.cat(max_response, dim=0)

        max_response = max_response.div(self.style_patches_norm)
        max_response = torch.argmax(max_response, dim=0)
        max_response = torch.reshape(max_response, (1, -1)).squeeze()
        # loss
        loss = 0
        for i in range(0, len(max_response), self.gpu_chunck_size):
            i_start = i
            i_end = min(i+self.gpu_chunck_size, len(max_response))
            tp_ind = tuple(range(i_start, i_end))
            sp_ind = max_response[i_start:i_end]
            loss += torch.sum(torch.mean(torch.pow(synthesis_patches[tp_ind, :, :, :]-self.style_patches[sp_ind, :, :, :], 2), dim=[1, 2, 3]))
        self.loss = loss/len(max_response)
        return input

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

    def cal_patches_norm(self):
        """
        calculate norm of style image patches
        :return:
        """
        # norm of style image patches
        norm_array = torch.zeros(self.style_patches.shape[0])
        for i in range(self.style_patches.shape[0]):
            norm_array[i] = torch.pow(torch.sum(torch.pow(self.style_patches[i], 2)), 0.5)
        return norm_array.to(self.device)


class TVLoss(nn.Module):
    def __init__(self):
        """
        tv loss layer
        """
        super(TVLoss, self).__init__()
        self.loss = None

    def forward(self, input):
        image = input.squeeze().permute([1, 2, 0])
        r = (image[:, :, 0] + 2.12) / 4.37
        g = (image[:, :, 1] + 2.04) / 4.46
        b = (image[:, :, 2] + 1.80) / 4.44

        temp = torch.cat([r.unsqueeze(2), g.unsqueeze(2), b.unsqueeze(2)], dim=2)
        gx = torch.cat((temp[1:, :, :], temp[-1, :, :].unsqueeze(0)), dim=0)
        gx = gx - temp

        gy = torch.cat((temp[:, 1:, :], temp[:, -1, :].unsqueeze(1)), dim=1)
        gy = gy - temp

        self.loss = torch.mean(torch.pow(gx, 2)) + torch.mean(torch.pow(gy, 2))
        return input