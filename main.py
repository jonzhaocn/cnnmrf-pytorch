from model import CNNMRF
import torch.optim as optim
from torchvision import transforms
import cv2
import argparse
import torch
import torchvision
import math
import os

"""
reference:
[1]. Li C, Wand M. Combining markov random fields and convolutional neural networks for image synthesis[C].
//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2479-2486.

[2]. https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/neural_style_transfer/main.py
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_synthesis_image(synthesis, denorm):
    """
    get synthesis image from tensor to numpy array
    :param synthesis: synthesis image tensor
    :param denorm: denorm transform
    :return:
    """
    image = synthesis.clone().squeeze()
    image = denorm(image).clamp_(0, 1)
    return image


def unsample_synthesis(height, width, synthesis, device):
    """
    unsample synthesis image to next level of training
    :param height: height of unsampled image
    :param width: width of unsampled image
    :param synthesis: synthesis image tensor to unsample
    :param device:
    :return:
    """
    # transform the tensor to numpy, and upsampled as a image
    synthesis = synthesis.cpu().detach().squeeze().permute([1, 2, 0]).numpy()
    synthesis = cv2.resize(synthesis, (width, height), cv2.INTER_LINEAR)
    synthesis = torch.Tensor(synthesis).permute([2, 0, 1]).unsqueeze(0).to(device)
    # finally, set requires grad, the node will be leaf node and require grad
    synthesis = synthesis.requires_grad_(True)
    return synthesis


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    "-------------------transform and denorm transform-----------------"
    # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406]
    # and std=[0.229, 0.224, 0.225].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])
    denorm_transform = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    "--------------read image------------------"
    if not os.path.exists(config.content_path):
        raise ValueError('file %s does not exist.' % config.content_path)
    if not os.path.exists(config.style_path):
        raise ValueError('file %s does not exist.' % config.style_path)
    content_image = cv2.imread(config.content_path)
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    style_image = cv2.imread(config.style_path)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    "resize image in several level for training"
    pyramid_content_image = []
    pyramid_style_image = []
    for i in range(config.num_res):
        content = cv2.resize(content_image, (math.ceil(content_image.shape[1]/pow(2, config.num_res-1-i)),
                                             math.ceil(content_image.shape[0]/pow(2, config.num_res-1-i))),
                             interpolation=cv2.INTER_LINEAR)

        style = cv2.resize(style_image, (math.ceil(style_image.shape[1]/pow(2, config.num_res-1-i)),
                                         math.ceil(style_image.shape[0]/pow(2, config.num_res-1-i))),
                           interpolation=cv2.INTER_LINEAR)
        pyramid_content_image.append(transform(content).unsqueeze(0).to(device))
        pyramid_style_image.append(transform(style).unsqueeze(0).to(device))
    "-----------------start training-------"
    global iter
    iter = 0
    synthesis = None
    # create cnnmrf model
    cnnmrf = CNNMRF(style_image=pyramid_style_image[0], content_image=pyramid_content_image[0], device=device,
                    content_weight=config.content_weight, gpu_chunck_size=config.gpu_chunck_size,
                    mrf_synthesis_stride=config.mrf_synthesis_stride, mrf_style_stride=config.mrf_style_stride).to(device)
    # Sets the module in training mode.
    cnnmrf.train()
    for i in range(0, config.num_res):
        # synthesis = torch.rand_like(content_image, requires_grad=True)
        if i == 0:
            # in lowest level init the synthesis from content resized image
            synthesis = pyramid_content_image[0].clone().requires_grad_(True).to(device)
        else:
            # in high level init the synthesis from unsampling the upper level synthesis
            synthesis = unsample_synthesis(pyramid_content_image[i].shape[2], pyramid_content_image[i].shape[3], synthesis, device)
            cnnmrf.update_params(style_image=pyramid_style_image[i], content_image=pyramid_content_image[i])
        # max_iter (int): maximal number of iterations per optimization step
        optimizer = optim.LBFGS([synthesis], lr=1, max_iter=config.max_iter)
        "--------------------"

        def closure():
            global iter
            optimizer.zero_grad()
            loss = cnnmrf(synthesis)
            if (iter+1) % config.max_iter == 0:
                loss.backward(retain_graph=False)
            else:
                loss.backward(retain_graph=True)
            # save image
            if (iter + 1) % config.sample_step == 0 or (iter + 1) == (i + 1) * config.max_iter:
                image = get_synthesis_image(synthesis, denorm_transform)
                torchvision.utils.save_image(image, 'result-%d.jpg' % (iter + 1))
                print('save image: result-%d.jpg' % (iter+1))
            # print loss
            if (iter + 1) % 10 == 0:
                print('iteration %d: %f' % (iter + 1, loss.item()))
            iter += 1
            return loss

        "----------------------"
        optimizer.step(closure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str, default='./data/content1.jpg')
    parser.add_argument('--style_path', type=str, default='./data/style1.jpg')
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=50)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--num_res', type=int, default=3)
    parser.add_argument('--gpu_chunck_size', type=int, default=256)
    parser.add_argument('--mrf_style_stride', type=int, default=2)
    parser.add_argument('--mrf_synthesis_stride', type=int, default=2)
    config = parser.parse_args()
    print(config)
    main(config)
