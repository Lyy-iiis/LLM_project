# This code based on https://github.com/enomotokenji/pytorch-Neural-Style-Transfer.git, which is the implementation of the original paper Gatys et al. (2016) 

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import os
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import argparse

import tqdm

import attention

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

############################################################################################################

def image_loader(image_name, imsize, gray = False):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize))  # scale imported image
        ])
    if not gray:
        image = Image.open(image_name)
    else:
        image = Image.open(image_name).convert('L')
        image = np.asarray(image)
        image = np.asarray([image,image,image])
        image = Image.fromarray(np.uint8(image).transpose(1,2,0))
    if image.size[0] == image.size[1]:
        image = loader(image)
    else:
        if image.size[0] > image.size[1]:
            image = image.crop((0, 0, image.size[1], image.size[1]))
        else:
            image = image.crop((0, 0, image.size[0], image.size[0]))
        image = loader(image)
    original_image = image
    image = Variable(transforms.ToTensor()(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def save_image(tensor, size, input_size, output_path, fname='transferred.png'):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(size)
    image = unloader(image).resize(input_size)
    out_path = output_path + fname
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    image.save(out_path)

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
    



def get_style_model_and_losses(cnn, style_img, content_img, style_weight=1000, content_weight=2, content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if True:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses

def get_input_param_optimizer(input_img, lr):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param], lr = lr)
    return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, lr, num_steps=300, style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    content_img_copy = content_img.clone().detach()
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img, lr)
    best_param, best_loss = None, float('inf')
    with tqdm.tqdm(total = num_steps) as pbar :
        run = [0]
        while run[0] <= num_steps - 20:

            def closure():
                nonlocal best_loss, best_param
                # correct the values of updated input image
                input_param.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_param)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.backward()
                for cl in content_losses:
                    content_score += cl.backward()

                run[0] += 1
                pbar.set_description('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                pbar.update(1)

                if style_score + content_score < best_loss :
                    best_loss = style_score + content_score
                    best_param = input_param.data.clone().detach()

                return style_score + content_score

            optimizer.step(closure)

    # a last correction...
    best_param.data.clamp_(0, 1)

    return best_param.data

def color_matching(content_img, style_img) :
    # [1, 3, l, l]
    content_img = content_img.squeeze(0)
    style_img = style_img.squeeze(0)
    # [3, l, l]
    mu_c = torch.mean(content_img, dim = (1, 2), keepdim = False) # [3]
    mu_s = torch.mean(style_img, dim = (1, 2), keepdim = False) # [3]
    Sigma_c = (content_img.reshape(3, -1) - mu_c.unsqueeze(1)) @ (content_img.reshape(3, -1) - mu_c.unsqueeze(1)).t() # [3, 3]
    Sigma_s = (style_img.reshape(3, -1) - mu_s.unsqueeze(1)) @ (style_img.reshape(3, -1) - mu_s.unsqueeze(1)).t() # [3, 3]
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_c) # [3], [3, 3]
    Sigma_c_sqrt = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.t() # [3, 3]
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_s) # [3], [3, 3]
    Sigma_s_sqrt_I = eigenvectors @ torch.diag(eigenvalues ** -0.5) @ eigenvectors.t() # [3, 3]
    A = Sigma_c_sqrt @ Sigma_s_sqrt_I # [3, 3]
    b = mu_c - A @ mu_s
    style_img = (A @ style_img.permute(1, 0, 2)).permute(1, 0, 2) + b.unsqueeze(1).unsqueeze(2)
    return style_img.unsqueeze(0)



def YIQ(img, mode = "encode") :
    A = torch.tensor([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]]).to(img.device)
    if mode == "decode" :
        A = torch.inverse(A)
    img = (A @ img.transpose(1, 2)).transpose(1, 2).contiguous()
    return img

def luminance_process(content_img, style_img) :
    content_img = YIQ(content_img)[:, :1, :, :] # [1, 1, l, l]
    style_img = YIQ(style_img)[:, :1, :, :] # [1, 1, l, l]
    mu_c = torch.mean(content_img, dim = (2, 3), keepdim = False) # [1, 1]
    mu_s = torch.mean(style_img, dim = (2, 3), keepdim = False) # [1, 1]
    sigma_c = torch.std(content_img, dim = (2, 3), keepdim = False) # [1, 1]
    sigma_s = torch.std(style_img, dim = (2, 3), keepdim = False) # [1, 1]
    style_img = (style_img - mu_s) * sigma_c / sigma_s + mu_c
    style_img = torch.cat([style_img, style_img, style_img], dim = 1).contiguous()
    content_img = torch.cat([content_img, content_img, content_img], dim = 1).contiguous()
    # style_img = YIQ(style_img, mode = "decode").clamp(0, 1)
    # content_img = YIQ(content_img, mode = "decode").clamp(0, 1)
    return content_img, style_img
