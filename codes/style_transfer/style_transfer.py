# This code based on https://github.com/enomotokenji/pytorch-Neural-Style-Transfer.git

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

############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--content', '-c', type=str, required=True, help='The path to the Content image')
parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
parser.add_argument('--epoch', '-e', type=int, default=300, help='The number of epoch')
parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
parser.add_argument('--style_weight', '-s_w', type=int, default=500, help='The weight of style loss')
# parser.add_argument('--initialize_noise', '-i_n', action='store_true', help='Initialize with white noise? elif initialize with content image')
parser.add_argument('--img_size', '-i_s', type=int, default=1024)
parser.add_argument('--output_path', '-o', type=str, default='transferred')
parser.add_argument('--gray', '-g', action='store_true')
args = parser.parse_args()

############################################################################

assert torch.cuda.is_available(), "WHY DONT YOU HAVE CUDA??????"
DEVICE = torch.device("cuda:0")
IMG_SIZE = args.img_size
OUTPUT_PATH = args.output_path

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval()

############################################################################


def image_loader(image_name, imsize, gray = False):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    if not gray:
        image = Image.open(image_name)
    else:
        image = Image.open(image_name).convert('L')
        image = np.asarray(image)
        image = np.asarray([image,image,image])
        image = Image.fromarray(np.uint8(image).transpose(1,2,0))
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def save_image(tensor, size, input_size, fname='transferred.png'):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(size)
    image = unloader(image).resize(input_size)
    out_path = OUTPUT_PATH + fname
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
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
    



def get_style_model_and_losses(cnn, style_img, content_img, style_weight=1000, content_weight=1, content_layers=content_layers_default, style_layers=style_layers_default):
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

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)
    with tqdm.tqdm(total=num_steps) as pbar :
        run = [0]
        while run[0] <= num_steps:

            def closure():
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

                return style_score + content_score

            optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data

############################################################################

style_img = image_loader(args.style, IMG_SIZE).type(torch.cuda.FloatTensor)
if args.gray:
    content_img = image_loader(args.content, IMG_SIZE, gray=True).type(torch.cuda.FloatTensor)
    input_img = image_loader(args.content, IMG_SIZE, gray=True).type(torch.cuda.FloatTensor)
else:
    content_img = image_loader(args.content, IMG_SIZE).type(torch.cuda.FloatTensor)
    input_img = image_loader(args.content, IMG_SIZE).type(torch.cuda.FloatTensor)

input_size = Image.open(args.content).size

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

output = run_style_transfer(cnn, content_img, style_img, input_img, args.epoch, args.style_weight, args.content_weight)

name_content, ext = os.path.splitext(os.path.basename(args.content))
name_style, _ = os.path.splitext(os.path.basename(args.style))
fname = name_content+'-'+name_style+ext

save_image(output, size=input_img.data.size()[1:], input_size=input_size, fname=fname)


