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

from utils import image_loader, save_image, ContentLoss, GramMatrix, StyleLoss, get_input_param_optimizer, get_style_model_and_losses, run_style_transfer, color_matching, YIQ, luminance_process

############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--content', '-c', type=str, required=True, help='The path to the Content image')
parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
parser.add_argument('--epoch', '-e', type=int, default=300, help='The number of epoch')
parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
parser.add_argument('--style_weight', '-s_w', type=int, default=500, help='The weight of style loss')
# parser.add_argument('--initialize_noise', '-i_n', action='store_true', help='Initialize with white noise? elif initialize with content image')
parser.add_argument('--img_size', '-i_s', type=int, default=1024)
parser.add_argument('--output_path', '-o', type=str, default='transferred/')
parser.add_argument('--gray', '-g', action='store_true')
parser.add_argument('--color_preserve', '-c_p', action='store_true', help = "doesn't work if gray is True")
parser.add_argument('--luminance_only', '-l_o', action='store_true', help = "doesn't work if gray is True")
parser.add_argument('--lr', '-lr', type=float, default=1)
args = parser.parse_args()

############################################################################

assert torch.cuda.is_available(), "WHY DONT YOU HAVE CUDA??????"
DEVICE = torch.device("cuda:0")
IMG_SIZE = args.img_size
OUTPUT_PATH = args.output_path

cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()


############################################################################

style_img = image_loader(args.style, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
if args.gray:
    content_img = image_loader(args.content, IMG_SIZE, gray=True).type(torch.cuda.FloatTensor).to(DEVICE)
    input_img = image_loader(args.content, IMG_SIZE, gray=True).type(torch.cuda.FloatTensor).to(DEVICE)
elif args.color_preserve:
    content_img = image_loader(args.content, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
    input_img = image_loader(args.content, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
    style_img = color_matching(content_img, style_img).to(DEVICE)
elif args.luminance_only:
    ori_img = image_loader(args.content, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
    content_img, style_img = luminance_process(ori_img, style_img)
    input_img = content_img.clone().detach().to(DEVICE)

else:
    content_img = image_loader(args.content, IMG_SIZE).type(torch.cuda.FloatTensor)
    input_img = image_loader(args.content, IMG_SIZE).type(torch.cuda.FloatTensor)

input_size = Image.open(args.content).size

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

output = run_style_transfer(cnn, content_img, style_img, input_img, args.lr, args.epoch, args.style_weight, args.content_weight)
# a bad news is that if we concat them into a batch, cuda out of memory
save_image(output, size=input_img.data.size()[1:], input_size=input_size, output_path = OUTPUT_PATH, fname="tmp1.jpg")
if not args.gray and not args.color_preserve and args.luminance_only:
    upper_bound, _ = (1 - ori_img[0]).min(dim = 0)
    lower_bound, _ = (-ori_img[0]).max(dim = 0)
    print(lower_bound.size(), upper_bound.size())
    output = YIQ(output)
    ori_img = YIQ(ori_img)
    lumi_delta = output[0][0] - ori_img[0][0]
    lumi_delta = torch.clamp(lumi_delta, lower_bound, upper_bound)
    ori_img[0][0] = ori_img[0][0] + lumi_delta
    output = YIQ(ori_img, mode = "decode")

name_content, ext = os.path.splitext(os.path.basename(args.content))
name_style, _ = os.path.splitext(os.path.basename(args.style))
fname = name_content+'-'+name_style+ext

save_image(output, size=input_img.data.size()[1:], input_size=input_size, output_path = OUTPUT_PATH, fname=fname)


