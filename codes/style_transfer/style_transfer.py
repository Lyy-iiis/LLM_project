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
import re
############################################################################

parser = argparse.ArgumentParser()
# parser.add_argument('--content', '-c', type=str, required=True, help='The relative path to the Content image')
parser.add_argument('--content', '-c', type=str, default = 'see list', help='The path to the Content image')
parser.add_argument('--data_path', type=str, default = '../data/', help='The path to input_list.txt')
parser.add_argument('--content_path', type=str, default = '../data/.tmp/generate/', help='The path to the content image')
parser.add_argument('--style_path', type=str, default = '../data/style/', help='The path to the style image')
# parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
parser.add_argument('--style', '-s', type=str, default = 'see list', help='The path to the style image')
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
DATA_PATH = args.data_path
CONTENT_PATH = args.content_path
STYLE_PATH = args.style_path

cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()


############################################################################
input_file_name = []
if args.content == 'see list':
    with open(DATA_PATH + 'input_list.txt', "r") as f :
        for line in f :
            input_file_name.append(line.rstrip())
    input_file_name = [re.sub(r'\.mp3$', '/0.png', audio) for audio in input_file_name]
    input_file_name = [re.sub(r'\.wav$', '/0.png', audio) for audio in input_file_name]
else:
    input_file_name.append(args.content)

style_file_name = []
if args.style == 'see list':
    with open(DATA_PATH + 'style_list.txt', "r") as f :
        for line in f :
            style_file_name.append(line.rstrip())

else:
    style_file_name.append(args.style)

for content in input_file_name:
    for style in style_file_name:
        if args.style == 'see list':
            style = STYLE_PATH + style
        # Now style is the complete path to the style image
        style_img = image_loader(style, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
        if args.content == 'see list':
            content = CONTENT_PATH + content
        ### image Loaded
        print(f"Transferring from {content} to {style}")
        if args.gray:
            content_img = image_loader(content, IMG_SIZE, gray=True).type(torch.cuda.FloatTensor).to(DEVICE)
            input_img = image_loader(content, IMG_SIZE, gray=True).type(torch.cuda.FloatTensor).to(DEVICE)
        elif args.color_preserve:
            content_img = image_loader(content, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
            input_img = image_loader(content, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
            style_img = color_matching(content_img, style_img).to(DEVICE)
        elif args.luminance_only:
            ori_img = image_loader(content, IMG_SIZE).type(torch.cuda.FloatTensor).to(DEVICE)
            content_img, style_img = luminance_process(ori_img, style_img)
            input_img = content_img.clone().detach().to(DEVICE)

        else:
            content_img = image_loader(content, IMG_SIZE).type(torch.cuda.FloatTensor)
            input_img = image_loader(content, IMG_SIZE).type(torch.cuda.FloatTensor)

        input_size = Image.open(content).size

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

        if args.content == 'see list':
            content = re.sub(r'/0.png', '.png', content)
        name_content, ext = os.path.splitext(os.path.basename(content))
        name_style, _ = os.path.splitext(os.path.basename(style))
        fname = name_content+'-'+name_style+ext

        save_image(output, size=input_img.data.size()[1:], input_size=input_size, output_path = OUTPUT_PATH, fname=fname)
        print(f"Transfer from {content} to {style} done")

