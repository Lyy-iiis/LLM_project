# This code based on https://github.com/enomotokenji/pytorch-Neural-Style-Transfer.git, which is the implementation of the original paper Gatys et al. (2016) 

from __future__ import print_function

import torch

import os
from PIL import Image
import numpy as np

import torchvision.models as models

import argparse

from utils import image_loader, save_image, run_style_transfer, color_matching, YIQ, luminance_process

############################################################################

parser = argparse.ArgumentParser()
# parser.add_argument('--content', '-c', type=str, required=True, help='The relative path to the Content image')
parser.add_argument('--content', '-c', type=str, default = 'see list', help='The path to the Content image')
parser.add_argument('--data_path', type=str, default = '../data/', help='The path to input_list.txt')
parser.add_argument('--content_path', type=str, default = '../data/.tmp/generate/', help='The path to the content image')
parser.add_argument('--style_path', type=str, default = '../data/style/illustration_style/', help='The path to the style image')
# parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
parser.add_argument('--style', '-s', type=str, default = 'see list', help='The path to the style image')
parser.add_argument('--epoch', '-e', type=int, default=200, help='The number of epoch')
parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
parser.add_argument('--style_weight', '-s_w', type=int, default=500, help='The weight of style loss')
# parser.add_argument('--initialize_noise', '-i_n', action='store_true', help='Initialize with white noise? elif initialize with content image')
parser.add_argument('--img_size', '-i_s', type=int, default=1024)
parser.add_argument('--output_path', '-o', type=str, default='transferred/')
parser.add_argument('--gray', '-g', action='store_true')
parser.add_argument('--color_preserve', '-c_p', action='store_true', help = "doesn't work if gray is True")
parser.add_argument('--luminance_only', '-l_o', action='store_true', help = "doesn't work if gray is True")
parser.add_argument('--aams', action = 'store_true')
parser.add_argument('--attn', action = 'store_true')
parser.add_argument('--lr', '-lr', type=float, default=1)
parser.add_argument('--num_non_char', '-nnc', type = int, default = 1)
parser.add_argument('--num_char', '-nc', type = int, default = 1)
args = parser.parse_args()

############################################################################

if args.luminance_only and args.aams :
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.outputs import OutputKeys
if args.attn :
    import attention

assert torch.cuda.is_available(), "WHY DONT YOU HAVE CUDA??????"
DEVICE = torch.device("cuda:0")
IMG_SIZE = args.img_size
OUTPUT_PATH = args.output_path
DATA_PATH = args.data_path
CONTENT_PATH = args.content_path
STYLE_PATH = args.style_path

cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()


############################################################################
if args.attn :
    attention.attn_init()

input_file_name = []
name = []
# put all the picture under generate folder into a list
if args.content == 'see list':
    with open(DATA_PATH + 'input_list.txt', "r") as f :
        for line in f :
            input_file_name.append(line.rstrip())
    tmp = [x[:-4] for x in input_file_name]
    suffix, input_file_name = [], {}
    for x in tmp:
        # if CONTENT_PATH[-1] == '/':
        #     CONTENT_PATH = CONTENT_PATH[:-1]
        print(os.listdir(CONTENT_PATH+x))
        input_file_name[x] = []
        for pic in os.listdir(CONTENT_PATH+x):
            if pic.endswith(".png"):
                input_file_name[x].append(pic)
    name = tmp
else:
    input_file_name.append(args.content)

# print("input_file_name:",input_file_name) # a dictionary

# print("name:",name) # contains the name of input music, without the suffix

style_file_name = {}
style_file_name_nc = {}
if args.style == 'see list':
    for fname in name :
        style_file_name[fname] = []
        style_file_name_nc[fname] = []
        for t in range(args.num_char) :
            with open(DATA_PATH + '.tmp/process/' + fname + '.style' + str(t), "r") as f :
                style_file_name[fname].append(f.readline().rstrip())
        for t in range(args.num_non_char) :
            with open(DATA_PATH + '.tmp/process/' + fname + '.style_nc' + str(t), "r") as f :
                style_file_name_nc[fname].append(f.readline().rstrip())
        style_file_name[fname] = [x + '.png' for x in style_file_name[fname]]
        style_file_name_nc[fname] = [x + '.png' for x in style_file_name_nc[fname]]
else:
    style_file_name["specified"].append(args.style)

# print("style_file_name:",style_file_name) # a dictionary

# tmp = {}
# for music in name:
#     tmp[music] = []
#     for pic in input_file_name[x]:
#         if len(pic) == 7:
#             tmp[music].append(style_file_name[0])
#         elif len(pic) == 9:
#             tmp[music].append(style_file_name[1])
#         else:
#             assert False, "You should not ask for more than 10 pictures one time"

# tmp_input, tmp_style = [], []
# for music in name:
#     for pic in input_file_name[music]:
#         tmp_input.append(music+"/"+pic)
#         tmp_style.append(style_file_name[music].pop(0))
# input_file_name, style_file_name = tmp_input, tmp_style

# write very good, not write next time.......

map = {}
for music in name:
    map[music] = {}
    for pic in input_file_name[music]:
        if len(pic) == 7:
            map[music][pic] = style_file_name[music][0]
            style_file_name[music].pop(0)
        elif len(pic) == 9:
            map[music][pic] = style_file_name_nc[music][0]
            style_file_name_nc[music].pop(0)
        else :
            print(pic)
            assert False, "You should not ask for more than 10 pictures one time"

# print("map:", map)

if args.luminance_only and args.aams :
    style_transfer = pipeline(Tasks.image_style_transfer, model_id='/ssdshare/LLMs/cv_aams_style-transfer_damo/')

for music in name:
    for key, value in map[music].items():
        content = music + "/" + key
        style = value
        print("content:", content)
        print("style:", style)
        if args.style == 'see list':
            style = STYLE_PATH + style
        # Now style is the complete path to the style image
        style_img = image_loader(style, IMG_SIZE)
        style_img = style_img.type(torch.cuda.FloatTensor).to(DEVICE)
        if args.content == 'see list':
            content = CONTENT_PATH + content
        ### image Loaded
        print(f"Transferring from {content} to {style}")
        if args.gray:
            content_img = image_loader(content, IMG_SIZE, gray=True)
            content_img = content_img.type(torch.cuda.FloatTensor).to(DEVICE)
            input_img = image_loader(content, IMG_SIZE, gray=True)
            input_img = input_img.type(torch.cuda.FloatTensor).to(DEVICE)
        elif args.color_preserve:
            content_img = image_loader(content, IMG_SIZE)
            content_img = content_img.type(torch.cuda.FloatTensor).to(DEVICE)
            input_img = image_loader(content, IMG_SIZE)
            input_img = input_img.type(torch.cuda.FloatTensor).to(DEVICE)
            style_img = color_matching(content_img, style_img).to(DEVICE)
        elif args.luminance_only:
            ori_img = image_loader(content, IMG_SIZE)
            ori_img = ori_img.type(torch.cuda.FloatTensor).to(DEVICE)
            content_img, style_img = luminance_process(ori_img, style_img)
            input_img = content_img.clone().detach().to(DEVICE)

        else:
            content_img = image_loader(content, IMG_SIZE)
            content_img = content_img.type(torch.cuda.FloatTensor)
            input_img = image_loader(content, IMG_SIZE)
            input_img = input_img.type(torch.cuda.FloatTensor)
        # print(attn, attn.max(), attn.min())

        input_size = Image.open(content).size

        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"

        if args.luminance_only and args.aams :
            content_img = Image.fromarray(np.uint8(content_img.squeeze(0).cpu() * 255).transpose(1, 2, 0))
            style_img = Image.fromarray(np.uint8(style_img.squeeze(0).cpu() * 255).transpose(1, 2, 0))
            output = style_transfer(dict(content = content_img, style = style_img))
            output = output[OutputKeys.OUTPUT_IMG]
            output = torch.tensor(output).to(DEVICE).permute(2, 0, 1).type(torch.cuda.FloatTensor).unsqueeze(0) / 255
            output = output.clamp(0, 1)
        else :
            output = run_style_transfer(cnn, content_img, style_img, input_img, args.lr, args.epoch, args.style_weight, args.content_weight)
        if not args.gray and not args.color_preserve and args.luminance_only:
            upper_bound, _ = (1 - ori_img[0]).min(dim = 0)
            lower_bound, _ = (-ori_img[0]).max(dim = 0)
            # print(lower_bound.size(), upper_bound.size())
            output = YIQ(output)
            ori_img = YIQ(ori_img)
            lumi_delta = output[0][0] - ori_img[0][0]
            lumi_delta = torch.clamp(lumi_delta, lower_bound, upper_bound)
            ori_img[0][0] = ori_img[0][0] + lumi_delta
            output = YIQ(ori_img, mode = "decode")

        name_content, ext = os.path.splitext(os.path.basename(content))
        name_style, _ = os.path.splitext(os.path.basename(style))
        fname = name_content+'-'+name_style+ext

        # output = torch.stack([0.299 * output, 0.587 * output, 0.114 * output], dim=0)
        if args.attn :
            content_img = image_loader(content, IMG_SIZE)
            content_img = content_img.type(torch.cuda.FloatTensor).to(DEVICE)
            # attn = attention.attn_map(Image.fromarray(np.uint8(content_img.squeeze(0).cpu() * 255).transpose(1, 2, 0))).squeeze(2).unsqueeze(0).to(DEVICE)
            attn = attention.attn_map(Image.open(content)).squeeze(2).unsqueeze(0).to(DEVICE)
            output = attn * output + (1 - attn) * content_img
        save_image(output, size=input_img.data.size()[1:], input_size=input_size, output_path = OUTPUT_PATH + music + "/", fname=fname)
        print(f"Transfer from {content} to {style} done")