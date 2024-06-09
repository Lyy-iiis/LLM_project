
########################################################

from diffusers import DiffusionPipeline
import torch
import os
torch.manual_seed(42)

import argparse 
import torch
import re

#######################################################


parser = argparse.ArgumentParser(description = 'Extractor')

parser.add_argument('--input_file_name', type = str, default = 'input_list.txt')
parser.add_argument('--model_path', type = str, default = '/ssdshare/LLMs/')
parser.add_argument('--data_path', type = str, default = '../data/')
parser.add_argument('--model', type = str, default = 'playground-v2.5-1024px-aesthetic')
parser.add_argument('--prompt_path', type = str, default = '../data/.tmp/process/')
parser.add_argument('--output_path', type = str, default = '../data/.tmp/generate/')
parser.add_argument('--image_num', type = int, default = 1)
parser.add_argument('--device_num', type = int, default = 1)
parser.add_argument('--num_non_char', '-nnc', type = int, default = 1)
parser.add_argument('--num_char', '-nc', type = int, default = 1)

args = parser.parse_args()

#######################################################

MODEL_PATH = args.model_path
PWD = os.getcwd()
DATA_PATH = args.data_path
PROMPT_PATH = args.prompt_path
OUTPUT_PATH = args.output_path
MODEL = args.model
if DATA_PATH[0] == '.' :
    DATA_PATH = PWD + "/" + DATA_PATH
if PROMPT_PATH[0] == '.' :
    PROMPT_PATH = PWD + "/" + PROMPT_PATH
if OUTPUT_PATH[0] == '.' :
    OUTPUT_PATH = PWD + "/" + OUTPUT_PATH
DEVICE = "cuda" if torch.cuda.is_available() else "xuwei"
assert DEVICE == "cuda", "WHY DONT YOU HAVE CUDA???????"
TEMPORARY_PATH = DATA_PATH + ".tmp/"
if not os.path.exists(TEMPORARY_PATH) :
    os.makedirs(TEMPORARY_PATH)
CUDA_NUM = args.device_num
assert CUDA_NUM > 0, "DO YOU WANT ME TO DONATE MY GPU TO YOU????"
assert CUDA_NUM <= torch.cuda.device_count(), "YOU ARE ASKING FOR TOO MANY GPUS"
CUDA_DEVICE = [f"cuda:{i}" for i in range(CUDA_NUM)]

#######################################################

# load prompt from file

print("Loading prompt from file")

audio_file_name = []
input_file_name = args.input_file_name
with open(DATA_PATH + input_file_name, "r") as f :
    for line in f :
        audio_file_name.append(line.rstrip())

ori_audio_file_name = audio_file_name

# Replace ".mp3" with ".prompt" in audio_file_name
audio_file_name = [re.sub(r'\.mp3$', '.prompt', audio) for audio in audio_file_name]
audio_file_name = [re.sub(r'\.wav$', '.prompt', audio) for audio in audio_file_name]
for audio in audio_file_name :
    print(audio)

prompt = {}
for audio in audio_file_name :
    prompt_list = []
    for t in range(args.num_char) :
        with open(PROMPT_PATH + audio + str(t), "r") as f :
            prompt_list.append(f.read())
    prompt[audio] = prompt_list
    # print(prompt[audio])

print("Prompt loaded")
print("Loading model")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH + MODEL + '/',
    # custom_pipeline = "/root/LLM_project/codes/generate/lpw_stable_diffusion_xl.py",
    custom_pipeline = "/ssdshare/MI-T/lpw_stable_diffusion_xl.py",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

print("Model loaded")

num_inference_steps = 50
guidance_scale = 7.5
image_num = args.image_num
negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, black man, black woman, piano, crowd, word, title"

for audio in audio_file_name :
    print(f"Generating for {audio}")
    for t in range(args.num_char) :
        output = pipe(prompt[audio][t], 
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=image_num).images
        store_path = OUTPUT_PATH + audio[:-7]
        if not os.path.exists(store_path) :
            os.makedirs(store_path)
        for i in range(image_num):
            output[i].save(store_path + f'/{t}-{i}.png')
    print(f"Generated for {audio}")

#######################################################

# Now generate from .prompt2

# load prompt from file
if args.num_non_char > 0 :
    print("Loading prompt from file")
    print("Generating image without characters")    


audio_file_name = ori_audio_file_name
# input_file_name = args.input_file_name
# with open(DATA_PATH + input_file_name, "r") as f :
#     for line in f :
#         audio_file_name.append(line.rstrip())

# Replace ".mp3" with ".prompt" in audio_file_name
audio_file_name = [re.sub(r'\.mp3$', '.prompt_nc', audio) for audio in audio_file_name]
audio_file_name = [re.sub(r'\.wav$', '.prompt_nc', audio) for audio in audio_file_name]
# for audio in audio_file_name :
#     print(audio)

prompt = {}
for audio in audio_file_name :
    prompt_list = []
    for t in range(args.num_non_char) :
        with open(PROMPT_PATH + audio + str(t), "r") as f :
            prompt_list.append(f.read())
    prompt[audio] = prompt_list

if args.num_non_char > 0 :
    print("Prompt loaded")

# num_inference_steps = 50
# guidance_scale = 7.5
# image_num = args.image_num
negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, character, people, robot, human, couple, woman, man, child, baby, boy, girl, kid, adult, old, young, piano, road, word, title, black woman, black man, black child, black skin"

for audio in audio_file_name :
    print(f"Generating for {audio}")
    for t in range(args.num_non_char) :
        output = pipe(prompt[audio][t], 
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=image_num).images
        store_path = OUTPUT_PATH + audio[:-10]
        if not os.path.exists(store_path) :
            os.makedirs(store_path)
        for i in range(image_num):
            output[i].save(store_path + f'/nc{t}-{i}.png')
    print(f"Generated for {audio}")