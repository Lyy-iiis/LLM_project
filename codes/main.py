import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--music_name', type = str, default = 'out.mp3')

DATA_PATH = os.getcwd() + '/data/'
MODEL_PATH = '/ssdshare/LLMs/'
MUSIC_PATH = os.getcwd() + '/data/music/'
LLM_MODEL = "glm-4"
GENRATE_MODEL = "playground-v2.5-1024px-aesthetic"
CONTENT_PATH = DATA_PATH + '.tmp/generate/'
STYLE_PATH = DATA_PATH + 'style/illustration_style/'

def run(music_name):
    MUSIC_NAME = music_name

    if os.path.exists(DATA_PATH + '.tmp/'):
        os.system(f'rm -rf {DATA_PATH}.tmp/')

    os.makedirs(DATA_PATH + '.tmp/')

    list = ['extract/', 'generate/', 'process/', 'inprompt', 'style_transfer']

    for folder in list:
        if not os.path.exists(DATA_PATH + '.tmp/' + folder):
            os.makedirs(DATA_PATH + '.tmp/' + folder)

    input_list = [
    MUSIC_NAME+'.mp3'
    ]
    prompts = [f''' ''',
    ]
    # Pick the style images in the style library
    style_list = [
    # 'opia.png'
    ]
    num_char = 2 # default 1
    num_non_char = 2 # default 1
    image_num = 1 
    # You should check both input_list and prompts modified!!!
    with open(DATA_PATH + 'input_list.txt', 'w') as f:
        for item in input_list:
            f.write("%s\n" % item)

    with open(DATA_PATH + 'style_list.txt', 'w') as f:
        for item in style_list:
            f.write("%s\n" % item)

    tmp_list = []
    for item in input_list:
        tmp_list.append(item[:-4])
    input_list = tmp_list

    # if not os.path.exists(DATA_PATH + '.tmp/inprompt/'):
    #   os.makedirs(DATA_PATH + '.tmp/inprompt/')
    for (prompt, name) in zip(prompts, input_list):
        with open(DATA_PATH + '.tmp/inprompt/' + name + '.prompt', 'w') as f:
            f.write(prompt)

    os.system(f'python extract/extract.py --model_path {MODEL_PATH} --data_path {DATA_PATH} --music_path {MUSIC_PATH} --output_path {DATA_PATH}.tmp/extract/ --device_num 2 --ignore_lyrics False')

    os.system(f'python process/process.py --model_path {MODEL_PATH} --data_path {DATA_PATH} --model {LLM_MODEL} --prompt_path {DATA_PATH}.tmp/extract/ --output_path {DATA_PATH}.tmp/process/ --num_char {num_char} --num_non_char {num_non_char}')

    os.system(f'python generate/generate.py --model_path {MODEL_PATH} --data_path {DATA_PATH} --model {GENRATE_MODEL} --output_path {DATA_PATH}.tmp/generate/ --prompt_path {DATA_PATH}.tmp/process/ --image_num {image_num} --num_char {num_char} --num_non_char {num_non_char}')

    # import os, glob
    # for file_name in input_list:
    #     [os.remove(f) for f in glob.glob(DATA_PATH + '.tmp/style_transfer/' + file_name + '/*')]

    os.system(f'python style_transfer/style_transfer.py --data_path {DATA_PATH} --output_path {DATA_PATH}.tmp/style_transfer/ --style_path {STYLE_PATH} --content_path {CONTENT_PATH} -l_o --num_char {num_char} --num_non_char {num_non_char}')

    if not os.path.exists(f'{DATA_PATH}output/'):
        os.mkdir(f'{DATA_PATH}output/')

    MUSIC_NAME = music_name.replace(" ", "")
    if not os.path.exists(f'{DATA_PATH}output/{MUSIC_NAME}'):
        os.mkdir(f'{DATA_PATH}output/{MUSIC_NAME}')

    print(f'{DATA_PATH}output/{MUSIC_NAME}')

    os.system(f'cp -r {DATA_PATH}.tmp/ {DATA_PATH}output/{MUSIC_NAME}')

def parse_name(music_name):
    music = music_name
    name = music.split('-')[-1].strip()
    return name

if __name__ == '__main__':
    os.system(f'ls {DATA_PATH}music/ > music_list.txt')
    with open('music_list.txt', 'r') as f:
        music_list = f.readlines()
    for music in music_list:
        music = music[:-5]
        print(parse_name(music))
        run(music)