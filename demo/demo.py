import os
from torch import cuda

DATA_PATH = os.getcwd() + '/codes/data/demo/'
MODEL_PATH = '/ssdshare/LLMs/'
MUSIC_PATH = os.getcwd() + '/codes/data/demo/music/'
LLM_MODEL = "glm-4"
GENRATE_MODEL = "playground-v2.5-1024px-aesthetic"
CONTENT_PATH = DATA_PATH + '.tmp/generate/'
STYLE_PATH = DATA_PATH + 'style/illustration_style/'
DEVICE_NUM = cuda.device_count()
IMAGE_NUM = 1

if os.path.exists(DATA_PATH + '.tmp/'):
  os.system(f'rm -r {DATA_PATH}.tmp/')

# if not os.path.exists(DATA_PATH + '.tmp/'):
os.makedirs(DATA_PATH + '.tmp/')

list = ['extract/', 'generate/', 'process/', 'inprompt', 'style_transfer']

for folder in list:
  if not os.path.exists(DATA_PATH + '.tmp/' + folder):
    os.makedirs(DATA_PATH + '.tmp/' + folder)

input_list = [
  'music.wav',
]

prompts = []

with open(DATA_PATH + 'input_list.txt', 'w') as f:
  for item in input_list:
    f.write("%s\n" % item)

tmp_list = []
for item in input_list:
  tmp_list.append(item[:-4])
input_list = tmp_list

with open(MUSIC_PATH + 'prompt.txt', 'r') as f:
  prompts = f.readlines()

for (prompt, name) in zip(prompts, input_list):
  with open(DATA_PATH + '.tmp/inprompt/' + name + '.prompt', 'w') as f:
    f.write(prompt)

os.system(f'python codes/extract/extract.py --model_path {MODEL_PATH} --data_path {DATA_PATH} --music_path {MUSIC_PATH} --output_path {DATA_PATH}.tmp/extract/ --device_num {DEVICE_NUM}')

if not os.path.exists(DATA_PATH + 'style/'):
  os.makedirs(DATA_PATH + 'style/')
  os.system(f'cp {DATA_PATH}/../style -r {DATA_PATH}/')

os.system(f'python codes/process/process.py --model_path {MODEL_PATH} --data_path {DATA_PATH} --model {LLM_MODEL} --prompt_path {DATA_PATH}.tmp/extract/ --output_path {DATA_PATH}.tmp/process/')

for file_name in input_list:
  with open(DATA_PATH + '.tmp/process/' + file_name + '.prompt', 'r') as f:
    print(f.read())

os.system(f'python codes/generate/generate.py --model_path {MODEL_PATH} --data_path {DATA_PATH} --model {GENRATE_MODEL} --output_path {DATA_PATH}.tmp/generate/ --prompt_path {DATA_PATH}.tmp/process/ --image_num {IMAGE_NUM}')

os.system(f'python codes/style_transfer/style_transfer.py --data_path {DATA_PATH} --output_path {DATA_PATH}.tmp/style_transfer/ --style_path {STYLE_PATH} --content_path {CONTENT_PATH} -c_p')