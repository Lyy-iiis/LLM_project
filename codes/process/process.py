from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI
import argparse 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

##############################################################

parser = argparse.ArgumentParser(description = 'Extractor')

parser.add_argument('--model', type = str, default = 'glm-4')
parser.add_argument('--model_path', type = str, default = '/ssdshare/LLMs/')
parser.add_argument('--input_file_name', type = str, default = 'input_list.txt')
parser.add_argument('--data_path', type = str, default = '../data/')
parser.add_argument('--output_path', type = str, default = '../data/.tmp/process/')
parser.add_argument('--prompt_path', type = str, default = '../data/.tmp/extract/')
parser.add_argument('--inprompt_path', type = str, default = '.tmp/inprompt/')
parser.add_argument('--num_non_char', '-nnc', type = int, default = 1)
parser.add_argument('--num_char', '-nc', type = int, default = 1)

args = parser.parse_args()

##############################################################

MODEL_NAME = args.model
IS_MODEL_LOCAL = {"glm-4" : False, "llama-3-8B" : True, "llama-3-70B": True}
MODEL = None
TOKENIZER = None
MODEL_PATH = args.model_path
MODEL_ID = {"llama-3-8B" : "Meta-Llama-3-8B-Instruct/", "llama-3-70B" : "Meta-Llama-3-70B-Instruct-AWQ/"}
DATA_PATH = args.data_path
OUTPUT_PATH = args.output_path
if not os.path.exists(OUTPUT_PATH) :
  os.makedirs(OUTPUT_PATH)
PROMPT_PATH = args.prompt_path
INPROMPT_PATH = args.inprompt_path
PROCESS_PATH = os.path.dirname(os.path.abspath(__file__))+ "/"

input_file_name = args.input_file_name

##############################################################


def load() :
  global MODEL, TOKENIZER
  print("Loading model")
  if MODEL_NAME == "glm-4" :
    load_dotenv()
    zhipuai_api_key = os.environ.get("ZHIPUAI_API_KEY")
    MODEL = ZhipuAI(api_key = zhipuai_api_key)
  elif MODEL_NAME == "llama-3-8B" :
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH + MODEL_ID[MODEL_NAME], trust_remote_code = True)
    MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH + MODEL_ID[MODEL_NAME],
    torch_dtype = torch.bfloat16,
    device_map = "auto",
    trust_remote_code = True
    )
  elif MODEL_NAME == "llama-3-70B" :
    assert torch.cuda.device_count() >= 4, f"YOU WANT TO USE ONLY {torch.cuda.device_count()} GPU TO RUN THIS MODEL ???"
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH + MODEL_ID[MODEL_NAME], trust_remote_code = True)
    MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH + MODEL_ID[MODEL_NAME],
    torch_dtype = torch.float16,
    device_map = "auto",
    trust_remote_code = True
    )
  else :
    raise ValueError("Model not found")
  print("Model loaded")
  
def f_response(messages) :
  if not IS_MODEL_LOCAL[MODEL_NAME] :
    response = MODEL.chat.completions.create(
        model = MODEL_NAME,
        messages = messages
    )
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return content, tokens
  else :
    input_ids = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(MODEL.device)

    terminators = [
        TOKENIZER.eos_token_id,
        TOKENIZER.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = MODEL.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return TOKENIZER.decode(response, skip_special_tokens=True), 0

# def run_llama3_70b(input_file_name, output_file_name):
#   server = "/share/ollama/ollama serve"
#   user = f"/share/ollama/ollama run llama3:70b < \"{input_file_name}\" > \"{output_file_name}\""
#   print("Loading model")
#   process_server = subprocess.Popen(server, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#   process_user = subprocess.Popen(user, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#   stdout, stderr = process_user.communicate()
#   # return_code_1 = process_server.returncode
#   return_code_2 = process_user.returncode
#   print(f"User return code: {return_code_2}")
#   print(f"User output: {stdout.decode()}")
#   print(f"User error: {stderr.decode()}")

audio_file_name = []
with open(DATA_PATH + input_file_name, "r") as f :
  for line in f :
    line = line.rstrip()
    assert line[-4:] in [".mp3", ".wav"]
    line = line[:-4]
    audio_file_name.append(line)
print(audio_file_name)

prompts = []
for file_name in audio_file_name :
  with open(PROMPT_PATH + file_name + ".prompt", "r") as f :
    prompts.append(f.read())


with open(PROCESS_PATH+"prompt", "r") as f:
  system_prompt = f.read()

style_dicription = {}
with open(DATA_PATH + "style/style_description.txt", "r") as f:
    style_num = 1
    for line in f :
        style_dicription[style_num] = line.rstrip()
        style_num += 1

with open(PROCESS_PATH+"style_prompt", "r") as f:
  style_prompt = f.read()

def binary_ask(i, j, inprompt, prompt):
    # random shuffle i, j
    shuffle = 0
    if torch.rand(1) > 0.5:
      shuffle = 1
      i, j = j, i
    prompt_ij = style_prompt
    prompt_ij += "1. " + style_dicription[i] + "\n"
    prompt_ij += "2. " + style_dicription[j] + "\n"
    user_input = inprompt + """The description is: 
    
    """+prompt
    # print(user_input)
    messages = [
        {"role": "system", "content": prompt_ij},
        {"role": "user", "content": user_input},
    ]
    response, tokens = f_response(messages)
    number = re.findall(r'\d+', response)[0]
    if number != '1' and number != '2':
        number = '1'
    if shuffle:
        number = '1' if number == '2' else '2'
    return int(number), tokens

def get_style(prompt, inprompt):
  total_tokens = 0
  best = [i for i in range(1, style_num)]
  while len(best) > 1 :
    # print(best)
    for i in range(1, len(best), 2) :
        result, tokens = binary_ask(best[i-1], best[i],inprompt, prompt)
        total_tokens += tokens
        best[i-1] = best[i-2+result]
    best = best[::2]
    total_tokens += tokens
  # print(total_tokens, best[0])
  print(best[0])
  return best[0], total_tokens

load()
token_spent = 0
print(type(MODEL), type(TOKENIZER))
for t in range(args.num_char) :
  for (prompt, file_name) in zip(prompts, audio_file_name) :
    with open(DATA_PATH + INPROMPT_PATH + file_name + ".prompt", "r") as f :
      inprompt = f.read()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": inprompt+'\n'+prompt},
    ]
    response, tokens = f_response(messages)
    with open(OUTPUT_PATH + file_name + ".prompt" + str(t), "w") as f :
      f.write(response)
    token_spent += tokens
    num, tokens = get_style(response,inprompt)
    token_spent += tokens
    with open(OUTPUT_PATH + file_name + ".style" + str(t), "w") as f :
      f.write(num.__str__())

# print("Token spent:", token_spent)

##############################################################
# prompt for generating image with no character

with open(PROCESS_PATH+"prompt_nc", "r") as f:
  system_prompt = f.read()
# The second example is DESTRUCTION 3,2,1

# load()
# for t in range(args.num_non_char) :
#   for (prompt, file_name) in zip(prompts, audio_file_name) :
#     with open(DATA_PATH + INPROMPT_PATH + file_name + ".prompt", "r") as f :
#       inprompt = f.read()
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": inprompt+'\n'+prompt},
#     ]
#     response, tokens = f_response(messages)
#     with open(OUTPUT_PATH + file_name + ".prompt_nc" + str(t), "w") as f :
#       f.write(response)
#     token_spent += tokens
#     messages = [
#         {"role": "system", "content": style_prompt},
#         {"role": "user", "content": response},
#     ]
#     response, tokens = f_response(messages)
#     token_spent += tokens
#     num = re.findall(r'\b\d+\b', response)[0]
#     with open(OUTPUT_PATH + file_name + ".style_nc" + str(t), "w") as f :
#       f.write(num)
# print("Token spent:", token_spent)

for t in range(args.num_non_char) :
  for (prompt, file_name) in zip(prompts, audio_file_name) :
    with open(DATA_PATH + INPROMPT_PATH + file_name + ".prompt", "r") as f :
      inprompt = f.read()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": inprompt+'\n'+prompt},
    ]
    response, tokens = f_response(messages)
    with open(OUTPUT_PATH + file_name + ".prompt_nc" + str(t), "w") as f :
      f.write(response)
    token_spent += tokens
    num, tokens = get_style(response,inprompt)
    token_spent += tokens
    with open(OUTPUT_PATH + file_name + ".style_nc" + str(t), "w") as f :
      f.write(num.__str__())

print("Token spent:", token_spent)