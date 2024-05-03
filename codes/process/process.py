from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI
import argparse 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

##############################################################

parser = argparse.ArgumentParser(description = 'Extractor')

parser.add_argument('--model', type = str, default = 'glm-4')
parser.add_argument('--model_path', type = str, default = '/ssdshare/LLMs/')
parser.add_argument('--input_file_name', type = str, default = 'input_list.txt')
parser.add_argument('--data_path', type = str, default = '../data/')
parser.add_argument('--output_path', type = str, default = '../data/.tmp/process/')
parser.add_argument('--prompt_path', type = str, default = '../data/.tmp/extract/')

args = parser.parse_args()

##############################################################

MODEL_NAME = args.model
IS_MODEL_LOCAL = {"glm-4" : False, "llama-3-8B" : True}
MODEL = None
TOKENIZER = None
MODEL_PATH = args.model_path
MODEL_ID = {"llama-3-8B" : "Meta-Llama-3-8B-Instruct/"}
DATA_PATH = args.data_path
OUTPUT_PATH = args.output_path
if not os.path.exists(OUTPUT_PATH) :
  os.makedirs(OUTPUT_PATH)
PROMPT_PATH = args.prompt_path


input_file_name = args.input_file_name

##############################################################


def load() :
  global MODEL, TOKENIZER
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
  else :
    raise ValueError("Model not found")
  
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

# Ex msg
#     messages=[
#         {"role": "user", "content": "作为一名营销专家，请为智谱开放平台创作一个吸引人的slogan"},
#         {"role": "assistant", "content": "当然，为了创作一个吸引人的slogan，请告诉我一些关于您产品的信息"},
#         {"role": "user", "content": "智谱AI开放平台"},
#         {"role": "assistant", "content": "智启未来，谱绘无限一智谱AI，让创新触手可及!"},
#         {"role": "user", "content": "创造一个更精准、吸引人的slogan"}
#     ]

##############################################################

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
# print(prompts)

# Here is the version with both positive and negative prompts

# system_prompt = """
# You are a chatbot that summarizes a description of an audio to generate a prompt for image-generation 
# for this audio. To specify, your prompt should contain and only contain **two parts, 
# one positive and one negative**, according to the description. Each part should be comprised of some 
# seperated words or phrases, but not sentences. Note that you should combine the descriptions
# of all pieces of the music into one pair of prompts. 

# Negative prompt refers to features that
# you don't want to see in the image. Note that your prompt should be suitable
# for text-to-image-generation tasks. For example, your prompt should contain items to appear, background 
# color or people's appearance if needed; but your prompt should not contain descriptions of the audio, 
# like what instruments occur in the music or tempo, melody of the music; instead you should represent the 
# emotional information of the audio by imaging proper scenes in the image.

# Here are a few examples of prompts:

# good prompt example1:

# Positive: a man, masterpiece, best quality, high quality

# Negative: text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry

# good prompt example2:

# Positive: detailed and refined, Zero Two from the anime Darling in the Franxx, distinctive pink hair, mesmerizing green eyes, dynamic pose, showcasing her strong and fearless personality, anime style, 8k resolution, 16:9 aspect ratio, battlefield background symbolizing the constant fights she has to face, confident and determined expression, black background

# Negative: text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry

# """

# Here is the version with only positive prompt

system_prompt = """
You are a chatbot that summarizes a description of an audio to generate a prompt for image-generation 
for this audio. 

**Note that you should combine the descriptions
of all pieces of the music into one prompt, and never generate anything other than the prompt.** 

The prompt should be comprised of some 
seperated words or phrases, but not sentences. Note that your prompt should be suitable
for text-to-image-generation tasks. 

For example, your prompt should contain items to appear, background 
color, people's appearance if needed, and so on; but your prompt should not contain direct descriptions 
of the audio, 
like what instruments occur in the music or tempo, melody of the music.

You can represent the emotional information of the audio by adding proper items to the image. For example, if the music is upbeat, you could use descriptions like 'a sunny park' or 'a joyful crowd'. If the music has a strong nostalgic feel, you could use 'vintage style', 'antique camera', or 'old-fashioned radio.'

You can also use background color to convey emotional information. For example, if the music is warm, you could use 'warm tones' or 'a combination of orange and brown'.

Here are a few examples of prompts:

good prompt example1:

a man, masterpiece, best quality, high quality

good prompt example2:

Zero Two from the anime Darling in the Franxx, detailed and refined, distinctive pink hair, mesmerizing green eyes, dynamic pose, showcasing her strong and fearless personality, anime style, 8k resolution, 16:9 aspect ratio, battlefield background symbolizing the constant fights she has to face, confident and determined expression, black background

good prompt example3:

Jay Chou in a fantasy world, playing piano, 8k resolution, 16:9 aspect ratio, 60fps, with a dreamy aesthetic.

bad prompt example:

strong beats, female vocalist, pulsing synthesizers
"""

load()
token_spent = 0
print(type(MODEL), type(TOKENIZER))
for (prompt, file_name) in zip(prompts, audio_file_name) :
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": prompt},
  ]
  response, tokens = f_response(messages)
  with open(OUTPUT_PATH + file_name + ".prompt", "w") as f :
    f.write(response)
  token_spent += tokens
print("Token spent:", token_spent)
