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
parser.add_argument('--inprompt_path', type = str, default = '.tmp/inprompt/')

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
INPROMPT_PATH = args.inprompt_path


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

# system_prompt = """
# You are a chatbot that summarizes a description of an audio to generate a prompt for image-generation 
# for this audio. 

# **Note that you should combine the descriptions
# of all pieces of the music into one prompt, and never generate anything other than the prompt.** 

# The prompt should be comprised of some 
# seperated words or phrases, but not sentences. Note that your prompt should be suitable
# for text-to-image-generation tasks. 

# For example, your prompt should contain items to appear, background 
# color, people's appearance if needed, and so on; but your prompt should not contain direct descriptions 
# of the audio, 
# like what instruments occur in the music or tempo, melody of the music.

# You can represent the emotional information of the audio by adding proper items to the image. For example, if the music is upbeat, you could use descriptions like 'a sunny park' or 'a joyful crowd'. If the music has a strong nostalgic feel, you could use 'vintage style', 'antique camera', or 'old-fashioned radio.'

# You can also use background color to convey emotional information. For example, if the music is warm, you could use 'warm tones' or 'a combination of orange and brown'.

# Here are a few examples of prompts:

# good prompt example1:

# a man, masterpiece, best quality, high quality

# good prompt example2:

# Zero Two from the anime Darling in the Franxx, detailed and refined, distinctive pink hair, mesmerizing green eyes, dynamic pose, showcasing her strong and fearless personality, anime style, 8k resolution, 16:9 aspect ratio, battlefield background symbolizing the constant fights she has to face, confident and determined expression, black background

# good prompt example3:

# Jay Chou in a fantasy world, playing piano, 8k resolution, 16:9 aspect ratio, 60fps, with a dreamy aesthetic.

# bad prompt example:

# strong beats, female vocalist, pulsing synthesizers, catchy melody

# These prompts are bad because they describe the music directly, rather than the image you want to generate.
# """

system_prompt = """
You are a chatbot that summarizes a description of an audio to generate a prompt for image-generation 
for this audio. 

**Note that you should combine the descriptions
of all pieces of the music into one prompt, and never generate anything other than the prompt.** 

The prompt should be comprised of some seperated words or phrases, but not sentences. 

For example, your prompt should contain items to appear, background 
color, people's appearance if needed, and so on; but your prompt **should not** contain direct descriptions 
of the audio, such as "strong beats, female vocalist, pulsing synthesizers, catchy melody", or instruments
like "piano, synthesizer bass, energetic drumming".

You can represent the emotional information of the audio by adding proper items to the image. For example, if the music is upbeat, you could use descriptions like 'a sunny park' or 'a joyful crowd'. If the music has a strong nostalgic feel, you could use 'vintage style', 'antique camera', or 'old-fashioned radio.'

You can also use background color to convey emotional information. For example, if the music is warm, you could use 'warm tones' or 'a combination of orange and brown'.

Note that the if the name of the audio is provided, your image should be related to it.

Here are a few examples of prompts:

<example1>:

<input>: The name of the audio is "Burn". Please generate a image of 8k resolution, 16:9 aspect ratio, 60fps.

This music is cut into 6 pieces. Each piece has a length of 30 seconds and an overlap of 5 seconds. The description of each piece is as follows:
Description piece 1: A pop/EDM instrumental with a fast tempo, featuring a repetitive piano melody, synthesizer bass, and energetic drumming. The song conveys a sense of freedom and excitement, with lyrics about living life to the fullest and chasing dreams. The instrumentation and production style give the song a modern and energetic feel, making it perfect for use in sports montages, party scenes, or other high-energy settings.
Description piece 2: This song is an upbeat electronic dance track with a catchy melody and a strong beat. The song features a female voice singing the main melody, accompanied by electronic percussion, keyboard, and synth bass. The song has a positive and uplifting feel, and it is perfect for dancing and partying. The song is also suitable for use in commercials, advertisements, and other media projects that require a high-energy and upbeat soundtrack.
Description piece 3: This is a high-energy, fast-paced dance track with a strong beat and pulsing synthesizers. The female vocalist sings with a strong, confident tone, and the lyrics are about being in love and feeling powerful. The music is perfect for a dance club or a high-energy sports event. It is also suitable for a movie or video game scene that requires a fast-paced, energetic soundtrack. Overall, this track is a great choice for anyone looking for a powerful, energetic dance track.
Description piece 4: This is a song whose genre is Electronic, and the lyrics are "When you pray for me".
Description piece 5: This is a high energy, intense dubstep track with a strong bassline, heavy drums, and a female vocal sample. The song has a strong beat and is very catchy. It would be great for a dance scene in a movie or video game.
Description piece 6: This is a dance track that features a female vocal singing over a loud and energetic beat. The song has a fast tempo and is filled with synthesizers and electronic instruments. The lyrics are in a foreign language and are not understandable. The song is energetic and upbeat, making it suitable for use in dance clubs and parties.

The lyrics are as follows:
Living in the clouded dream
Searching for the quiet that you need to breathe
Gave up on your sanity to hide behind your shadow
While you tried to take the sun down
Hearts will never change to gold
Out there thinking that you're in the world alone
No one ever told you that you'll have to fight for something
Or you never learn to balance
It's too cold
Standing in the middle of the downfall
Looking in the mirror cuz it's only you
It's too cold
Standing in the middle of the downfall
Looking in the mirror cuz it's only you
When you crash and burn

<output>: a girl dressed in red, holding a blanket, red or yellow background representing fire, mystical warmth, 8k resolution, 16:9 aspect ratio, 60fps

<example2>:

<input>: The name of the audio is "infinity heaven". Please generate a image of 8k resolution, 16:9 aspect ratio, 60fps. Animation style.

This music is cut into 6 pieces. Each piece has a length of 30 seconds and an overlap of 5 seconds. The description of each piece is as follows:
Description piece 1: A fast-paced, energetic track with a strong beat, powerful synths, and piano.
Description piece 2: This fast-paced electronic track features a variety of synthesizers and drums. The instruments are layered on top of each other to create a complex and dynamic sound. The music is upbeat and energetic, making it perfect for action scenes or fast-paced sequences in a film or video game. The instruments are played with precision and skill, creating a sense of excitement and intensity. Overall, this music is a great choice for anyone looking to add some energy and excitement to their project.
Description piece 3: This fast-paced, intense and dynamic track is a perfect fit for any action-packed, high-energy project. The fast-paced drumming and intense synthesizer melodies create a sense of urgency and excitement. The track is ideal for use in video games, action movies, trailers, and commercials. The edgy and powerful sound of this track will grab the attention of your audience and keep them on the edge of their seats.
Description piece 4: This is a fast-paced, energetic electronic track featuring piano, synthesizers, and drums. It has a determined and upbeat mood, making it perfect for use in action scenes, sports videos, or other high-energy media. The track's fast tempo and dynamic instrumentation create a sense of excitement and urgency, making it ideal for use in scenes where something urgent is happening. Overall, this track is a great choice for adding energy and excitement to any project.
Description piece 5: This is a techno trance piece that is energetic, exciting, and uplifting. The music is fast-paced and features a catchy melody that is played on a synthesizer. The tempo is fast and the music is intense and driving. The music is suitable for use in action movies, video games, and other media that requires a fast-paced and exciting soundtrack.
Description piece 6: This is a short piano piece that evokes a sense of nostalgia and longing. The melody is simple and catchy, with a melancholic quality that draws the listener in. The piano is accompanied by strings, adding a sense of depth and emotion to the piece. The overall atmosphere is one of reflection and contemplation. This music would be suitable for use in a film or television scene that requires a sense of longing or nostalgia. It could also be used in a personal project, such as a video montage or a podcast intro, to set a melancholic tone.

<output>: angel with white wings, dressed in flowing white garment, purity, soft color, stars, anime, 8k resolution, 16:9 aspect ratio, 60fps

"""


load()
token_spent = 0
print(type(MODEL), type(TOKENIZER))
for (prompt, file_name) in zip(prompts, audio_file_name) :
  with open(DATA_PATH + INPROMPT_PATH + file_name + ".prompt", "r") as f :
    inprompt = f.read()
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": inprompt+'\n'+prompt},
  ]
  response, tokens = f_response(messages)
  with open(OUTPUT_PATH + file_name + ".prompt", "w") as f :
    f.write(response)
  token_spent += tokens
print("Token spent:", token_spent)
