from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI
import argparse 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess
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

def run_llama3_70b(input_file_name, output_file_name):
  server = "/share/ollama/ollama serve"
  user = f"/share/ollama/ollama run llama3:70b < \"{input_file_name}\" > \"{output_file_name}\""
  print("Loading model")
  process_server = subprocess.Popen(server, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  process_user = subprocess.Popen(user, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process_user.communicate()
  # return_code_1 = process_server.returncode
  return_code_2 = process_user.returncode
  print(f"User return code: {return_code_2}")
  print(f"User output: {stdout.decode()}")
  print(f"User error: {stderr.decode()}")

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


system_prompt = """
You are a chatbot that summarizes a description of an audio to generate a prompt for image-generation 
for this audio. 

**Note that you should combine the descriptions of all pieces of the music into one prompt, and never generate anything other than the prompt.** 

The prompt should be comprised of some seperated words or phrases, but not sentences. Besides, don't use any Chinese characters in the prompt even if the audio is in Chinese.

You should try to understand the emotional information of the audio and generate a prompt that conveys this information.

For example, your prompt should contain items to appear, background color, people's appearance if needed, and so on; but your prompt **should not** contain direct descriptions of the audio, such as "strong beats, female vocalist, pulsing synthesizers, catchy melody", or instruments like "piano, synthesizer bass, energetic drumming". The prompt should NOT CONTAIN MORE THAN ONE CHARACTER, "couples walk together in the park", "a group of friends having a picnic" is forbidden. UNLESS USER SPECIFICALLY ASKED FOR IT.

You can represent the emotional information of the audio by adding proper items to the image. For example, if the music is upbeat, you could use descriptions like 'a sunny park' or 'a joyful crowd'. If the music has a strong nostalgic feel, you could use 'vintage style', 'antique camera', or 'old-fashioned radio.'

You can also use background color to convey emotional information. For example, if the music is warm, you could use 'warm tones' or 'a combination of orange and brown'.

Note that the if the name of the audio is provided, your image should be related to it.

Here are a few examples of prompts:

<example1>:

<input>: The name of the audio is "Burn". Please generate an image of 8k resolution, 16:9 aspect ratio, 60fps.

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

<output>: vibrant and dynamic scene with a central figure, anime-style character, anime fashion, a girl with with long flowing golden hair, winter outfit, red hooded jacket with white fur trim at the collar and cuffs, a warm look, left hand holding a basket, focused gaze, one eye visible and the other obscured by the hair, a determined or intense expression, various abstract shapes and elements in background, shades of orange yellow and red, sense of fire or energy, moon, circles of light scattered throughout, mystical atmosphere, 8k resolution, 16:9 aspect ratio, 60fps

<example 2>

<input>: The name of the audio is 晴天.

<output>: This music is cut into 11 pieces. Each piece has a length of 30 seconds and an overlap of 5 seconds. The description of each piece is as follows:
Description piece 1: This is a song whose genre is Pop, and the lyrics are "故事的小黄花".
Description piece 2: This is a song whose genre is Pop, and the lyrics are "童年的荡秋千 随记忆一直晃到现在 吹着前奏望着天空".
Description piece 3: This is a song whose genre is Pop, and the lyrics are "我想起花瓣试着掉落 为你翘课的那一天 花落的那一天 教室的那一间 我怎么看不见 消失的下雨天 我好想再淋一遍".
Description piece 4: This is a song whose genre is Pop, and the lyrics are "没想到失去的勇气我还留着 好想再问一遍 你会等待还是离开 刮风这天 我试过握着你手 但偏偏 雨渐渐 大到我看你不见".
Description piece 5: This is a song whose genre is Pop, and the lyrics are "在你身边的那幅风景的那天 也许我会比较好一点 从前从前 有个人爱你很久 但渐渐风渐渐把距离吹的好远 好不容易 我们再多爱一天".
Description piece 6: This is a song whose genre is Pop, and the lyrics are "还要多久 我才能在你身边
等到放晴的那天 也许我会比较好一点 从前从前 有个人爱你很久 但偏偏 风渐渐 把距离吹得好远 好不容易 又能再多爱一天 但故事的最后 你好像还是说了拜拜".
Description piece 7: This is a song whose genre is Pop, and the lyrics are "刮风这天 我试过握着你手".
Description piece 8: This is a song whose genre is Pop, and the lyrics are "但偏偏 雨渐渐 大到我看你不见 还要多久 我才能够在你身边 等到放晴的那天 也许 我会比较好一点".
Description piece 9: This is a song whose genre is Pop, and the lyrics are "等到放晴的那天 也许 我会比较好一点 从前从前 有个人爱你很久".
Description piece 10: This is a song whose genre is Pop, and the lyrics are "等到放晴的那天 也许 我会比较好一点 从前从前 有个人爱你很久 但偏偏 风渐渐 把距离吹得好远 好不容易 又能再多爱一天".
Description piece 11: This is a song whose genre is Pop, and the lyrics are "从前从前 有个人爱你很久 但偏偏 风渐渐 把距离吹得好远 好不容易 又能再多爱一天 但故事的最后 你好像还是说了拜拜".

The lyrics are as follows:
故事的小黄花 从出生那年就飘着
童年的荡秋千 随记忆一直晃到现在
吹着前奏望着天空
我想起花瓣试着掉落
为你翘课的那一天 花落的那一天
教室的那一间 我怎么看不见
消失的下雨天 我好想再淋一遍
没想到失去的勇气我还留着
好想再问一遍 你会等待还是离开
刮风这天 我试过握着你手
但偏偏 雨渐渐 大到我看你不见
还要多久 我才能在你身边
等到放晴的那天 也许我会比较好一点
从前从前 有个人爱你很久
但偏偏 风渐渐 把距离吹得好远
好不容易 又能再多爱一天
但故事的最后 你好像还是说了拜拜
刮风这天 我试过握着你手
但偏偏 雨渐渐 大到我看你不见
还要多久 我才能够在你身边
等到放晴的那天 也许 我会比较好一点
从前从前 有个人爱你很久
但偏偏 风渐渐 把距离吹得好远
好不容易 又能再多爱一天
但故事的最后 你好像还是说了拜拜

sunny day, yellow flowers, Swing set, blowing wind, falling petals, classroom, rainy day, courage, hands held, obscured by rain, waiting, clear skies, distant love, blowing wind, separation, reunion, goodbye, emotional atmosphere, vibrant colors, 8k resolution, 16:9 aspect ratio, 60fps.

<example3>:

<input>: The name of the audio is "infinity heaven". Please generate an image of 8k resolution, 16:9 aspect ratio, 60fps. Animation style.

This music is cut into 6 pieces. Each piece has a length of 30 seconds and an overlap of 5 seconds. The description of each piece is as follows:
Description piece 1: A fast-paced, energetic track with a strong beat, powerful synths, and piano.
Description piece 2: This fast-paced electronic track features a variety of synthesizers and drums. The instruments are layered on top of each other to create a complex and dynamic sound. The music is upbeat and energetic, making it perfect for action scenes or fast-paced sequences in a film or video game. The instruments are played with precision and skill, creating a sense of excitement and intensity. Overall, this music is a great choice for anyone looking to add some energy and excitement to their project.
Description piece 3: This fast-paced, intense and dynamic track is a perfect fit for any action-packed, high-energy project. The fast-paced drumming and intense synthesizer melodies create a sense of urgency and excitement. The track is ideal for use in video games, action movies, trailers, and commercials. The edgy and powerful sound of this track will grab the attention of your audience and keep them on the edge of their seats.
Description piece 4: This is a fast-paced, energetic electronic track featuring piano, synthesizers, and drums. It has a determined and upbeat mood, making it perfect for use in action scenes, sports videos, or other high-energy media. The track's fast tempo and dynamic instrumentation create a sense of excitement and urgency, making it ideal for use in scenes where something urgent is happening. Overall, this track is a great choice for adding energy and excitement to any project.
Description piece 5: This is a techno trance piece that is energetic, exciting, and uplifting. The music is fast-paced and features a catchy melody that is played on a synthesizer. The tempo is fast and the music is intense and driving. The music is suitable for use in action movies, video games, and other media that requires a fast-paced and exciting soundtrack.
Description piece 6: This is a short piano piece that evokes a sense of nostalgia and longing. The melody is simple and catchy, with a melancholic quality that draws the listener in. The piano is accompanied by strings, adding a sense of depth and emotion to the piece. The overall atmosphere is one of reflection and contemplation. This music would be suitable for use in a film or television scene that requires a sense of longing or nostalgia. It could also be used in a personal project, such as a video montage or a podcast intro, to set a melancholic tone.

<output>: two figures, one an angelic being, with large white wings behind, blonde hair, white dress or robe with delicate details and draped fabric, one arm extended upwards and fingers slightly curled, another figure smaller in scale, dressed in a traditional East Asian garmen style, high waistline and wide, flowing sleeves, white clothing with some blue accents and patterns, dark hair falling straight down his back, contrasting with the lighter tones of the angel, sense of admiration or curiosity, vibrant and intricate background, a mix of geometric shapes and celestial motifs, stars and moons, sunburst design, cosmic or mystical, rich with gold, green, blue, and hints of purple, feeling of otherworldly elegance and serenity, 8k resolution, 16:9 aspect ratio, 60fps


"""

with open(DATA_PATH + "style/style_description.txt", "r") as f :
  style_description = []
  style_num = 0
  for line in f :
    style_description.append(line.rstrip())
    style_num += 1

style_prompt = f"""
You are a chatbot that choose a style for the description.

The user input will be a description to the generated image, and you should choose a style closest to the description, from the following {style_num} descriptions of style images.

You should answer a number from 1 to {style_num} to choose the style.

Here are the descriptions of the styles:

"""
for i in range(style_num) :
  style_description[i] = style_description[i].split("%")[0]
  style_prompt += f"{style_description[i]}\n"

style_prompt += """

<EXAMPLE 1>:
<INPUT> : two figures, one an angelic being, with large white wings behind, blonde hair, white dress or robe with delicate details and draped fabric, one arm extended upwards and fingers slightly curled, another figure smaller in scale, dressed in a traditional East Asian garmen style, high waistline and wide, flowing sleeves, white clothing with some blue accents and patterns, dark hair falling straight down his back, contrasting with the lighter tones of the angel, sense of admiration or curiosity, vibrant and intricate background, a mix of geometric shapes and celestial motifs, stars and moons, sunburst design, cosmic or mystical, rich with gold, green, blue, and hints of purple, feeling of otherworldly elegance and serenity, 8k resolution, 16:9 aspect ratio, 60fps

<OUTPUT> : 6

<EXAMPLE 2>:
<INPUT> : vibrant and dynamic scene with a central figure, anime-style character, anime

<OUTPUT> : 1

"""

# if MODEL_NAME == "llama-3-70B" :
#   for (prompt, file_name) in zip(prompts, audio_file_name) :
#     with open(DATA_PATH + INPROMPT_PATH + file_name + ".prompt", "r") as f :
#       inprompt = f.read()
#     with open(DATA_PATH + INPROMPT_PATH + file_name + ".prompt_total", "w") as f :
#       f.write(system_prompt)
#       f.write(inprompt+'\n'+prompt)
#     print(DATA_PATH + INPROMPT_PATH + file_name + ".prompt_total")
#     run_llama3_70b(DATA_PATH + INPROMPT_PATH + file_name + ".prompt_total", OUTPUT_PATH + file_name + ".prompt")
#     exit(0)
# else:
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
  messages = [
      {"role": "system", "content": style_prompt},
      {"role": "user", "content": response},
  ]
  response, tokens = f_response(messages)
  token_spent += tokens
  num = re.findall(r'\b\d+\b', response)[0]
  with open(OUTPUT_PATH + file_name + ".style", "w") as f :
    f.write(num)

# print("Token spent:", token_spent)

##############################################################
# prompt for generating image with no character

system_prompt = """
You are a chatbot that summarizes a description of an audio to generate a prompt for image-generation  for this audio. 

**Note that you should combine the descriptions of all pieces of the music into one prompt, and never generate anything other than the prompt.** 

You should try to understand the emotional information of the audio and generate a prompt that conveys this information.

The prompt should be comprised of some seperated words or phrases, but not sentences. And here the key point is that the image **should not contain any character**! Besides, don't use any Chinese characters in the prompt even if the audio is in Chinese.

For example, your prompt should contain background color, items to appear, their color or features, and so on; but your prompt **should not** contain direct descriptions of the audio, such as "strong beats, female vocalist, pulsing synthesizers, catchy melody", or instruments like "piano, synthesizer bass, energetic drumming". The prompt should NOT CONTAIN ANY HUMAN as well, such as "a girl dressed in red, holding a blanket", "angel with white wings", "a robot with a sword" and so on.

You can represent the emotional information of the audio by adding proper items to the image. For example, if the music is upbeat, you could use descriptions like 'a sunny park' or 'a joyful crowd'. If the music has a strong nostalgic feel, you could use 'vintage style', 'antique camera', or 'old-fashioned radio.'

You can also use background color to convey emotional information. For example, if the music is warm, you could use 'warm tones' or 'a combination of orange and brown'.

Note that the if the name of the audio is provided, your image should be related to it.

Here are a few examples of prompts:

<example1>:

<input>: The name of the audio is "Clock Paradox". Please generate an image of 8k resolution, 16:9 aspect ratio, 60fps.

This music is cut into 6 pieces. Each piece has a length of 30 seconds and an overlap of 5 seconds. The description of each piece is as follows:
Description piece 1: This is a dynamic and powerful track that is perfect for use in trailers, video games, and other media projects. The music features a blend of electronic and orchestral elements, with powerful drums, intense synths, and lush strings. The track is also very energetic and exciting, making it ideal for use in action scenes and other high-energy moments. Overall, this music is perfect for adding a sense of excitement and intensity to any project.
Description piece 2: This is a dynamic, powerful, and energetic electronic music track featuring synthesizers, percussion, and bass. The music is uplifting, energetic, and inspiring, making it perfect for use in sports, fitness, and workout videos, as well as in corporate and business presentations, advertising and marketing campaigns, and other media projects that require a strong and inspiring soundtrack.
Description piece 3: Uplifting energetic melodic track with a strong bass line, plucks, and drums.
Description piece 4: A powerful and energetic music track with synthesizers, strings, piano, and drums. The epic and uplifting mood will evoke feelings of joy and happiness. It will certainly work well with corporate videos, business projects, presentations, commercials, advertising, TV ads, YouTube videos, vlogs, and more.
Description piece 5: This is a powerful and epic electronic music track with a strong melody and driving rhythm. The track features piano, strings, brass, and powerful drums. It has a modern and energetic sound and is perfect for use in trailers, video games, and other media projects that require a powerful and epic soundtrack.
Description piece 6: This is a high-energy, action-packed, electronic rock track. The song features electric guitars, synthesizers, drums, and a catchy melody. The song is perfect for action movies, video games, and trailers. The song is energetic, upbeat, and exciting. The song is a mix of electronic and rock music. The song is intense, and it has a strong beat. The song is perfect for action movies, video games, and trailers.

<output>: a large circular, unusual-designed clock with multiple layers, golden rings around it, fragmented and broken frames or portals to depict alternate timelines or dimensions, abstract shapes and lines connecting different elements, cool blues and grays color, soft yet dramatic lighting, light sources coming from above and behind the central clock, casting shadows and highlights, sense of motion, complexity, mystery, and thought-provoking contemplation, science fiction, 8k resolution, 16:9 aspect ratio, 60fps

<example2>:

<input>: Please generate an image of 8k resolution, 16:9 aspect ratio, 60fps. Animation style.

This music is cut into 7 pieces. Each piece has a length of 30 seconds and an overlap of 5 seconds. The description of each piece is as follows:
Description piece 1: The music is a fast-paced electronic dance music with distorted guitar and synthesizer riffs, fast-paced drums, and glitchy effects. It is intense, aggressive, and energetic, with a sense of urgency and excitement. The music is suitable for action scenes, extreme sports, and high-energy content. It is also well-suited for video games and other forms of media that require fast-paced and intense music.
Description piece 2: This is a fast-paced electronic instrumental. The music is fast tempo with synthesizer, drum machine, and other electronic instruments. The music is very loud and the instruments are distorted. The music is aggressive and energetic. The music is suitable for use in a video game or in a movie scene where there is a fast-paced action scene.
Description piece 3: This is a techno music piece. It is fast-paced and has a glitchy, glitchy feel. The instruments used are synthesizers and electronic drums. The mood of this piece is energetic and intense. It would be suitable for use in a video game or an action movie.
Description piece 4: This is a high energy electro dubstep track with a lot of drive and energy. It has a lot of hard hitting drums and edgy bass lines. The piano and strings are used to add a sense of emotion and depth to the track. It is perfect for use in a video game, film, or any other project that needs a high energy soundtrack.
Description piece 5: This is a fast-paced, high-energy electronic track that is sure to get your heart racing. The driving beat and distorted synthesizers create a sense of urgency and excitement, making it perfect for action scenes or high-energy activities. The fast tempo and complex rhythms add to the intensity of the track, making it a great choice for use in extreme sports or video games. Overall, this music is intense, fast-paced, and highly energetic, making it a great choice for any project that needs a boost of energy.
Description piece 6: This is a techno piece that is fast-paced and energetic. It features a lot of glitchy, distorted sounds and electronic beats. It sounds like something you would hear at a techno club.
Description piece 7: A high energy, powerful and aggressive metal track. This is the ideal soundtrack for extreme sports, fight scenes, car chases, war and battle footage, as well as for a variety of other high intensity applications.

<output>: abstract graphical elements throughout the image that resemble digital glitches, distortion effects, shades of blue and purple with some pink highlights, feeling of coolness and futurism, gradient transitioning from dark at the top to lighter colors towards the bottom, space-like or digital atmosphere, a slender beam of light on the right running vertically downwards, ray of hope or contrast, dynamic and chaotic, rapid movement, information overload, digital apocalypse, 8k resolution, 16:9 aspect ratio, 60fps

<example 3>

<input>: The name of the audio is 晴天.

<output>: This music is cut into 11 pieces. Each piece has a length of 30 seconds and an overlap of 5 seconds. The description of each piece is as follows:
Description piece 1: This is a song whose genre is Pop, and the lyrics are "故事的小黄花".
Description piece 2: This is a song whose genre is Pop, and the lyrics are "童年的荡秋千 随记忆一直晃到现在 吹着前奏望着天空".
Description piece 3: This is a song whose genre is Pop, and the lyrics are "我想起花瓣试着掉落 为你翘课的那一天 花落的那一天 教室的那一间 我怎么看不见 消失的下雨天 我好想再淋一遍".
Description piece 4: This is a song whose genre is Pop, and the lyrics are "没想到失去的勇气我还留着 好想再问一遍 你会等待还是离开 刮风这天 我试过握着你手 但偏偏 雨渐渐 大到我看你不见".
Description piece 5: This is a song whose genre is Pop, and the lyrics are "在你身边的那幅风景的那天 也许我会比较好一点 从前从前 有个人爱你很久 但渐渐风渐渐把距离吹的好远 好不容易 我们再多爱一天".
Description piece 6: This is a song whose genre is Pop, and the lyrics are "还要多久 我才能在你身边
等到放晴的那天 也许我会比较好一点 从前从前 有个人爱你很久 但偏偏 风渐渐 把距离吹得好远 好不容易 又能再多爱一天 但故事的最后 你好像还是说了拜拜".
Description piece 7: This is a song whose genre is Pop, and the lyrics are "刮风这天 我试过握着你手".
Description piece 8: This is a song whose genre is Pop, and the lyrics are "但偏偏 雨渐渐 大到我看你不见 还要多久 我才能够在你身边 等到放晴的那天 也许 我会比较好一点".
Description piece 9: This is a song whose genre is Pop, and the lyrics are "等到放晴的那天 也许 我会比较好一点 从前从前 有个人爱你很久".
Description piece 10: This is a song whose genre is Pop, and the lyrics are "等到放晴的那天 也许 我会比较好一点 从前从前 有个人爱你很久 但偏偏 风渐渐 把距离吹得好远 好不容易 又能再多爱一天".
Description piece 11: This is a song whose genre is Pop, and the lyrics are "从前从前 有个人爱你很久 但偏偏 风渐渐 把距离吹得好远 好不容易 又能再多爱一天 但故事的最后 你好像还是说了拜拜".

The lyrics are as follows:
故事的小黄花 从出生那年就飘着
童年的荡秋千 随记忆一直晃到现在
吹着前奏望着天空
我想起花瓣试着掉落
为你翘课的那一天 花落的那一天
教室的那一间 我怎么看不见
消失的下雨天 我好想再淋一遍
没想到失去的勇气我还留着
好想再问一遍 你会等待还是离开
刮风这天 我试过握着你手
但偏偏 雨渐渐 大到我看你不见
还要多久 我才能在你身边
等到放晴的那天 也许我会比较好一点
从前从前 有个人爱你很久
但偏偏 风渐渐 把距离吹得好远
好不容易 又能再多爱一天
但故事的最后 你好像还是说了拜拜
刮风这天 我试过握着你手
但偏偏 雨渐渐 大到我看你不见
还要多久 我才能够在你身边
等到放晴的那天 也许 我会比较好一点
从前从前 有个人爱你很久
但偏偏 风渐渐 把距离吹得好远
好不容易 又能再多爱一天
但故事的最后 你好像还是说了拜拜

sunny sky, yellow flowers, swinging on a playground, memories, falling petals, rainy day, wind, hidden visibility, waiting for clear skies, emotional distance, love story, courage, 8k resolution, 16:9 aspect ratio, 60fps
"""

# The second example is DESTRUCTION 3,2,1

# load()
for (prompt, file_name) in zip(prompts, audio_file_name) :
  with open(DATA_PATH + INPROMPT_PATH + file_name + ".prompt", "r") as f :
    inprompt = f.read()
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": inprompt+'\n'+prompt},
  ]
  response, tokens = f_response(messages)
  with open(OUTPUT_PATH + file_name + ".prompt2", "w") as f :
    f.write(response)
  token_spent += tokens
  messages = [
      {"role": "system", "content": style_prompt},
      {"role": "user", "content": response},
  ]
  response, tokens = f_response(messages)
  token_spent += tokens
  num = re.findall(r'\b\d+\b', response)[0]
  with open(OUTPUT_PATH + file_name + ".style2", "w") as f :
    f.write(num)
print("Token spent:", token_spent)
