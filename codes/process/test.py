from dotenv import load_dotenv
import os
from zhipuai import ZhipuAI
import re
import random

load_dotenv()
zhipuai_api_key = os.environ.get("ZHIPUAI_API_KEY")
MODEL = ZhipuAI(api_key = zhipuai_api_key)

def f_response(messages) :
    response = MODEL.chat.completions.create(
        model = "glm-4",
        messages = messages
    )
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return content, tokens

style_dicription = {}
with open("../data/style/style_description.txt", "r") as f :
    style_num = 1
    for line in f :
        style_dicription[style_num] = line.rstrip()
        style_num += 1

style_prompt = f"""
You are a chatbot that choose a proper style for the description given by user, the description is about a song.

The user input will be a description to the generated image, and you should choose a style closest to the description, from the following 2 descriptions of style images.

You should ONLY answer a number from 1 or 2 to choose the style.

Here are the descriptions of the styles:

"""

prompt = """
The description is: 

heavenly landscape, two figures in the center, angelic being with large wings, glowing aura, intense gaze, mortal figure in awe, fast-paced motion lines, ethereal light, synthesizer-like patterns in the background, celestial bodies, stars and nebulae, gold and purple, contrasting bright and dark areas, emotional atmosphere, dynamic composition, 8k resolution, 16:9 aspect ratio, 60fps.

"""

inprompt = "The name of this song is \"Infinity Heaven\"."

def binary_ask(i,j,inprompt, prompt):
    prompt_ij = style_prompt
    prompt_ij += "1. " + style_dicription[i] + "\n"
    prompt_ij += "2. " + style_dicription[j] + "\n"
    messages = [
        {"role": "system", "content": prompt_ij},
        {"role": "user", "content": inprompt+'\n'+prompt},
    ]
    # print(messages)
    response, tokens = f_response(messages)
    number = re.findall(r'\d+', response)[0]
    print(number)
    if number != '1' and number != '2':
        binary_ask(i,j,inprompt, prompt)
    return int(number), tokens

def binary_ask_test(i,j):
    return random.randint(1,2), 1

total_tokens = 0
best = [i for i in range(1, style_num)]
while len(best) > 1 :
    print(best)
    for i in range(1, len(best), 2) :
        result, tokens = binary_ask(best[i-1], best[i],inprompt, prompt)
        total_tokens += tokens
        best[i-1] = best[i-2+result]
    best = best[::2]
    total_tokens += tokens

print(total_tokens, best[0])