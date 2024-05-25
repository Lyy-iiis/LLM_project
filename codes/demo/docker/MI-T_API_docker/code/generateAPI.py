import torch
import numpy as np
from diffusers import DiffusionPipeline
         
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from io import  BytesIO
import base64
import json
# import torchaudio
from scipy.io.wavfile import write

import argparse

parser = argparse.ArgumentParser(description='None')

parser.add_argument('--mp3path', '-m', default='/ssdshare/MI-T/music/music.mp3')

args = parser.parse_args()

# MP3_PATH = "/root/LLM_project/codes/demo/music.mp3"
MP3_PATH = args.mp3path

app = FastAPI()

class ChatRequest(BaseModel):
    sample_rate: int
    music: list
    prompt: str

pipe = DiffusionPipeline.from_pretrained(
    "/ssdshare/LLMs/playground-v2.5-1024px-aesthetic/",
    custom_pipeline = "/ssdshare/MI-T/lpw_stable_diffusion_xl.py",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

@app.post("/generate")
async def process_chat(prompt: ChatRequest):
    music = (prompt.sample_rate, np.array(prompt.music))

    # print(music.shape)
    # print(music)
    # # Convert the music to MP3 format

    # torchaudio.save(MP3_PATH, music, 
    #                 sample_rate=prompt.sample_rate, format="wav")
    # torchaudio.save(MP3_PATH, music, 
    #                 sample_rate=prompt.sample_rate)
    # music = torchaudio.load(MP3_PATH)[0]

    write(MP3_PATH, music[0], music[1].astype(np.int16)) # save as .wav


    user_prompt = prompt.prompt
    
    # You should feed music and user_prompt into main function here
    # modify these two lines after finish main
    image = pipe(prompt=user_prompt,
                num_inference=50,guidance_scale=3).images[0]
    description = prompt.prompt

    # return image and description to the frontend
    byte_arr = BytesIO()
    image.save(byte_arr, format='PNG')
    encoded_image = base64.encodebytes(byte_arr.getvalue()).decode('ascii')

    return json.dumps({'image': encoded_image, 'text': description})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=54224, workers=1)