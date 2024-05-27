import torch
import numpy as np
from diffusers import DiffusionPipeline
         
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from io import  BytesIO
import base64
import json
import os
from scipy.io.wavfile import write
from PIL import Image

CODE_PATH = os.getcwd()
MUSIC_PATH = CODE_PATH + "/codes/data/demo/music/"
IMAGE_PATH = CODE_PATH + "/codes/data/demo/.tmp/style_transfer/"

app = FastAPI()

class ChatRequest(BaseModel):
    sample_rate: int
    music: list
    prompt: str

# pipe = DiffusionPipeline.from_pretrained(
#     "/ssdshare/LLMs/playground-v2.5-1024px-aesthetic/",
#     # custom_pipeline = "/root/LLM_project/codes/generate/lpw_stable_diffusion_xl.py",
#     custom_pipeline = "/ssdshare/MI-T/lpw_stable_diffusion_xl.py",
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")

@app.post("/generate")
async def process_chat(prompt: ChatRequest):
    music = (prompt.sample_rate, np.array(prompt.music))
    user_prompt = prompt.prompt
    # Convert the music to WAV format
    if not os.path.exists(MUSIC_PATH):
        os.makedirs(MUSIC_PATH)
        
    write(MUSIC_PATH + "music.wav", music[0], music[1].astype(np.int16))
    with open(MUSIC_PATH + "prompt.txt", "w") as f:
        f.write(user_prompt)
    
    # You should feed music and user_prompt into main function here
    # modify these two lines after finish main
    # image = pipe(prompt=user_prompt,
    #             num_inference=50,guidance_scale=3).images[0]
    # description = prompt.prompt
    os.system(f'python {CODE_PATH}/demo/demo.py')
    
    for filename in os.listdir(IMAGE_PATH):
        if filename.startswith("0"):
            image_1 = Image.open(IMAGE_PATH + filename)
        elif filename.startswith("1"):
            image_2 = Image.open(IMAGE_PATH + filename)

    # return image and description to the frontend
    byte_arr = BytesIO()
    image_1.save(byte_arr, format='PNG')
    encoded_image_1 = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    byte_arr = BytesIO()
    image_2.save(byte_arr, format='PNG')
    encoded_image_2 = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    return json.dumps({'image_1': encoded_image_1, 'image_2': encoded_image_2})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=54224, workers=1)