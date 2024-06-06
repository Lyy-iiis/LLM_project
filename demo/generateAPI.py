import torch
import numpy as np
import torchvision.models as models
         
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MUSIC_PATH = CODE_PATH + "/codes/data/demo/music/"
IMAGE_PATH = CODE_PATH + "/codes/data/demo/.tmp/style_transfer/"
IMAGE_PATH_N = CODE_PATH + "/codes/data/demo/.tmp/generate/"

app = FastAPI()

class ChatRequest(BaseModel):
    sample_rate: int
    music: list
    prompt: str

cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()

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
        
    os.system(f'python {CODE_PATH}/demo/demo.py')

    image = []
    for filename in os.listdir(IMAGE_PATH_N + "music/"):
        if filename.endswith(".png"):
            image.append(Image.open(IMAGE_PATH_N + "music/" + filename))
    for filename in os.listdir(IMAGE_PATH + "music/"):
        if filename.endswith(".png"):
            image.append(Image.open(IMAGE_PATH + "music/" + filename))

    encoded_image = {}
    for i in range(len(image)):
        byte_arr = BytesIO()
        image[i].save(byte_arr, format='PNG')
        encoded_image[i] = base64.encodebytes(byte_arr.getvalue()).decode('ascii')

    return json.dumps(encoded_image)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=54224, workers=1)