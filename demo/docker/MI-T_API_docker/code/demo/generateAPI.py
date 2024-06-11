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
MUSIC_PATH = os.getcwd() + "/data/demo/music/"
IMAGE_PATH = os.getcwd() + "/data/demo/.tmp/style_transfer/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

class ChatRequest(BaseModel):
    sample_rate: int
    music: str
    music_name: str
    prompt: str

# cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()

@app.post("/generate")
async def process_chat(prompt: ChatRequest):
    print("Received request")
    music = base64.b64decode(prompt.music.encode("utf-8"))
    music = (prompt.sample_rate, np.frombuffer(music, dtype=np.int32))
    music_name = prompt.music_name
    user_prompt = prompt.prompt
    # Convert the music to WAV format
    if not os.path.exists(MUSIC_PATH):
        os.makedirs(MUSIC_PATH)
        
    write(MUSIC_PATH + "music.wav", music[0], music[1].astype(np.int16))
    # print("ajdlnvk;snv", music_name)
    # print(type(music_name))
    with open(MUSIC_PATH + "prompt.txt", "w") as f:
        if music_name == "":
            f.write("not provided\n")
        else:
            f.write("The name of the music is " + music_name + "\n")
    with open(MUSIC_PATH + "prompt.txt", "a") as f:
        f.write(user_prompt)
        
    os.system(f'python {CODE_PATH}/demo.py')

    image = []
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