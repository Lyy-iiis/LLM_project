import gradio as gr
import requests
import json
from io import  BytesIO
import base64
from PIL import Image

API_SERVER_URL = "http://localhost:54224/generate" 
# Don't forget to start your local API server

def generate(music, prompt):
    # music type: Tuple[int, np.ndarray]
    
    # encode music into json
    encode_music = music[1].tolist()
    data = {"sample_rate": music[0], "music": encode_music, "prompt": prompt}
    encoded = json.dumps(data).encode("utf-8")

    response = requests.post(API_SERVER_URL, data=encoded).json()

    # load response from model
    encoded_image = json.loads(response)['image']
    description = json.loads(response)['text']
    byte_arr = base64.decodebytes(encoded_image.encode('ascii'))
    png = Image.open(BytesIO(byte_arr))

    return png, description

gr.Interface(generate,
             inputs=[gr.Audio("Music"), "text"],
             outputs=[gr.Image(label="Image"), gr.Textbox(label="description", lines=3)]).launch()