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
    encoded_image_1 = json.loads(response)['image_1']
    encoded_image_2 = json.loads(response)['image_2']
    # description = json.loads(response)['text']
    byte_arr_1 = base64.decodebytes(encoded_image_1.encode('ascii'))
    png_1 = Image.open(BytesIO(byte_arr_1))
    byte_arr_2 = base64.decodebytes(encoded_image_2.encode('ascii'))
    png_2 = Image.open(BytesIO(byte_arr_2))

    return png_1, png_2

gr.Interface(generate,
             inputs=[gr.Audio("Music"), "text"],
             outputs=[gr.Image(label="Image 1"), gr.Image(label="Image 2")],
                    #   gr.Textbox(label="description", lines=3)]
            ).launch(server_name="0.0.0.0")
            # ).queue(default_concurrency_limit=50).launch(server_name="0.0.0.0", share=True, max_threads=50)