import gradio as gr
import requests
import json
from io import  BytesIO
import base64
from PIL import Image

API_SERVER_URL = "http://localhost:54224/generate" 
# Don't forget to start your local API server

def generate(Music, Name, Prompt):
    # music type: Tuple[int, np.ndarray]
    
    # encode music into json
    encode_music = Music[1].tolist()
    data = {"sample_rate": Music[0], "music": encode_music, "music_name": Name, "prompt": Prompt}
    encoded = json.dumps(data).encode("utf-8")

    response = requests.post(API_SERVER_URL, data=encoded).json()

    # load response from model
    encoded_image = {}
    byte_arr = {}
    png = {}

    for i in range(8):
        encoded_image[i] = json.loads(response)[f'{i}']
        byte_arr[i] = base64.decodebytes(encoded_image[i].encode('ascii'))
        png[i] = Image.open(BytesIO(byte_arr[i]))

    return png[0], png[1], png[2], png[3], png[4], png[5], png[6], png[7]

gr.Interface(generate,
             title="Music to Image Transfer",
             description="Please provide music and prompt, we will generate image for you.",
             inputs=[gr.Audio("Music"), gr.Textbox(label="Music Name (optional)"), gr.Textbox(label="Prompt")],
             outputs=[gr.Image(label="Image 1"), gr.Image(label="Image 2"),
                      gr.Image(label="Image 3"), gr.Image(label="Image 4"),
                      gr.Image(label="Image 5"), gr.Image(label="Image 6"),
                      gr.Image(label="Image 7"), gr.Image(label="Image 8"),],
            ).launch(server_name="0.0.0.0")