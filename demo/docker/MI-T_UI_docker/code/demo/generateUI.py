import gradio as gr
import requests
import json
from io import  BytesIO
import base64
import os
from PIL import Image

API_SERVER_URL = os.environ["API_URL"]
THEME = "NoCrypt/miku"
# Don't forget to start your local API server

def generate(Music, Name, Prompt):
    # music type: Tuple[int, np.ndarray]
    
    # encode music into json

    encode_music = base64.b64encode(Music[1].tobytes()).decode("utf-8")
    data = {"sample_rate": Music[0], "music": encode_music, "music_name": Name, "prompt": Prompt}
    encoded = json.dumps(data).encode("utf-8")

    response = requests.post(API_SERVER_URL, data=encoded).json()

    # load response from model
    encoded_image = json.loads(response)
    byte_arr = {}
    png = {}

    
    for i in range(4):
        byte_arr[i] = base64.decodebytes(encoded_image[i.__str__()].encode('ascii'))
        png[i] = Image.open(BytesIO(byte_arr[i]))

    combined_image = Image.new('RGB', (png[0].width * 2, png[0].height * 2))

    for i in range(2):
        combined_image.paste(png[i], (0, i * png[0].height))
        combined_image.paste(png[i + 2], (png[0].width, i * png[0].height))
    
    return combined_image

def get_theme():
    os.environ["http_proxy"] = "http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ["https_proxy"] = "http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ["all_proxy"] = "socks5://Clash:QOAF8Rmd@10.1.0.213:7893"
    my_theme = gr.Theme.from_hub(THEME)
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["all_proxy"] = ""
    return my_theme

gr.Interface(generate,
             title="Music to Image Transfer",
             description="Please provide music and prompt, we will generate image for you.",
             inputs=[gr.Audio("Music"), gr.Textbox(label="Music Name (Suggested)"), 
                     gr.Textbox(label="Prompt (optional)")],
            outputs=gr.Image(label="Generated Image"),
            theme=get_theme(),
            ).launch(server_name="0.0.0.0")