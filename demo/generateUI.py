import gradio as gr
import requests
import json
from io import  BytesIO
import base64
import os
from PIL import Image

API_SERVER_URL = "http://localhost:54224/generate"
THEME = "NoCrypt/miku"
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

    combined_image = Image.new('RGB', (png[0].width * 2, png[0].height * 4))

    for i in range(4):
        combined_image.paste(png[i], (0, i * png[0].height))
        combined_image.paste(png[i + 4], (png[0].width, i * png[0].height))
    
    # return png[0], png[1], png[2], png[3], png[4], png[5], png[6], png[7]
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
            #  outputs=[gr.Image(label="Image 1"), gr.Image(label="Image 2"),
            #           gr.Image(label="Image 3"), gr.Image(label="Image 4"),
            #           gr.Image(label="Image 5"), gr.Image(label="Image 6"),
            #           gr.Image(label="Image 7"), gr.Image(label="Image 8"),],
            outputs=gr.Image(label="Generated Image"),
            # theme="compact"
            theme=get_theme(),
            ).launch(server_name="0.0.0.0")