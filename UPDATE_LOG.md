## 24.05.25 19:35 SQA

- Complete a UI using docker, but still long way to go.

You can use it by first downloading the demo directory to your local machine, then `cd demo/docker/MI-T_helm_template`, and then __remember to change the `NameSpace` in `project.yaml`__. Now you might use the __makefile__ I write for you: (remember to first helm delete your own pod)

```
make podup
make connect
```

The first command will create the API & UI pod, and the second command will connect these two. Then you can use the UI by visiting `http://sqabuhui.ddns.net:9443`.

After using you can do

```
make poddown
```

which will delete the pods and the connection.

- To use your own ip, modify `IngressHost: sqabuhui.ddns.net` in `./project.yaml`.

- The containers I build is all in `harbor.ai.iiis.co:9443/llmproject4sqalyyxqc`, in which there are `/api/try{x}` or `/ui/try{x}` (You see, this cannot be a good name, though.) To modify the code (also the dockerfile or requirments), push your own docker image and modify the `project.yaml` to use your own image.

- The `apiurl: http://api-service:54224/generate` line in `project.yaml`, you had better not modify this unless talk to me, since this is important why this can work.

- Note that in my implementation, I modify lyy's code so that in API, the music will be saved into `MP3_PATH: /ssdshare/MI-T/music/music.mp3` which you may modify. __But actually it should be `.wav`__. (Sorry for the mistake and I will fix it later...) So the expected type will be `.wav`. __TODO:__ implement the rest of our model based on the music saved.

The final part is the biggest problem, which I have not figured out why, in that __one cannot upload a large music in the gradio__ (tested, 10s music is definitely fine, but `Burn` is too large). 

The error is contained in the `error.txt`.

Lyy's implementation does not have this problem, so I guess that the problem is either
1. on the gradio pod there is limitation of uploading
2. there is limitation on `sqabuhui.ddns.net`

That's the end and I love docker so much..


## 24.05.22 18:00 XQC

- Try to figure out why Qwen doesn't work sometimes. Probably related with luck, since this time when I changed back to 4 gpus, it still returned trash second time on the same gpu.


## 24.05.15 13:30 XQC

- Complete style image description. Replace all descriptions with gpt-4o.

## 24.05.13 16:00 XQC

- Add style images. Please `cd codes`, `chmod +x environment.sh`, `./environment.sh` to set the correct environment.

- Modify `style_transfer.py` and `process.py`. 

- TODO : Complete style description.

## 24.05.12 18:00 XQC

- Various fix. Plz `pip install -r requirements.txt` to update the environment.

- Modify style transfer.

## 24.05.12 17:38 SQA

- When trying styles, meet with a problem with Qwen, saying that 

```
AssertionError: Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。请使用model.chat_stream(...)代替model.chat(..., stream=True)。
```

Had better give it to XQC to fix.

- Modify the prompt to contain more information, meaning that the prompt is longer..

## 24.05.09 12:00 LYY

- Add `generateAPI.py` and `generateUI.py`, you can try them: `python generateAPI.py`  to start server, `python generateUI.py` to start UI. (The image generated is only according to the prompt, the description returned is your original prompt. I will change them after finishing project.)

- Test them as much as you can, thanks.

## 24.05.08 19:10 LYY

- Make a tiny change in `llama3.py`, but it still doesn't work (I change it earlier, but forget to push).

- fix SQA's update log.

## 24.05.06 9:26 SQA

- Fix a very small bug in `main.ipynb`. And I have thrown lots of style images in `/ssdshare/style/`.

## 24.05.06 20:30 LYY

- Try to add "llama3-70B" model to `process.py`. However, fail due to some strange errors. Don't call this model temporarily.

- Strange error occurs in `generate.py`, could someone fix it ?

## 24.05.06 16:09 SQA

- Implement style_transfer API. And also modify `style_transfer.py` to make it more runnable.

To specify, I keep the original way to run `style_transfer.py` with argument `-c`, `-s`, but they will be no longer required.

Instead, you can write a file `style_list.txt` in `data`, containing style image names from style library `data/style/` (of course, you must first upload your style image into the library). Style images will be read from this file by default. And content images will be read from `input_list.txt` by default. (Now I only implement `0.png` generated, but later...)

The code will automatically transfer any content to any style. (You still need to add `c_p` if wanted). 

In `main.py` everything is done except it only transfers `0.png` in `.tmp/generate/`. The output will be saved in `.tmp/style_transfer/`.

## 24.05.05 20:27 SQA

- Add prompt for no-character image generation. It seems that the model is rather good at this aspect.

- Also modify some code to be run more conveniently.

- For no-character, prompt will be `<audio>.prompt2`, image will be `<audio>.10-12`.

## 24.05.05 19:30 LYY

- Try Meta-Llama-3-70B-Instruct-AWQ model as process, it needs 4 GPU to run and the output is really slow (4 minutes per question). The reason is the model is too large s.t. 4 GPU is almost full. The output seems to be good, but I haven't run enough tests. I think someone may need to ask TA for at least 6 GPU to test it.

- Add Meta-Llama-3-70B-Instruct-AWQ model to `process.py`.

- Our prompt really needs to be modified !!!

## 24.05.05 15:00 XQC

- Use https://arxiv.org/pdf/1606.05897 to preserve the color of the content image. 

- Modify prompts.

## 24.05.04 20:48 SQA

- Modify some paths to make `extract.py` runnable independently.

## 24.05.04 17:45 XQC

- Find and modify a style transfer, based on Gatys' paper. It's a little bit slow, but the result is good.  This is a standalone implementation, and the interface has not been written yet.

## 24.05.04 16:00 XQC

- Modify several files. Now work properly.

## 24.05.04 14:48 SQA

- Improve prompts in process. Maybe we should try another way.

## 24.05.04 9:23 SQA

- Add input propmt.

To run the code, first create a file `data/input_prompt.txt` and write your prompt here. It can be empty.

## 24.05.03 21:00 LYY

- Create `generate.py`. Now we can generate images using `main.ipynb` from the original music! A memorable milestone!

- We need to find a way to get more information during process step. Because only several words prompt seems to be not enough to generate a good image.

## 24.05.03 17:00 XQC

- Modify `process.py`. File output now.

- Create `main.ipynb` to organize the process.

## 24.05.03 14:31 SQA

- Prompting engineering in `precess.py`. And tiny change in `test_playground.ipynb` in output directory.

## 24.05.03 14:30 LYY

- Change `tryInstantStyle.ipynb`. Don't use stable-diffusion model anymore! The image generated really damages the eyes. However, it works well with the `playground-v2.5-1024px-aesthetic` model.

- Try some examples to generate images using `playground-v2.5-1024px-aesthetic` with long prompts, `lpw_stable_diffusion_xl.py` really works!

- Now can use `stable-diffusion-xl-base-1.0` to generate images with `use_safetensors=True` and `variant="fp16"`. (But don't use it makes better images.)

## 24.05.03 12:30 XQC
- Fix SQA's `tryInstantStyle.ipynb`. We'd better not use this, huh.

## 24.05.03 10:05 SQA
- Throw `tryInstantStyle.ipynb`, which attempts to use InstantStyle but fails. The repo of InstantStyle has been cloned into `/ssdshare/LLMs/InstantStyle`. The original repo url is `https://github.com/InstantStyle/InstantStyle.git`. Note that you should read the README in the repo before you try to run the codes.

- The codes are all from README, but a problem is that it must connect huggingface to access the base model
`stabilityai/stable-diffusion-xl-base-1.0`. I tried to change this into the base model in ssdshare by lyy but failed, raising error saying that something was missing in the base model path. Then I tried to change the `StableDiffusionXLPipeline.from_pretrained` into `DiffusionPipeline.from_pretrained`, which was used in lyy's codes, but more confusing errors occured and I gave up..

- You can also find a demo notebook in the repo. But there is an error `cannot import name 'StableDiffusionXLControlNetPipeline' from 'diffusers'.` in some cell. (It seems that if you have the repo, you don't need to run all the `install dependency` at the beginning of this notebook.)

- Why can't I connect to hugging face???

## 24.05.03 0:45 XQC
- Fix the pipeline `lpw_stable_diffusion_xl.py`. Now load the custom pipeline from local file. Tested it works by asking "background be black" at the very very end of the prompt. This bug is caused by not copying codes completely from the original pipeline. Ah, what can I say.

## 24.05.02 15:00 XQC
- Half of process part.

- Modify `extract.py`, create `data` directory.

## 24.05.02 14:30 LYY

- Try Stable-Diffusion-xl-based-1.0

- Try Stable-Diffusion-xl-based-1.0 + stable-diffusion-xl-refiner-1.0. With the refiner model, the output is more realistic. However, the refiner model can't able to handle person's hands and other characteristics.

- Try playground-v2.5-1024px-aesthetic, the output is really fantastic. You need to download latest version of diffuser to run. Limits: only 77 tokens for input.

## 24.05.02 11:00 XQC
- Modify `extract.py`, now it accepts multi inputs. But still don't know why sometimes it throws trash.

## 24.04.25 19:30 XQC
- Try Qwen-chat. Ali's Qwen-chat based on Qwen-audio performs well (which doesn't use MERT, better than that of tecent).

- It can conclude the music and extract lyrics, which satisfies all the requirements. The point is that it can only deal with a window of 30s (?not sure), maybe music should be segmented.

## 24.04.24 16:00 XQC
- Try the MERT-v1-330M model. But modelscope deceived me that it outputs text, but it actually outputs vectors. It's hard to deal with these vectors.

- Now there're two ways to go. One is to train or find models dealing with these tensors. The other is to find models music -> text. I think the latter is more feasible.