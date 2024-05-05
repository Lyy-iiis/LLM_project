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