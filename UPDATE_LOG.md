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