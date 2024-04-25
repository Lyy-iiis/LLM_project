## 23.04.25 19:30 XQC
- Try Qwen-chat. Ali's Qwen-chat based on Qwen-audio performs well (which doesn't use MERT, better than that of tecent).

- It can conclude the music and extract lyrics, which satisfies all the requirements. The point is that it can only deal with a window of 30s (?not sure), maybe music should be segmented.

## 23.04.24 16:00 XQC
- Try the MERT-v1-330M model. But modelscope deceived me that it outputs text, but it actually outputs vectors. It's hard to deal with these vectors.

- Now there're two ways to go. One is to train or find models dealing with these tensors. The other is to find models music -> text. I think the latter is more feasible.