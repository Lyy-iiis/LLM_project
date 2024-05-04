# LLM_Project

## Main structure

```
LLM_project
│   README.md
│   docs
│   │   # literature review, project proposal, project report, etc.
|   codes
|   |   extract
|   |   |   # extracting info from music
|   |   process
|   |   |   # processing info
|   |   generate
|   |   |   # generating image
```

## Detailed tasks
### Extract
Two parts, from lyrics and from music. Qwen does both jobs well.

Input : music

Output : prompt

### Process
Throwing everything to GPT/GLM.

Input : prompt

Output :  prompt

### Generate
Generate image using the prompt.

Input : prompt/dictionary

Output : image

### Style Transfer
Transfer the style of the image. Make it more like an illustration or a cover.

# To run the codes
### Extract
To run `extract.py` with default parameters, you should create a file in `./codes/data/input_list.txt`, in which you should declare the audio you want to deal with. 

- If `.mp3` audio is detected in the file, we will transfer it into `.wav` automatically.

The output prompt will be in `./codes/data/.tmp/extract/*.prompt`.

Of course you can run `extract.py` with your parameters, but do it yourself....