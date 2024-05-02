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
Two parts, from lyrics and from music, each with a LM.

Input : music

Output : prompt/dictionary

### Process
Throwing everything to GPT/GLM.

Input : prompt/dictionary

Output :  prompt/dictionary

### Generate
Generate image using the prompt/dictionary.

Input : prompt/dictionary

Output : image

# To run the codes
### Extract
To run `extract.py` with default parameters, you should create a file in `./codes/data/input_list.txt`, in which you should declare the audio you want to deal with. 

- If `.mp3` audio is detected in the file, we will transfer it into `.wav` automatically.

The output prompt will be in `./codes/data/.tmp/extract/*.prompt`.

Of course you can run `extract.py` with your parameters, but do it yourself....