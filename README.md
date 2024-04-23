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
|   |   |   # generating music
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
