# LLM_Project

## Main structure

```
LLM_project
│   docs
│   │   # literature review, project proposal, project report, etc.
|   codes
|   |   extract
|   |   |   # extracting info from music
|   |   process
|   |   |   # processing info
|   |   generate
|   |   |   # generating image
|   |   style_transfer
|   |   |   # transfering style
|   |   data
|   |   |   # data used in the project and output
|   |   main.ipynb
|   |   |   # ipynb for testing the pipeline
|   demo
|   |   docker
|   |   |   # docker files
|   |   generateAPI.py
|   |   generateUI.py
|   |   demo.py
|   |   |   # demo code
│   README.md
|   requirements.txt
|   environment.sh
```

## Getting Started

### Environment

```bash
bash environment.sh
```

### Demo

To run the demo

```bash
python demo/generateAPI.py
```

```bash
python demo/generateUI.py
```

in two terminals, then open the browser and visit http://localhost:7860/ to see the demo

### Docker

If you don't want to build a docker, you can change `IngressHost` and `NameSpace` in `project_lyy.yaml` into yours and use the docker we already built by running the following command.

```bash
cd demo/docker/MI-T_helm_template
make our
make connect
```




Before building the docker, please download pretrained model from https://download.pytorch.org/models/vgg19-dcbb9e9d.pth and put it in `demo/docker/MI-T_API_docker/`.

Build the docker:

```bash
docker login harbor.ai.iiis.co:9443
cd demo/docker/MI-T_API_docker
make docker USER=your_docker_username
make push USER=your_docker_username
```

```bash
cd demo/docker/MI-T_UI_docker
make docker USER=your_docker_username
make push USER=your_docker_username
```

You can remove local images after pushing.

```bash
make remove USER=your_docker_username
```

We already build the image for you. In `codes/demo/docker/MI-T_helm_template/project.yaml`, change your namespace to your own namespace, `ContainerImage` to your api image, `GradioImage` to your ui image, `IngressHost` to your own host.

To run the docker

```bash
cd demo/docker/MI-T_helm_template
make podup
make connect
```

Then you can open the browser and visit https://llmlab.ddns.net:9443 (Your ddns) to see the demo.

```bash
make poddown
```

remove the pod after you finish.

<!-- ### Extract
To run `extract.py` with default parameters, you should create a file in `./codes/data/input_list.txt`, in which you should declare the audio you want to deal with. 

- If `.mp3` audio is detected in the file, we will transfer it into `.wav` automatically.

The output prompt will be in `./codes/data/.tmp/extract/*.prompt`.

Of course you can run `extract.py` with your parameters, but do it yourself.... -->

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