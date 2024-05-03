# using Qwen

########################################################

from pydub import AudioSegment
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
torch.manual_seed(42)
from pydub import AudioSegment
import argparse 
import warnings

#######################################################

parser = argparse.ArgumentParser(description = 'Extractor')

parser.add_argument('--input_file_name', type = str, default = 'input_list.txt')
parser.add_argument('--model_path', type = str, default = '/ssdshare/LLMs/')
parser.add_argument('--data_path', type = str, default = './data/')
parser.add_argument('--music_path', type=str, default='./data/music/')
parser.add_argument('--output_path', type = str, default = './data/.tmp/extract/')
parser.add_argument('--window_size', type = int, default = 30_000)
parser.add_argument('--overlap_size', type = int, default = 5_000)
parser.add_argument('--device_num', type = int, default = 1)

args = parser.parse_args()

#######################################################

MODEL_PATH = args.model_path
PWD = os.getcwd()
DATA_PATH = args.data_path
if DATA_PATH[0] == '.' :
    DATA_PATH = PWD + "/" + DATA_PATH
MUSIC_PATH = args.music_path
DEVICE = "cuda" if torch.cuda.is_available() else "xuwei"
assert DEVICE == "cuda", "WHY DONT YOU HAVE CUDA???????"
TEMPORARY_PATH = DATA_PATH + ".tmp/"
if not os.path.exists(TEMPORARY_PATH) :
    os.makedirs(TEMPORARY_PATH)
CUDA_NUM = args.device_num
assert CUDA_NUM > 0, "DO YOU WANT ME TO DONATE MY GPU TO YOU????"
assert CUDA_NUM <= torch.cuda.device_count(), "YOU ARE ASKING FOR TOO MANY GPUS"
CUDA_DEVICE = [f"cuda:{i}" for i in range(CUDA_NUM)]
WINDOW_SIZE = args.window_size
OVERLAP_SIZE = args.overlap_size
OUTPUT_PATH = args.output_path
input_file_name = args.input_file_name

#######################################################

audio_file_name = []
with open(DATA_PATH + input_file_name, "r") as f :
    for line in f :
        audio_file_name.append(line.rstrip())
        print(audio_file_name[-1])

# mp3 cast 2 wav
from pydub import AudioSegment
for (file_name, i) in zip(audio_file_name, range(len(audio_file_name))) :
    if file_name[-4:] == ".mp3" :
        audio = AudioSegment.from_mp3(MUSIC_PATH + file_name)
        audio = audio.set_channels(1)
        audio.export(MUSIC_PATH + file_name[:-4] + ".wav", format = "wav")
        audio_file_name[i] = file_name[:-4] + ".wav"
print(audio_file_name)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + "Qwen-Audio-Chat/", trust_remote_code = True)

models = []
for device in CUDA_DEVICE :
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH + "Qwen-Audio-Chat/", device_map = device, trust_remote_code = True).eval()
    models.append(model)


def meaningful_lyrics(lyrics):
    if "NOLYRICS" in lyrics:
        return False
    if len(lyrics) < 100:
        return False

    words = lyrics.split()
    total_words = len(words)
    unique_words = len(set(words))
    if unique_words / total_words < 0.17:
        return False
    max_len = 0
    for word in words:
        max_len = max(max_len, len(word))
    if max_len > 15 and lyrics.isascii() :
        return False
    return True


def extract(file_name, device = 0, path = MUSIC_PATH) :
    query = tokenizer.from_list_format([
        {'audio': path + file_name + '.wav'}, 
        {'text': 'Please give a detailed description (emotion, background) of this piece of music, with no less than 5 sentences. You should give 5 sentences, not the lyrics, not words. '},
    ])
    decription, _ = models[device].chat(tokenizer, query = query, history = None)

    query = tokenizer.from_list_format([
        {'audio': path + file_name + '.wav'},
        {'text': 'Extract all the lyrics of this music if it has. Say "NOLYRICS" if it does not have lyrics or the lyrics are meaningless.'},
    ])
    
    lyrics, _ = models[device].chat(tokenizer, query = query, history = None)
    lyrics = lyrics.split('"')[1]
    if not meaningful_lyrics(lyrics) :
        lyrics = None
    return decription, lyrics


def partition_extract(file_name, device_start = 0, no_clear = False) :
    #####################################
    # I don't recommend parallel because running the same model continiously causes problems.
    #####################################
    file_name = file_name + ''
    audio = AudioSegment.from_wav(MUSIC_PATH + file_name + ".wav")
    duration = len(audio)
    num_of_pieces = (duration - OVERLAP_SIZE) // (WINDOW_SIZE - OVERLAP_SIZE) + 1
    file_name = file_name.replace(" ", "_")
    if not os.path.exists(TEMPORARY_PATH) :
        os.makedirs(TEMPORARY_PATH)
    if not os.path.exists(TEMPORARY_PATH + file_name) :
        os.makedirs(TEMPORARY_PATH + file_name)

    description = []
    lyrics = []

    for i in range(num_of_pieces) :
        start = i * (WINDOW_SIZE - OVERLAP_SIZE)
        end = start + WINDOW_SIZE
        if end > duration :
            end = duration
        piece = audio[start:end]
        piece.export(TEMPORARY_PATH + file_name + f"/{i}.wav", format = "wav")
        print(f"using device {(i + device_start) % CUDA_NUM}")
        description_piece, lyrics_piece = extract(f"/{i}", device = (i + device_start) % CUDA_NUM, path = TEMPORARY_PATH + file_name + "/")
        description.append(description_piece)
        lyrics.append(lyrics_piece)

    if not no_clear :
        os.system(f"rm -rf {TEMPORARY_PATH + file_name + '/'}")

    return description, lyrics, (num_of_pieces + device_start) % CUDA_NUM


def make_prompt(file_name, device_start = 0) :
    description, lyrics, device_start = partition_extract(file_name, device_start = device_start)
    prompt = f"This music is cut into {len(description)} pieces. Each piece has a length of {WINDOW_SIZE // 1000} seconds and an overlap of {OVERLAP_SIZE // 1000} seconds. The description of each piece is as follows:\n"
    for i, d in enumerate(description) :
        prompt += f"Description piece {i + 1}: {d}\n"

    have_lyrics = False
    for l in lyrics :
        if l is not None :
            have_lyrics = True
            break

    if have_lyrics :
        prompt += f"\nThe lyrics are as follows:\n"
        for i, l in enumerate(lyrics) :
            if l is not None :
                prompt += f"{l}\n"
    return prompt, device_start

prompts = []
device_start = 0
for file_name in audio_file_name :
    tmp, device_start = make_prompt(file_name[:-4], device_start = device_start)
    prompts.append(tmp)
    print("successfully add prompt for " + file_name)

if not os.path.exists(OUTPUT_PATH) :
    os.makedirs(OUTPUT_PATH)

for file_name, prompt in zip(audio_file_name, prompts) :
    with open(OUTPUT_PATH + file_name[:-4] + ".prompt", "w") as f :
        f.write(prompt)
    print("successfully write prompt for " + file_name)