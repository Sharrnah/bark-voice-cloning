########################################################################
# Bark Voice Cloning
# sources:
# https://huggingface.co/GitMylo/bark-voice-cloning
# https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
# https://github.com/gitmylo/audio-webui
########################################################################

import hashlib
import os
import threading
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

from bark import hubert

import click
import torch
import torchaudio
import numpy
from encodec import EncodecModel


# force add all hydra plugins (for pyinstaller)
#from hydra.plugins import *

def sha256_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file_simple(url, target_path, expected_sha256=None):
    progress_lock = threading.Lock()
    file_name = os.path.basename(urlparse(url).path)

    def show_progress(count, total_size):
        with progress_lock:
            percentage = int(count * 100 / total_size)
            print(f'\rDownloading {file_name}: {percentage}%', end='')

    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, file_name)

    with closing(urllib.request.urlopen(url)) as remote_file:
        headers = remote_file.info()
        total_size = int(headers.get('Content-Length', -1))

        with open(target_path, 'wb') as local_file:
            block_size = 8192
            downloaded_size = 0
            for block in iter(lambda: remote_file.read(block_size), b''):
                local_file.write(block)
                downloaded_size += len(block)
                show_progress(downloaded_size, total_size)
            print()

    if expected_sha256:
        actual_sha256 = sha256_checksum(target_path)
        if actual_sha256.lower() != expected_sha256.lower():
            os.remove(target_path)
            raise ValueError(
                f"Downloaded file has incorrect SHA256 hash. Expected {expected_sha256}, but got {actual_sha256}.")
        else:
            print("SHA256 hash verified.")


MODELS = {
    "hubert_base": {
        "url": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
        "sha256": "1703cf8d2cdc76f8c046f5f6a9bcd224e0e6caf4744cad1a1f4199c32cac8c8d",
    },
    "quantifier_hubert_base": {
        "url": "https://huggingface.co/GitMylo/bark-voice-cloning/resolve/main/quantifier_hubert_base_ls960_14.pth",
        "sha256": "9cf7eeab58835c5fc1cfbd3fd19c457fbd07859a5f036a6bfea4b6840716c1e7",
        "name": "quantifier_hubert_base_ls960_14.pth",
    },
    "quantifier_hubert_base_large": {
        "url": "https://huggingface.co/GitMylo/bark-voice-cloning/resolve/main/quantifier_V1_hubert_base_ls960_23.pth",
        "sha256": "0d94c5dd646bcfe1a8bb470372f0004c189acf65d913831f3a6ed6414c9ba86f",
        "name": "quantifier_V1_hubert_base_ls960_23.pth",
    },
}


@click.command()
@click.option('--audio_file', help='audio file to clone voice from', type=str)
@click.option('--npz_file', default=None, help='generated NPZ file of the cloned voice (Bark compatible)', type=str)
@click.option("--small_model", is_flag=True, help="use small model", type=bool)
@click.option("--offload_cpu", is_flag=True, help="offload to CPU", type=bool)
@click.option("--use_gpu", is_flag=True, help="use GPU", type=bool)
@click.pass_context
def main(ctx, audio_file, npz_file, small_model, offload_cpu, use_gpu):
    if audio_file is None:
        print("Please specify an audio file using '--audio_file' to clone voice from.")
        return

    # check if file exists
    if not os.path.isfile(audio_file):
        print(f"Audio file {audio_file} does not exist.")
        return

    if not small_model:
        print("Using large model")

    print("Loading models...")

    if not Path("hubert_base_ls960.pt").is_file():
        download_file_simple(MODELS["hubert_base"]["url"], "hubert_base_ls960.pt", MODELS["hubert_base"]["sha256"])
    hubert_model = hubert.CustomHubert(checkpoint_path='hubert_base_ls960.pt')

    # load tokenizer
    #hubert_tokenizer = hubert.CustomTokenizer()
    quantifier_model_type = "quantifier_hubert_base"
    if not small_model:
        quantifier_model_type = "quantifier_hubert_base_large"
    if not Path(MODELS[quantifier_model_type]["name"]).is_file():
        download_file_simple(MODELS[quantifier_model_type]["url"], MODELS[quantifier_model_type]["name"],
                             MODELS[quantifier_model_type]["sha256"])

    #hubert_tokenizer.load_state_dict(torch.load(MODELS[quantifier_model_type]["name"]))
    hubert_tokenizer = hubert.CustomTokenizer.load_from_checkpoint(MODELS[quantifier_model_type]["name"])

    if npz_file is None:
        clone_history_prompt_save = audio_file + ".npz"
    else:
        clone_history_prompt_save = npz_file

    print("Cloning Voice...")

    # Run the model to extract semantic features from an audio file, where wav is your audio file
    wav, sr = torchaudio.load(audio_file)

    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=sr)

    # Process the semantic vectors from the previous HuBERT run (This works in batches, so you can send the entire HuBERT output)
    semantic_tokens = hubert_tokenizer.get_token(semantic_vectors)

    # Extract discrete codes from EnCodec
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    wav, sr = torchaudio.load(audio_file)
    wav = hubert.hubert_convert_audio(wav, sr, encodec_model)
    wav = wav.unsqueeze(0)
    if not (offload_cpu or not use_gpu):
        wav = wav.to('cuda')
    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
    fine_prompt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
    fine_prompt = fine_prompt.cpu().numpy()

    # extract coarse prompt
    coarse_prompt = fine_prompt[:2, :]

    # write semantic tokens to file
    numpy.savez(clone_history_prompt_save, semantic_prompt=semantic_tokens, fine_prompt=fine_prompt,
                coarse_prompt=coarse_prompt)

    print("Cloning Voice for Bark finished in file: " + clone_history_prompt_save)


main()
