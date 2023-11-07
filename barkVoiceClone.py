########################################################################
# Bark Voice Cloning
# sources:
# https://huggingface.co/GitMylo/bark-voice-cloning
# https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
# https://github.com/gitmylo/audio-webui
########################################################################

import hashlib
import io
import os
import random
import threading
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

import numpy as np

import audioprocessing
from bark import hubert, bark
from scipy.io.wavfile import write as write_wav

import secrets
import string

import click
import torch
import torchaudio
import numpy
from encodec import EncodecModel

SAMPLE_RATE = 24000


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

bark_plugin_dir = Path(Path.cwd() / "Plugins" / "bark_plugin")
os.makedirs(bark_plugin_dir, exist_ok=True)


def set_seed(seed: int = 0):
    """Set the seed
    seed = 0         Generate a random seed
    seed = -1        Disable deterministic algorithms
    0 < seed < 2**32 Set the seed
    Args:
        seed: integer to use as seed
    Returns:
        integer used as seed
    """

    original_seed = seed

    # See for more informations: https://pytorch.org/docs/stable/notes/randomness.html
    if seed == -1:
        # Disable deterministic

        print("Disabling deterministic algorithms")

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]

        torch.use_deterministic_algorithms(False)  # not sure if needed, yes it is

    else:

        print("Enabling deterministic algorithms")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # not sure if this is needed, yes it is,
        torch.use_deterministic_algorithms(True)  # not sure if needed, yes it is

    if seed <= 0:
        # Generate random seed
        # Use default_rng() because it is independent of np.random.seed()
        seed = np.random.default_rng().integers(1, 2 ** 32 - 1)

    assert (0 < seed and seed < 2 ** 32)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Set seed to {seed}")

    return original_seed if original_seed != 0 else seed


def clone_voice(audio_file, npz_file=None, small_model=False, offload_cpu=True, use_gpu=False):
    if not small_model:
        print("Using large model")

    print("Loading models...")
    hubert_model_target_path = str(Path(bark_plugin_dir / "hubert_base_ls960.pt").resolve())
    if not Path(hubert_model_target_path).is_file():
        download_file_simple(MODELS["hubert_base"]["url"], hubert_model_target_path, MODELS["hubert_base"]["sha256"])
    hubert_model = hubert.CustomHubert(checkpoint_path=hubert_model_target_path)

    # load tokenizer
    quantifier_model_type = "quantifier_hubert_base"
    if not small_model:
        quantifier_model_type = "quantifier_hubert_base_large"

    quantifier_target_path = str(Path(bark_plugin_dir / MODELS[quantifier_model_type]["name"]).resolve())

    if not Path(quantifier_target_path).is_file():
        download_file_simple(MODELS[quantifier_model_type]["url"], quantifier_target_path,
                             MODELS[quantifier_model_type]["sha256"])

    hubert_tokenizer = hubert.CustomTokenizer.load_from_checkpoint(quantifier_target_path)

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


def chunk_up_text(text_prompt, split_character_goal_length, split_character_max_length,
                  split_character_jitter=0):
    if split_character_jitter > 0:
        split_character_goal_length = random.randint(split_character_goal_length - split_character_jitter,
                                                     split_character_goal_length + split_character_jitter)
        split_character_max_length = random.randint(split_character_max_length - split_character_jitter,
                                                    split_character_max_length + split_character_jitter)

    audio_segments = bark.split_general_purpose(text_prompt,
                                                split_character_goal_length=split_character_goal_length,
                                                split_character_max_length=split_character_max_length)

    print(f"Splitting long text aiming for {split_character_goal_length} chars max {split_character_max_length}")

    return audio_segments


def audio_processing(audio_data, skip_infinity_lufs=True, **kwargs):
    normalize = kwargs.get("normalize", True)
    sample_rate = kwargs.get("sample_rate", 24000)

    trim_silence = kwargs.get("trim_silence", True)
    remove_silence_parts = kwargs.get("remove_silence_parts", False)

    # Normalize audio
    if normalize:
        lower_threshold = kwargs.get("normalize_lower_threshold", -24.0)
        upper_threshold = kwargs.get("normalize_upper_threshold", -16.0)
        gain_factor = kwargs.get("normalize_gain_factor", 1.3)
        audio_data, lufs = audioprocessing.normalize_audio_lufs(audio_data, sample_rate, lower_threshold,
                                                                upper_threshold,
                                                                gain_factor)
        if lufs == float('-inf') and skip_infinity_lufs:
            print("Audio seems to be unusable. skipping")
            return None

    # Trim silence
    if trim_silence:
        audio_data = audioprocessing.trim_silence(audio_data)

    # Remove silence parts
    if remove_silence_parts:
        audio_data = audioprocessing.remove_silence_parts(audio_data, sample_rate)

    # return early if no audio data
    if len(audio_data) == 0:
        return None

    return audio_data


def generate_speech(text, text_temp=0.7, waveform_temp=0.7, min_eos_p=0.05, write_last_history_prompt=False,
                    last_hisory_prompt_file=None, prompt_wrap="##", skip_infinity_lufs=True, long_text=False,
                    long_text_stable_frequency=0, long_text_split_pause=0.0, **kwargs):
    use_gpu = kwargs.get("use_gpu", False)

    print("download and load all bark models")
    bark.preload_models(
        text_use_gpu=use_gpu,
        coarse_use_gpu=use_gpu,
        fine_use_gpu=use_gpu,
        codec_use_gpu=use_gpu,
    )
    print("bark models loaded")

    history_prompt = kwargs.get("history_prompt", "")
    split_character_goal_length = kwargs.get("split_character_goal_length", 110)
    split_character_max_length = kwargs.get("split_character_max_length", 170)
    split_character_jitter = kwargs.get("split_character_jitter", 0)

    if history_prompt == "":
        history_prompt = None

    # seed settings
    seed = kwargs.get("seed", -2)
    if seed > -2:
        try:
            _ = set_seed(seed)
        except Exception as e:
            print("error setting seed: " + str(e))

    if long_text:
        audio_arr_segments = []

        estimated_time = bark.estimate_spoken_time(text)
        print(f"estimated_time: {estimated_time}")

        audio_segments = chunk_up_text(text,
                                       split_character_goal_length=split_character_goal_length,
                                       split_character_max_length=split_character_max_length,
                                       split_character_jitter=split_character_jitter
                                       )
        print(f"audio_segments: {len(audio_segments)}")

        history_prompt_for_next_segment = history_prompt

        for i, segment_text in enumerate(audio_segments):
            estimated_time = bark.estimate_spoken_time(segment_text)

            print(f"segment: {i}")
            print(f"estimated_time: {estimated_time}")

            segment_text = prompt_wrap.replace("##", segment_text)

            history_prompt_data, audio_data_np_array = bark.generate_audio(segment_text,
                                                                           history_prompt=history_prompt_for_next_segment,
                                                                           text_temp=text_temp,
                                                                           waveform_temp=waveform_temp,
                                                                           min_eos_p=min_eos_p,
                                                                           output_full=True
                                                                           )
            audio_data_np_array = audio_processing(audio_data_np_array, skip_infinity_lufs=skip_infinity_lufs)

            audio_arr_segments.append(audio_data_np_array)

            if history_prompt is None and history_prompt_data is not None:
                history_prompt = history_prompt_data

            if long_text_stable_frequency > 0 and (i + 1) % long_text_stable_frequency == 0:
                history_prompt_for_next_segment = history_prompt
            else:
                history_prompt_for_next_segment = history_prompt_data

        # put all audio together
        if len(audio_arr_segments) > 0 and long_text_split_pause > 0.0:
            audio_with_pauses = []
            pause_samples = np.zeros(int(long_text_split_pause * SAMPLE_RATE))
            # Iterate over each audio segment
            for segment in audio_arr_segments:
                # Add the audio segment
                audio_with_pauses.append(segment)
                # Add a pause
                audio_with_pauses.append(pause_samples)
            # Remove the last added pause as it's not needed after the last segment
            audio_arr_segments = audio_with_pauses[:-1]

        # put all audio together
        audio_data_np_array = np.concatenate(audio_arr_segments)

    else:
        text = prompt_wrap.replace("##", text)
        if write_last_history_prompt:
            history_prompt_data, audio_data_np_array = bark.generate_audio(text,
                                                                           history_prompt=history_prompt,
                                                                           text_temp=text_temp,
                                                                           waveform_temp=waveform_temp,
                                                                           min_eos_p=min_eos_p,
                                                                           output_full=write_last_history_prompt)
            bark.save_as_prompt(last_hisory_prompt_file, history_prompt_data)
        else:
            audio_data_np_array = bark.generate_audio(text, history_prompt=history_prompt,
                                                      text_temp=text_temp, waveform_temp=waveform_temp, min_eos_p=min_eos_p)

        audio_data_np_array = audio_processing(audio_data_np_array, skip_infinity_lufs=skip_infinity_lufs)

    sample_rate = SAMPLE_RATE

    #voicefixer_enabled = kwargs.get("voicefixer", False)
    #voicefixer_mode = kwargs.get("voicefixer_mode", 0)
    ## optimize voice
    #if voicefixer_enabled:
    #    audio_data_np_array = audioprocessing.voicefixer_preprocess(audio_data_np_array, mode=voicefixer_mode,
    #                                                                cuda=use_gpu)
    #    # sample back to self.sample_rate (should later be using the generated sample rate for better quality)
    #    audio_data_np_array = audioprocessing.resample_audio(audio_data_np_array, 44100, 44100, target_channels=-1,
    #                                                         is_mono=True, dtype="float32")
    #    sample_rate = 44100

    audio_data_16bit = np.int16(audio_data_np_array * 32767)  # Convert to 16-bit PCM

    buff = io.BytesIO()
    write_wav(buff, sample_rate, audio_data_16bit)

    buff.seek(0)

    return buff.getvalue(), sample_rate


def batch_generate(batch_prompts, batch_size=1, batch_folder="bark_prompts/multi_generations", text_temp=0.7,
                   waveform_temp=0.7, min_eos_p=0.05, prompt_wrap="##", history_prompt="", ):
    # generate multiple voices in a batch
    write_last_history_prompt = True
    os.makedirs(batch_folder, exist_ok=True)

    text_list = batch_prompts.split("\n")
    # remove empty lines
    text_list = [x for x in text_list if x.strip() != ""]

    if batch_size > 0:
        print("Batch Generating " + str(
            batch_size * len(text_list)) + " audios...\n(" + str(
            batch_size) + " per prompt)\nstarted.\n\nlook for them in '" + batch_folder + "' directory.")
        promt_num = 0
        for text_line in text_list:
            if text_line.strip() != "":
                prmpt_dir = batch_folder + "/prompt-" + str(promt_num)
                os.makedirs(prmpt_dir, exist_ok=True)
                # write prompt text to file
                with open(prmpt_dir + "/prompt.txt", "w", encoding='utf-8') as f:
                    f.write(text_line.strip())
                for i in range(batch_size):
                    file_name = prmpt_dir + "/" + str(i)
                    # generate wav and history prompt
                    wav, sample_rate = generate_speech(text_line.strip(),
                                                       text_temp=text_temp,
                                                       waveform_temp=waveform_temp,
                                                       min_eos_p=min_eos_p,
                                                       write_last_history_prompt=write_last_history_prompt,
                                                       last_hisory_prompt_file=file_name + ".npz",
                                                       prompt_wrap=prompt_wrap,
                                                       skip_infinity_lufs=False,
                                                       history_prompt=history_prompt, )
                    # write wav to file
                    if wav is not None:
                        # write wav to file
                        wav_file_name = file_name + ".wav"
                        with open(wav_file_name, "wb") as f:
                            f.write(wav)
                promt_num += 1
        print("Batch Generating finished.\n\nlook for them in '" + batch_folder + "' directory.")
    else:
        error_msg = "Invalid batch size. must be number of runs per line"
        print(error_msg)


def generate_random_filename(extension='.wav', length=10, directory='./'):
    ALPHABET = string.ascii_letters + string.digits

    while True:
        filename = ''.join(secrets.choice(ALPHABET) for i in range(length)) + extension
        file_path = os.path.join(directory, filename)

        # check if the file already exists
        if not os.path.isfile(file_path):
            return filename

@click.command()
# clone voice options
@click.option("--clone", is_flag=True, help="Clone Voice", type=bool)
@click.option('--audio_file', help='audio file to clone voice from.', type=str)
@click.option('--npz_file', default=None, help='generated NPZ file of the cloned voice (Bark compatible)', type=str)
# generate speech options
@click.option("--generate", is_flag=True, help="Generate speech", type=bool)
@click.option('--text', default=None, help='text to generate speech from.', type=str)
@click.option('--history_prompt', default=None, help='history prompt (voice) to generate speech with.', type=str)
@click.option('--text_temp', default=0.7, help='text temperature', type=float)
@click.option('--waveform_temp', default=0.7, help='waveform temperature', type=float)
@click.option('--min_eos_p', default=0.05, help='min. end of sentence probability', type=float)
@click.option('--last_history_prompt_file', default=None,
              help='last history prompt file, the last speech was generated with', type=str)
@click.option('--prompt_wrap', default="##",
              help='wrap prompt with this string (the ## will be replaced with the prompt)', type=str)
@click.option('--skip_infinity_lufs', default=True, help='skip audio with infinity lufs (very likely unusable audio)',
              type=bool)
@click.option('--long_text', default=True,
              help='long text mode (splits the text and generates each part and combines final audios', type=bool)
@click.option('--long_text_stable_frequency', default=1, help='stable frequency for long text mode', type=int)
@click.option('--long_text_split_pause', default=0.0, help='pause between long text splits', type=float)
@click.option('--split_character_goal_length', default=110, help='goal length of each split character', type=int)
@click.option('--split_character_max_length', default=150, help='max length of each split character', type=int)
@click.option('--split_character_jitter', default=0, help='jitter of each split character', type=int)
#@click.option('--voicefixer', default=False, help='use voicefixer', type=bool)
#@click.option('--voicefixer_mode', default=0, help='voicefixer mode', type=int)
@click.option('--seed', default=-2,
              help='set seed to -2 to use default, -1 to use random, or a positive integer to use a specific seed.',
              type=int)
# general options
@click.option("--small_model", is_flag=True, help="use small model", type=bool)
@click.option("--offload_cpu", is_flag=True, help="offload to CPU", type=bool)
@click.option("--use_gpu", is_flag=True, help="use GPU", type=bool)
@click.pass_context
def main(ctx, clone, audio_file, npz_file,
         # generate speech options
         generate, text, history_prompt, text_temp, waveform_temp, min_eos_p,
         last_history_prompt_file, prompt_wrap, skip_infinity_lufs,
         long_text, long_text_stable_frequency, long_text_split_pause, split_character_goal_length,
         split_character_max_length, split_character_jitter,
         #voicefixer, voicefixer_mode,
         seed,
         # general options
         small_model, offload_cpu, use_gpu,
         ):
    if clone and generate:
        print("Please specify either '--clone' or '--generate', not both.")
        return

    if clone:
        if audio_file is None:
            print("Please specify an audio file using '--audio_file' to clone voice from.")
            return

        # check for file existence
        if not os.path.isfile(audio_file):
            print(f"Audio file {audio_file} does not exist.")
            return

        clone_voice(audio_file, npz_file=npz_file, small_model=small_model, offload_cpu=offload_cpu, use_gpu=use_gpu)

    if generate:
        if text is None:
            print("Please specify text to generate speech from using '--text'.")
            return

        write_last_history_prompt = False
        if last_history_prompt_file is not None and last_history_prompt_file != "":
            write_last_history_prompt = True
        wav, sample_rate = generate_speech(text,
                                           use_gpu=use_gpu,
                                           history_prompt=history_prompt,
                                           text_temp=text_temp, waveform_temp=waveform_temp, min_eos_p=min_eos_p,
                                           write_last_history_prompt=write_last_history_prompt,
                                           last_hisory_prompt_file=last_history_prompt_file,
                                           prompt_wrap=prompt_wrap, skip_infinity_lufs=skip_infinity_lufs,
                                           long_text=long_text, long_text_stable_frequency=long_text_stable_frequency,
                                           long_text_split_pause=long_text_split_pause,
                                           split_character_goal_length=split_character_goal_length,
                                           split_character_max_length=split_character_max_length,
                                           split_character_jitter=split_character_jitter,
                                           #voicefixer=voicefixer, voicefixer_mode=voicefixer_mode,
                                           seed=seed,
                                           )
        # write wav to file
        if wav is not None:
            file_name = generate_random_filename()
            # write wav to file
            with open(file_name, "wb") as f:
                f.write(wav)


main()
