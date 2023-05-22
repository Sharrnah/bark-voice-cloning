from .generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio
import torchaudio
import torch


def clone_voice(audio_filepath, text, dest_filename, device="cpu"):
    if len(text) < 1:
        print('No transcription text entered!')
        return False

    if device == "cpu":
        use_gpu = False
    else:
        use_gpu = True
    print("Loading Codec")
    model = load_codec_model(use_gpu=use_gpu)
    print("Converting WAV")

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)
    print("Extracting codes")

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # get seconds of audio
    seconds = wav.shape[-1] / model.sample_rate
    # generate semantic tokens
    semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7)

    # move codes to cpu
    codes = codes.cpu().numpy()

    import numpy as np
    output_path = dest_filename + '.npz'
    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
    return True
