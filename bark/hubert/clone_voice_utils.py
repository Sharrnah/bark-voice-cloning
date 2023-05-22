from encodec.utils import convert_audio


def hubert_convert_audio(wav, sr, encodec_model):
    return convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
