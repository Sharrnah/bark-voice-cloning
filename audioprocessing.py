#import os

#import librosa
import numpy as np
import pyloudnorm
import resampy
#import voicefixer


# Function to calculate LUFS
def calculate_lufs(audio, sample_rate):
    meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    return loudness


# Function to normalize the audio based on LUFS
def normalize_audio_lufs(audio, sample_rate, lower_threshold=-24.0, upper_threshold=-16.0, gain_factor=2.0):
    lufs = calculate_lufs(audio, sample_rate)

    print(f"LUFS: {lufs}")

    # If LUFS is lower than the lower threshold, increase volume
    if lufs < lower_threshold:
        print(f"audio is too quiet, increasing volume")
        gain = (lower_threshold - lufs) / gain_factor
        audio = audio * np.power(10.0, gain / 20.0)

    # If LUFS is higher than the upper threshold, decrease volume
    elif lufs > upper_threshold:
        print(f"audio is too loud, decreasing volume")
        gain = (upper_threshold - lufs) * gain_factor
        audio = audio * np.power(10.0, gain / 20.0)

    # Limit audio values to [-1, 1] (this is important to avoid clipping when converting to 16-bit PCM)
    audio = np.clip(audio, -1, 1)

    return audio, lufs


def trim_silence(audio, silence_threshold=0.01):
    # Compute absolute value of audio waveform
    audio_abs = np.abs(audio)

    # Find the first index where the absolute value of the waveform exceeds the threshold
    start_index = np.argmax(audio_abs > silence_threshold)

    # Reverse the audio waveform and do the same thing to find the end index
    end_index = len(audio) - np.argmax(audio_abs[::-1] > silence_threshold)

    # If start_index is not 0, some audio at the start has been trimmed
    if start_index > 0:
        print(f"Trimmed {start_index} samples from the start of the audio")

    # If end_index is not the length of the audio, some audio at the end has been trimmed
    if end_index < len(audio):
        print(f"Trimmed {len(audio) - end_index} samples from the end of the audio")

    # Return the trimmed audio
    return audio[start_index:end_index]


def remove_silence_parts(audio, sample_rate, silence_threshold=0.01, max_silence_length=1.1,
                         keep_silence_length=0.06):
    audio_abs = np.abs(audio)
    above_threshold = audio_abs > silence_threshold

    # Convert length parameters to number of samples
    max_silence_samples = int(max_silence_length * sample_rate)
    keep_silence_samples = int(keep_silence_length * sample_rate)

    last_silence_end = 0
    silence_start = None

    chunks = []

    for i, sample in enumerate(above_threshold):
        if not sample:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                silence_duration = i - silence_start
                if silence_duration > max_silence_samples:
                    # Subtract keep_silence_samples from the start and add it to the end
                    start = max(last_silence_end - keep_silence_samples, 0)
                    end = min(silence_start + keep_silence_samples, len(audio))
                    chunks.append(audio[start:end])
                    last_silence_end = i
                silence_start = None

    # Append the final chunk of audio after the last silence
    if last_silence_end < len(audio):
        start = max(last_silence_end - keep_silence_samples, 0)
        end = len(audio)
        chunks.append(audio[start:end])

    if len(chunks) == 0:
        print("No non-silent sections found in audio.")
        return np.array([])
    else:
        print(f"found {len(chunks)} non-silent sections in audio")
        return np.concatenate(chunks)


#voicefixer_model = None


#def voicefixer_preprocess(audio, mode=0, cuda=True, sample_rate=44100):
#    global voicefixer_model
#
#    if voicefixer_model is None:
#        voicefixer_model = voicefixer.VoiceFixer()
#
#    if isinstance(audio, str) and os.path.isfile(audio):
#        sample_rate = 44100
#        audio, _ = librosa.load(audio, sr=sample_rate)
#
#    out_audio = voicefixer_model.restore_inmem_resample(audio, cuda=cuda, mode=mode, input_sample_rate=sample_rate)
#
#    # use vocoder
#    #if self.vocoder is None:
#    #    self.vocoder = self.voicefixer_module.Vocoder(sample_rate=44100)
#    #out_audio = self.vocoder.oracle(fpath=out_audio,
#    #                    cuda=cuda)
#
#    return out_audio


# resample_audio function to resample audio data to a different sample rate and convert it to mono.
# set target_channels to '-1' to average the left and right channels to create mono audio (default)
# set target_channels to '0' to extract the first channel (left channel) data
# set target_channels to '1' to extract the second channel (right channel) data
# set target_channels to '2' to keep stereo channels (or copy the mono channel to both channels if is_mono is True)
# to Convert the int16 numpy array to bytes use .tobytes()
def resample_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=-1, is_mono=None, dtype="int16"):
    audio_data_dtype = np.int16
    if dtype == "int16":
        audio_data_dtype = np.int16
    elif dtype == "float32":
        audio_data_dtype = np.float32
    audio_data = np.frombuffer(audio_chunk, dtype=audio_data_dtype)

    # try to guess if the audio is mono or stereo
    if is_mono is None:
        is_mono = audio_data.shape[0] % 2 != 0

    if target_channels < 2 and not is_mono:
        # Reshape the array to separate the channels
        audio_data = audio_data.reshape(-1, 2)

    if target_channels == -1 and not is_mono:
        # Average the left and right channels to create mono audio
        audio_data = audio_data.mean(axis=1)
    elif target_channels == 0 or target_channels == 1 and not is_mono:
        # Extract the first channel (left channel) data
        audio_data = audio_data[:, target_channels]
    elif target_channels == 2 and is_mono:
        # Duplicate the mono channel to create left and right channels
        audio_data = np.column_stack((audio_data, audio_data))
        # Flatten the array and convert it back to int16 dtype
        audio_data = audio_data.flatten()

    # Resample the audio data to the desired sample rate
    audio_data = resampy.resample(audio_data, recorded_sample_rate, target_sample_rate)
    # Convert the resampled data back to int16 dtype
    return np.asarray(audio_data, dtype=audio_data_dtype)
