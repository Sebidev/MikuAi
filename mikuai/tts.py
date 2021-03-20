"""
 MikuAi - Is a toolkit for creating Conversational AI applications.
 Copyright by Sebastian Bürger [sebidev]
"""

import torch
import numpy as np
from scipy.io.wavfile import write

def tts_t2waveglow(text):
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2 = tacotron2.to('cuda')
    tacotron2.eval()

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    # preprocessing
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

    # run the models
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
        audio = waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050

    write("audio.wav", rate, audio_numpy)
