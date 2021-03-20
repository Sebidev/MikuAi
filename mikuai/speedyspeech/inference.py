"""Synthesize audio from text
"""

import argparse, sys, os, time
import torch
from librosa.output import write_wav

from mikuai.speedyspeech.main import SpeedySpeech
from mikuai.speedyspeech.melgan.model.generator import Generator
from mikuai.speedyspeech.melgan.utils.hparams import HParam
from mikuai.speedyspeech.hparam import HPStft, HPText
from mikuai.speedyspeech.utils.text import TextProcessor
from mikuai.speedyspeech.functional import mask


def speedyspeech_tts(text_str, device_str):
    print('Loading model checkpoints')
    m = SpeedySpeech(
        device=device_str
    ).load('models/speedyspeech.pth', device_str)
    m.eval()

    checkpoint = torch.load('models/melgan.pth', device_str)
    hp = HParam("mikuai/speedyspeech/melgan/config/default.yaml")
    melgan = Generator(hp.audio.n_mel_channels).to(device_str)
    melgan.load_state_dict(checkpoint["model_g"])
    melgan.eval(inference=False)

    print('Processing text')
    txt_processor = TextProcessor(HPText.graphemes, phonemize=HPText.use_phonemes)
    text = [text_str]

    phonemes, plen = txt_processor(text)
    # append more zeros - avoid cutoff at the end of the largest sequence
    phonemes = torch.cat((phonemes, torch.zeros(len(phonemes), 5).long() ), dim=-1)
    phonemes = phonemes.to('cpu')

    print('Synthesizing')
    # generate spectrograms
    with torch.no_grad():
        spec, durations = m((phonemes, plen))


    # invert to log(mel-spectrogram)
    spec = m.collate.norm.inverse(spec)

    # mask with pad value expected by MelGan
    msk = mask(spec.shape, durations.sum(dim=-1).long(), dim=1).to('cpu')
    spec = spec.masked_fill(~msk, -11.5129)

    # Append more pad frames to improve end of the longest sequence
    spec = torch.cat((spec.transpose(2,1), -11.5129*torch.ones(len(spec), HPStft.n_mel, 5).to('cpu')), dim=-1)

    # generate audio
    with torch.no_grad():
        audio = melgan(spec).squeeze(1)

    print('Saving audio')
    # TODO: cut audios to proper length
    for i,a in enumerate(audio.detach().cpu().numpy()):
        write_wav(('output.wav'), a, HPStft.sample_rate, norm=False)
