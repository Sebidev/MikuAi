"""
 MikuAi - Is a toolkit for creating Conversational AI applications.
 Copyright by Sebastian Bürger [sebidev]
"""

import getopt, sys
import requests
from tqdm import tqdm
from mikuai.test import test_func
from mikuai.dialog import dialog
from mikuai.stt import deepspeech_stt
from mikuai.speedyspeech.inference import speedyspeech_tts

argumentList = sys.argv[1:]

options = "whtpr:"

long_options = ["server", "help", "train", "pull", "run"]

try:
    arguments, values = getopt.getopt(argumentList, options, long_options)

    for currentArgument, currentValue in arguments:

        if currentArgument in ("-s", "--server"):
            speedyspeech_tts('See you later, thanks for visiting', 'cpu')

        elif currentArgument in ("-h", "--help"):
            print("MikuAi Version: v1.0.0")
            print("Is a toolkit for creating Conversational AI applications.")
            print()
            print("Arguments:")
            print("miku.py --pull     #Pull the pre-models for the AI.")
            print("miku.py --train    #Training the data sets and generating models so that they can then be used.")
            print("miku.py --help     #Shows this help text.")
            print("miku.py --run      #Executing AI with microphones processes with DeepSpeech further and generates a voice for speech output.")
            print("miku.py --server   #Starting a HTTP-Server receives audio file and processes with DeepSpeech further and generates a voice for speech output.")

        elif currentArgument in ("-t", "--train"):
            print("Displaying file_name:", sys.argv[2])
            test_func()

        elif currentArgument in ("-p", "--pull"):
            print("Downloading deepspeech models ...")
            r = requests.get('https://github.com/mozilla/DeepSpeech/releases/download/v0.9.1/deepspeech-0.9.1-models.pbmm', allow_redirects=True)
            open('models/deepspeech.pbmm', 'wb').write(r.content)
            r = requests.get('https://github.com/mozilla/DeepSpeech/releases/download/v0.9.1/deepspeech-0.9.1-models.scorer', allow_redirects=True)
            open('models/deepspeech.scorer', 'wb').write(r.content)
            print("Downloading speedyspeech models ...")
            r = requests.get('https://github.com/seungwonpark/melgan/releases/download/v0.1-alpha/nvidia_tacotron2_LJ11_epoch3200.pt', allow_redirects=True)
            open('models/melgan.pth', 'wb').write(r.content)
            r = requests.get('https://github.com/janvainer/speedyspeech/releases/download/v0.2/speedyspeech.pth', allow_redirects=True)
            open('models/speedyspeech.pth', 'wb').write(r.content)


        elif currentArgument in ("-r", "--run"):
            text_deepspeech = deepspeech_stt()
            print('YOU: ' + text_deepspeech)
            text_dialog = dialog(text_deepspeech)
            speedyspeech_tts(text_dialog, 'cpu')

except getopt.error as err:
    print(str(err))
