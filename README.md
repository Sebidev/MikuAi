MikuAi: Is a toolkit for easy creating Conversational AI applications.
-------------
Toolkit makes it possible to use neural network models for your conversation AI projects or applications in a simple way.

## Provided Models and Sources

Conversational-Bot from one straightforward with a Feed Forward Neural 

This code is adapted and modified from a number of original sources, including:

Speech to Text:
* DeepSpeech [paper](https://arxiv.org/abs/1412.5567) & [github-repository](https://github.com/mozilla/DeepSpeech)

Text to Speech:
* speedyspeech [paper](https://arxiv.org/abs/2008.03802) & [github-repository](https://github.com/janvainer/speedyspeech)
* Tacotron2 [paper](https://arxiv.org/abs/1712.05884) & [github-repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* WaveGlow [paper](https://arxiv.org/abs/1811.00002) & [github-repository](https://github.com/NVIDIA/waveglow)
* MelGAN [github-repository](https://github.com/seungwonpark/melgan)
* ```mikuai/speedyspeech/utils/stft.py``` - torch-stft by Prem Seetharaman (BSD 3-Clause License)
* ```mikuai/speedyspeech/pytorch_ssim``` - pytorch-ssim by Po-Hsun-Su (MIT)

Many thanks to Jan Vainer, Seung-won Park, NVIDIA, Mozilla

## Usage
```deepspeech_stt()```

```speedyspeech_tts(string, device)```

## First steps

Requirement is [Anaconda](https://www.anaconda.com/) installed is on your system.

If you want to use CUDA, you need the [CUDA toolkit and proprietary Nvidia driver](https://developer.nvidia.com/cuda-downloads) on your system.

```#For Linux and Windows create a environment with CUDA support environment
conda create -n mikuai python=3.7.0 pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch

#For Linux and Windows create a environment only CPU support environment
conda create -n mikuai python=3.7.0 pytorch==1.5.0 torchvision==0.6.0 cpuonly -c pytorch

#For MacOS create you a environment only CPU support
conda create -n mikuai python=3.7.0 pytorch==1.5.0 torchvision==0.6.0 -c pytorch

#Switch to the newly created environment
conda activate mikuai

#Install the remaining packages with pip
cd MikuAi
pip install -r requirements.txt

#For Windows you still have to do this to be able to install PyAudio
pip install pipwin
pipwin install pyaudio

#Pull the necessary data sets with you [1.2GB]
python mikuai.py --pull

#Testing of MikuAi
python mikuai.py --run
```

## License
This code is published under the BSD 3-Clause License.
