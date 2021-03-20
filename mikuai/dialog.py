"""
 MikuAi - Is a toolkit for creating Conversational AI applications.
 Copyright by Sebastian Bürger [sebidev] and Patrick Loeber
"""

import torch
import json
import random

from mikuai.model import NeuralNet
from mikuai.nltk_utils import tokenize
from mikuai.nltk_utils import stem
from mikuai.nltk_utils import bag_of_words

def dialog(input_text):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('models/intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "models/dialog.pth"
    data = torch.load(FILE)


    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Miku"
    sentence = input_text
  #  if sentence == "quit":
  #      break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    text = None

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                text = random.choice(intent['responses'])
    else:
        print(f"{bot_name}: i do not understand that")
        text = "i do not understand that"

    return text
