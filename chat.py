import random
import json
import torch
from model import *
from nltk_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Shyam"

print("Let's chat! Type 'quit' to exit")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0]) # NOTE: remember the reshaping (when and why)
    X = torch.from_numpy(X).to(device) # NOTE: remember the conversion (when and why)

    output = model(X)
    _,predicted = torch.max(output,dim=1)
    #its .item for every torch object
    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}:{random.choice(intent['responses'])}")

    else:
        print(f"{bot_name}:I don't understand what you are saying..")





