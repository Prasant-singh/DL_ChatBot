import random
import json
from nltk.corpus import stopwords
import torch
import numpy as np
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
 
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
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

bot_name = "Bot"
print("Let's chat! Ask me anything related to Anti AI")
stop_words = set(stopwords.words('english'))
custom_stop_words = stop_words - {"AI", "ai"} 

while True:

    sentence = input("You: ")
    
    sentence = [word for word in sentence.split() if word.lower() not in custom_stop_words]
    sentence=" ".join(sentence)
    print("STOPWORD REMOVED left sentence:---->",sentence)
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X should be a NumPy array")

    X = X.reshape(1, X.shape[0])
    
    try:
        X = torch.from_numpy(X).to(device)
    except RuntimeError as e:
        print(f"Error converting to tensor: {e}")
        break

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print("TAG - ",tag)
                print("Probability ",prob.item())
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                print()

    else:

        print(f"{bot_name}: I do not understand... please simplify ")
