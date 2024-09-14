import random
import json
import nltk
from nltk.corpus import stopwords
import torch
import numpy as np
from model import NeuralNet

nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

class Chatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.custom_stop_words = self.stop_words - {"AI", "ai"}

        self.intents = self.load_intents()
        self.data = torch.load("data.pth")
        self.model = self.load_model()

    def load_intents(self):
        with open('intents.json', 'r') as json_data:
            return json.load(json_data)

    def load_model(self):
        input_size = self.data["input_size"]
        hidden_size = self.data["hidden_size"]
        output_size = self.data["output_size"]
        all_words = self.data['all_words']
        tags = self.data['tags']
        model_state = self.data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
        model.load_state_dict(model_state)
        model.eval()
        self.all_words = all_words
        self.tags = tags
        return model

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        return self.stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence):
        sentence_words = [self.stem(word) for word in tokenized_sentence]
        bag = np.zeros(len(self.all_words), dtype=np.float32)
        for idx, w in enumerate(self.all_words):
            if w in sentence_words:
                bag[idx] = 1
        return bag

    def process_input(self, sentence):
        stop_words = set(stopwords.words('english'))
        custom_stop_words = stop_words - {"AI", "ai"} 
        sentence = [word for word in sentence.split() if word.lower() not in custom_stop_words]
        sentence = " ".join(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace("anti ai", "antiai").replace("anti.ai", "antiai").replace("anti-ai", "antiai").replace("anti_ai", "antiai")

        if "and" in sentence:
            parts = sentence.split("and")
        else:
            parts = sentence.split("also")

        responses = []

        for part in parts:
            part = self.tokenize(part.strip())
            X = self.bag_of_words(part)
            X = torch.from_numpy(X.reshape(1, X.shape[0])).to(self.device)

            output = self.model(X)
            probs = torch.softmax(output, dim=1)

            _, predicted = torch.max(output, dim=1)
            tag = self.tags[predicted.item()]
            prob = probs[0][predicted.item()]

            if prob.item() > 0.85:
                for intent in self.intents['intents']:
                    if tag == intent["tag"]:
                        response = random.choice(intent['responses'])
                        if response not in responses:
                            responses.append(response)
            else:
                responses.append(f"I do not understand ... please simplify")

        return ' '.join(responses)
