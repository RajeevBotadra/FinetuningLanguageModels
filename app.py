import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

#Title
st.title("Toxicity Analysis of Tweets")

#Subtitle
st.markdown("## Using a fine-tuned roBERTa model")

st.markdown("Link to the app - [Basic Sentiment Analyzer on 🤗 Spaces](https://huggingface.co/spaces/rbbotadra/toxicity-analyzer-app)")

#Dropdown menu for model options
model_opt = st.selectbox(
    'Select a finetuned model:',
    ('roBERTa tuned on Tweets [6-class toxicity analysis]',''))
st.write('Model selected:', model_opt)

#Tuned Model Path
MODEL = f"./cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#Label mapping
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

text = st.text_input("Text Input", "War is cruelty. There is no use trying to reform it. The crueler it is, the sooner it will be over.")
st.write("Current Text:", text)

if st.button('Run Model'):
    MODEL = f"./cardiffnlp/twitter-roberta-base-sentiment/config.json"
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        st.write(f"{i+1}) {l}: {np.round(float(s), 4)}")
else:
    st.write("Press button to run Senitment Analysis.")


