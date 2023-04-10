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

#title
st.title("Basic Sentiment Analyzer")

#subtitle
st.markdown("## Basic Sentiment Analysis using roBERTa and Streamlit for Sentiment Analysis")

st.markdown("Link to the app - [Basic Sentiment Analyzer on 🤗 Spaces](https://huggingface.co/spaces/rbbotadra/basic-sentiment-analysis-app)")
    
task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL, force_download=True)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

text = st.text_input("Text Input", "War is cruelty. There is no use trying to reform it. The crueler it is, the sooner it will be over.")
st.write("Current Sample Text:", text)

if st.button('Run Model'):
    MODEL = f"./cardiffnlp/twitter-roberta-base-{task}/config.json"
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
        st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
else:
    st.write("Press button to run Senitment Analysis.")


