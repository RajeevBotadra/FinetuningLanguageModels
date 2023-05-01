---
title: Toxicity Analyzer of Tweets
emoji: 📹
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

## Milestone 1 Update:
Installation instructions (Windows 10 Machine):

1) Install Docker Desktop for Windows from this link: https://docs.docker.com/desktop/install/windows-install/
2) Run the downloaded .exe file
3) Enable use of WSL 2 backend(already enabled through Powershell on my machine)
4) In VSCode, install Docker and Dev Container extensions


## Milestone 2 Update:
See Streamlit Space for App: https://huggingface.co/spaces/rbbotadra/basic-sentiment-analysis-app

Note that the space is synced with the main branch of this repository.

## Milestone 3 Update:
See Streamlit Space for App: https://huggingface.co/spaces/rbbotadra/toxicity-analyzer-app

Note that the space is synced with the milestone-3 branch of this repository.

## Milestone 4 Update:
See Landing Page: https://sites.google.com/njit.edu/toxicity-analyzer-app/home 

See video demo of app: https://youtu.be/Mpfrlbr0-LU

See Documentation.md or read below for project documentation.

# Documentation

## Introduction

The internet has been a medium of social connections since its conception, and over the past few decades it has become more accessible than any such medium in history. This accessibility has fostered as much toxic content, whether it be hateful comments or bullying posts, as it has productive, positive, and constructive content. Therefore, an important task of many web forums -particularly giant social media platforms such as Twitter- to address toxic content in a timely manner. Using human reviewers may provide a reliable way of weeding out toxic content, but is an impractical and costly approach when scaled (and also introduces personal biases). The Toxic Analyzer App is a more robust solution to the problem, using a Language Model for a sentiment analysis task of Tweets. Specifically, the app uses a roBERTa Large Language Model that has been fine-tuned on a dataset of Tweets. The model performs six-category sentiment classification for the following classes: Toxic, Insult, Obsence, Identity Hate, Threat, and Severe Toxic. 

## Tuning & Results

The dataset was curated by Jigsaw for their Toxic Comment Classification Challenge. The training dataset is nearly 68MB and contains tens of thousands of tweets, each with an associated (one-hot encoded) set of class labels. The base roBERTa model was downloaded using HuggingFace Transformers API. Next, a custom dataset class was created to convert the raw data to a pytorch dataset. Additionally, there were some simple preprocessing steps for the data. Lastly, the roBERTa model was tuned on the dataset. Note that if you do not have a CUDA capable device (i.e. an Nvidia GPU) and are using the CPU for training, it will take a very long time (~100hrs on a 16-core, 32-thread CPU). However, with a CUDA enable GPU, this is drastically reduced to (~4hrs) depending on the specific GPU.

After tuning, the model achieved 86% accuracy on the test data and 91% accuracy on the training data.

## Deployment

The tuned model was deployed in a HuggingFace Space using a Streamlit App, the link is found in the README.md file. The app takes input text (with a default text present upon loading), passes it through the tuned model, and presents the classification results in a table format. In addition to the app, a web page was created to present a simple introduction to the app and all relevant links, and a video demo of the Streamlit app in the HuggingFace space was also created to briefly demonstrate the capability of the app.
