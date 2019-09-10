from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import pickle
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import ast
import json
from collections import Counter

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Device configuration
device = torch.device('cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def process(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path,map_location=lambda storage, loc: storage))
    decoder.load_state_dict(torch.load(args.decoder_path,map_location=lambda storage, loc: storage))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    return sentence
    #plt.imshow(np.asarray(image))






def predict(img_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=img_path,help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='/Users/shivamgaur/Desktop/pytorch/cap/models/encoder-40-330.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/Users/shivamgaur/Desktop/pytorch/cap/models/decoder-40-330.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='/Users/shivamgaur/Desktop/pytorch/cap/models/vocab.pkl', help='path for vocabulary wrapper')
                        
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2, help='number of layers in lstm')
    args = parser.parse_args()
    predicted_caption=process(args)
    
    return predicted_caption



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict(file_path)
        hashtags_list=preds.split(' ')
        hashtags_list.remove('<start>')
        hashtags_list.remove('<end>')
        tags=[]
        for tag in hashtags_list:
            if tag not in tags:
                tags.append(tag)
        text=""
        for tag in tags:
            text=text+'  #'+str(tag)
        text=' '+text
        return text
    return None


if __name__ == '__main__':
    app.run(debug=True)

                        
