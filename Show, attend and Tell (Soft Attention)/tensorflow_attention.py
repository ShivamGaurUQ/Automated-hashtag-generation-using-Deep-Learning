# Import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9
from __future__ import absolute_import, division, print_function, unicode_literals
#pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
import numpy as np
from keras.callbacks import History
#keras.callbacks.History()
import six
import tensorflow as tf
import time
import os
# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
#import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import tensorflow as tf
tf.enable_eager_execution()
from keras.callbacks import ModelCheckpoint
import keras
from tqdm import tqdm
config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
import nltk
import argparse
from collections import Counter
#from pycocotools.coco import COCO
import ast
global SEEDTEXT
def main():
    train_captions_path='/media/raid6/shivam/imagecaption/data/train_data.txt'
    val_captions_path='/media/raid6/shivam/imagecaption/data/val_data.txt'
    
    train_img_path='/media/raid6/shivam/imagecaption/data/train_resize/'
    val_img_path='/media/raid6/shivam/imagecaption/data/val_resize/'
    with open(train_captions_path, "r") as data:
        train_data = ast.literal_eval(data.read())
    counter = Counter()
    all_train_img_paths=[]
    all_train_captions=[]
    ids = train_data.keys()
    for i, id in enumerate(ids):
        #print("\ni and id {} , {}".format(i, id))
        #print("\ntrain_data_key_val {}: {}".format(id, train_data[id]))
        #print("\ncategory : {}".format(train_data[id][0]))
        #print("\nimage name : {}".format(train_data[id][1]))
        #print("\ncaption : {}".format(train_data[id][2]))
        complete_path=train_img_path+train_data[id][0]+'_'+train_data[id][1]
        #print("\nimage path : {}".format(complete_path))
        all_train_img_paths.append(complete_path)
        caption='<start> '+train_data[id][2]+' <end>'
        all_train_captions.append(caption)

    #print(len(all_train_img_paths))
    #print(len(all_train_captions))
    
    
    with open(val_captions_path, "r") as vdata:
            val_data = ast.literal_eval(vdata.read())
    counter = Counter()
    #all_val_img_paths=[]
    #all_val_captions=[]
    vids = val_data.keys()
    for i, id in enumerate(vids):
        #print("\ni and id {} , {}".format(i, id))
        #print("\nval_data_key_val {}: {}".format(id, val_data[id]))
        #print("\ncategory : {}".format(val_data[id][0]))
        #print("\nimage name : {}".format(val_data[id][1]))
        #print("\ncaption : {}".format(val_data[id][2]))
        complete_path=val_img_path+val_data[id][0]+'_'+val_data[id][1]
        #print("\nimage path : {}".format(complete_path))
        all_train_img_paths.append(complete_path)
        caption='<start> '+val_data[id][2]+' <end>'
        all_train_captions.append(caption)

    #print(len(all_train_img_paths))
    #print(len(all_train_captions))
    
      
    train_captions=all_train_captions
    img_name_vector=all_train_img_paths
    
    
    
    def load_image(image_path):
        img = tf.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize_images(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    image_model = tf.keras.applications.InceptionV3(include_top=False, 
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
    # getting the unique images
    #encode_train = sorted(set(img_name_vector))
    
    # feel free to change the batch_size according to your system configuration
    #image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    #image_dataset = image_dataset.map(
      #load_image, num_parallel_calls=4).batch(16)
    
    #for img, path in tqdm(image_dataset):
        #print("\nimage path {} : {}".format(img, path))
        #batch_features = image_features_extract_model(img)
        #batch_features = tf.reshape(batch_features,
                                    #(batch_features.shape[0], -1, batch_features.shape[3]))
    
        #for bf, p in zip(batch_features, path):
            #path_of_feature = p.numpy().decode("utf-8")
            #np.save(path_of_feature, bf.numpy())

    
            
    # This will find the maximum length of any caption in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)
    
    # The steps above is a general process of dealing with text processing
    
    # choosing the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                      oov_token="<unk>", 
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    
    tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value <= top_k}
    # putting <unk> token in the word2idx dictionary
    tokenizer.word_index[tokenizer.oov_token] = top_k + 1
    tokenizer.word_index['<pad>'] = 0
    
    # creating the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    
    # creating a reverse mapping (index -> word)
    index_word = {value:key for key, value in tokenizer.word_index.items()}
    
    # padding each vector to the max_length of the captions
    # if the max_length parameter is not provided, pad_sequences calculates that automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    
    # calculating the max_length 
    # used to store the attention weights
    max_length = calc_max_length(train_seqs)
    
    # Create training and validation sets using 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, 
                                                                        cap_vector, 
                                                                        test_size=0.1, 
                                                                        random_state=0)    
          
          



    



    
    # feel free to change these parameters according to your system's configuration
    
    BATCH_SIZE = 1
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = len(tokenizer.word_index)
    # shape of the vector extracted from InceptionV3 is (64, 2048)
    # these two variables represent that
    features_shape = 2048
    attention_features_shape = 64
    
    
    # loading the numpy files 
    def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap
    
    
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    
    # using map to load the numpy files in parallel
    # NOTE: Be sure to set num_parallel_calls to the number of CPU cores you have
    # https://www.tensorflow.org/api_docs/python/tf/py_func
    dataset = dataset.map(lambda item1, item2: tf.py_func(
              map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)
    
    # shuffling and batching
    dataset = dataset.shuffle(BUFFER_SIZE)
    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/batch_and_drop_remainder
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    
    
    def gru(units):
      # If you have a GPU, we recommend using the CuDNNGRU layer (it provides a 
      # significant speedup).
        if tf.test.is_gpu_available():
            print('gpu running')
        
            return tf.keras.layers.CuDNNGRU(units, 
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform')
        else:
            return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform')
        
        
        
    class BahdanauAttention(tf.keras.Model):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)
      
        def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
            hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, 64, hidden_size)
            score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
            attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * features
            context_vector = tf.reduce_sum(context_vector, axis=1)
        
            return context_vector, attention_weights
        
        
    class CNN_Encoder(tf.keras.Model):
        # Since we have already extracted the features and dumped it using pickle
        # This encoder passes those features through a Fully connected layer
        def __init__(self, embedding_dim):
            super(CNN_Encoder, self).__init__()
            # shape after fc == (batch_size, 64, embedding_dim)
            self.fc = tf.keras.layers.Dense(embedding_dim)
            
        def call(self, x):
            x = self.fc(x)
            x = tf.nn.relu(x)
            return x
        
        
    class RNN_Decoder(tf.keras.Model):
        def __init__(self, embedding_dim, units, vocab_size):
            super(RNN_Decoder, self).__init__()
            self.units = units
    
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = gru(self.units)
            self.fc1 = tf.keras.layers.Dense(self.units)
            self.fc2 = tf.keras.layers.Dense(vocab_size)
        
            self.attention = BahdanauAttention(self.units)
            
        def call(self, x, features, hidden):
        # defining attention as a separate model
            context_vector, attention_weights = self.attention(features, hidden)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
            output, state = self.gru(x)
        
        # shape == (batch_size, max_length, hidden_size)
            x = self.fc1(output)
        
        # x shape == (batch_size * max_length, hidden_size)
            x = tf.reshape(x, (-1, x.shape[2]))
        
        # output shape == (batch_size * max_length, vocab)
            x = self.fc2(x)
    
            return x, state, attention_weights
    
        def reset_state(self, batch_size):
            return tf.zeros((batch_size, self.units))
        
        
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    
    
    optimizer = tf.train.AdamOptimizer()
    
    # We are masking the loss calculated for padding
    def loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)
    
    # adding this in a separate cell because if you run the training cell 
    # many times, the loss_plot array will be reset
    loss_plot = []
    
    checkpoint_dir = '/media/raid6/shivam/imagecaption/colab_attention/beyond_40training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    
    
    EPOCHS=60
    print('\n---------------------- EPOCHS : {}-------------------------\n'.format(EPOCHS))
    loss_per_epoch={}
    perplex_per_epoch={}
    save_in_file_loss = open('/media/raid6/shivam/imagecaption/colab_attention/beyondepoch_loss.txt', "w")
    save_in_file_perplex=open('/media/raid6/shivam/imagecaption/colab_attention/beyondepoch_perplex.txt', "w")
    
    checkpoint.restore(tf.train.latest_checkpoint('/media/raid6/shivam/imagecaption/colab_attention/40training_checkpoints'))
    
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        
        
        
        for (batch, (img_tensor, target)) in enumerate(dataset):
            loss = 0
            
            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = decoder.reset_state(batch_size=target.shape[0])
    
            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
            
            with tf.GradientTape() as tape:
                features = encoder(img_tensor)
                
                for i in range(1, target.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden, _ = decoder(dec_input, features, hidden)
    
                    loss += loss_function(target[:, i], predictions)
                    
                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)
            
            total_loss += (loss / int(target.shape[1]))
            
            variables = encoder.variables + decoder.variables
            
            gradients = tape.gradient(loss, variables) 
            
            optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
            
            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Perplexity {}'.format(epoch + 1, 
                                                              batch, 
                                                              loss.numpy() / int(target.shape[1]), tf.exp(loss.numpy() / int(target.shape[1])) ))
                print('\nLength of Cap_Vector {}'.format(len(cap_vector)))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / len(cap_vector))
       
        # saving (checkpoint) the model every 1 epoch
        if (epoch + 1) % 1 == 0:
            print('Saving checkpoint')
            checkpoint.save(file_prefix = checkpoint_prefix)
            print('\nsaved ')
            print(epoch+1)
            
            
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
                                             total_loss/len(cap_vector)))
                                             
        print('Length of Cap_Vector {}'.format(len(cap_vector)))
        loss_per_epoch[epoch+1]=total_loss/len(cap_vector)
        loss_text=str(epoch+1)+' : '+str(loss_per_epoch[epoch+1])
        save_in_file_loss.write(loss_text)
        save_in_file_loss.write('\n')
        
        print('\nloss_text : '+loss_text)
        
        perplex_per_epoch[epoch+1]=tf.exp(total_loss/len(cap_vector))
        perplex_text=str(epoch+1)+' : '+str(perplex_per_epoch[epoch+1])
        save_in_file_perplex.write(perplex_text)
        save_in_file_perplex.write('\n')
        print('\nperplex_text : '+perplex_text)
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
        
        
    #return checkpoint
    save_in_file_loss.close()
    save_in_file_perplex.close()
    
    
    
    print('\nFinished training')
    #checkpoint_dir = '/media/raid6/shivam/imagecaption/colab_attention/training_checkpoints'
    
    
    
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    
    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))
    
        hidden = decoder.reset_state(batch_size=1)
    
        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    
        features = encoder(img_tensor_val)
    
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
    
        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
    
            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
    
            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])
    
            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot
    
            dec_input = tf.expand_dims([predicted_id], 0)
    
        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot    
        
        
        
        
        
   

        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    test_captions_path='/media/raid6/shivam/imagecaption/data/test_data.txt'
    #val_captions_path='/media/raid6/shivam/imagecaption/data/val_data.txt'
    
    test_img_path='/media/raid6/shivam/imagecaption/data/test_resize/'
    #val_img_path='/media/raid6/shivam/imagecaption/data/val_resize/'
    with open(test_captions_path, "r") as data:
        test_data = ast.literal_eval(data.read())
    counter = Counter()
    all_test_img_paths=[]
    all_test_captions=[]
    ids = test_data.keys()
    save_in_file = open('/media/raid6/shivam/imagecaption/colab_attention/test_result2iter_40to60.txt', "w")
    #image_pathplot='/media/raid6/shivam/imagecaption/colab_attention/img.jpg'
    
    for i, id in enumerate(ids):
        print("\ni and id {} , {}".format(i, id))
        print("\ntrain_data_key_val {}: {}".format(id, test_data[id]))
        print("\ncategory : {}".format(test_data[id][0]))
        print("\nimage name : {}".format(test_data[id][1]))
        print("\ncaption : {}".format(test_data[id][2]))
        complete_path=test_img_path+test_data[id][0]+'_'+test_data[id][1]
        print("\nimage path : {}".format(complete_path))
        all_test_img_paths.append(complete_path)
        caption='<start> '+test_data[id][2]+' <end>'
        all_test_captions.append(caption)
        
        save_in_file.write(complete_path)
        save_in_file.write(',')
        text='original caption : '+str(caption)
        save_in_file.write(text)
        save_in_file.write(',')
    
    #save_caption=open('/media/raid6/shivam/imagecaption/colab_attention/pred_hashtag.txt',"w")
    result,attention_plot = evaluate(complete_path)
    #SEEDTEXT=str(result)
    print('\nGenerating hashtags. Saving in file...............\n')
    text='predicted caption : '+str(result)
    print('\n')
    print(text)
    save_in_file.write(text)
    save_in_file.write('\n')
    

        

    
    save_in_file.close()
    

      
      
      
      
      
         
      
      
      
      
  
  
      
  
      
        
        
        








if __name__ == '__main__':
    
    main()  
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    