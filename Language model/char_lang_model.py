
# coding: utf-8

# In[1]:

from keras import optimizers
import nltk
import json
import ast
#from quiver_engine import server
import numpy as np
#from getdata import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
import numpy as np
from keras.callbacks import History
#keras.callbacks.History()
import six
import tensorflow as tf
import time
import os
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  


def main():
  
  SHAKESPEARE_TXT = '/media/raid6/shivam/langmodels/data/new_persona.txt'
  
  tf.logging.set_verbosity(tf.logging.INFO)
  
  def transform(txt, pad_to=None):
      """
      Transform chars in txt to ascii values np.array, drop non-ascii chars.
      """
      # drop any non-ascii characters
      output = np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)
      if pad_to is not None:
          output = output[:pad_to]
          output = np.concatenate([
              np.zeros([pad_to - len(txt)], dtype=np.int32),
              output,
          ])
      return output
  
  def training_generator(seq_len=500, batch_size=128):
      """A generator yields (source, target) arrays for training."""
      with tf.gfile.GFile(SHAKESPEARE_TXT, 'r') as f:
          txt = f.read()
  
      tf.logging.info('Input text [%d] %s', len(txt), txt[:50])
      source = transform(txt)
      while True:
          # One batch of offsets for sampling sequences randomly.
          offsets = np.random.randint(low=0, high=len(source) - seq_len, size=batch_size)
  
          # Our model uses sparse crossentropy loss, but Keras requires labels
          # to have the same rank as the input logits.  We add an empty final
          # dimension to account for this.
          yield (
              np.stack([source[idx:idx + seq_len] for idx in offsets]),
              np.expand_dims(
                  np.stack([source[idx + 1:idx + seq_len + 1] for idx in offsets]),
                  -1),
          )
  
  x, y = six.next(training_generator(seq_len=10, batch_size=2))
  print('x, shape:', x.shape, ', x:', x)
  print('y, shape:', y.shape, ', y:', y)
  
  
  # In[2]:
  
  
  transform('123 bc ABC')
  
  
  # In[3]:
  
  
  EMBEDDING_DIM = 1024
  MAX_TOKENS = 256
  def lstm_model(seq_len=500, batch_size=None, stateful=True, max_tokens = 256):
      """Language model: predict the next char given the current char."""
      source = tf.keras.Input(
          name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)
  
      embedding = tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=EMBEDDING_DIM)(source)
      lstm_1 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(embedding)
      lstm_2 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(lstm_1)
      predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_tokens, activation='softmax'))(lstm_2)
      model = tf.keras.Model(inputs=[source], outputs=[predicted_char])
      model.compile(
          optimizer=tf.train.RMSPropOptimizer(learning_rate=0.1),
          loss='sparse_categorical_crossentropy',
          metrics=['sparse_categorical_accuracy'])
      return model
  
  
  # In[22]:
  
  
  tf.keras.backend.clear_session()
  '''
  
  training_model = lstm_model(seq_len=500, batch_size=128, stateful=False, max_tokens = MAX_TOKENS)
  
  # tpu_model = tf.contrib.tpu.keras_to_tpu_model(
  #     training_model,
  #     strategy=tf.contrib.tpu.TPUDistributionStrategy(
  #         tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
  
  training_model.summary()
  
  
  #csv_logger = CSVLogger('/media/raid6/shivam/log.csv', append=True, separator=';')
  history=training_model.fit_generator(
      training_generator(seq_len=500, batch_size=128),
      steps_per_epoch=50,
      epochs=50, # 10
      
  )
  
  
  train_loss=history.history['loss']
  #val_loss=history.history['val_loss']
  #train_acc=history.history['acc']
  #val_acc=history.history['val_acc']
  
  numpy_loss_train = np.array(train_loss)
  np.savetxt('/media/raid6/shivam/imagecaption/labelclass/chartrain_loss_history50epoch.txt', numpy_loss_train, delimiter=",")
  
  
  #numpy_loss_val = np.array(val_loss)
  #np.savetxt('/media/raid6/shivam/imagecaption/labelclass/val_loss_history.txt', numpy_loss_val, delimiter=",")
  
  #numpy_acc_train = np.array(train_acc)
  #np.savetxt('/media/raid6/shivam/imagecaption/labelclass/train_acc_history.txt', numpy_acc_train, delimiter=",")
  
  #numpy_acc_val = np.array(val_acc)
  #np.savetxt('/media/raid6/shivam/imagecaption/labelclass/val_acc_history.txt', numpy_acc_val, delimiter=",")
  
  training_model.save_weights('bard-GPU-epoch50.h5', overwrite=True)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  #save_in_file_loss = open('/media/raid6/shivam/char_lang5.txt', "w")
  #save_in_file_loss.write(history)
  #save_in_file.close()
  
 
  
  # In[179]:
  
  '''
  
  BATCH_SIZE = 5
  PREDICT_LEN = 500
  
  # Keras requires the batch size be specified ahead of time for stateful models.
  # We use a sequence length of 1, as we will be feeding in one character at a 
  # time and predicting the next character.
  prediction_model = lstm_model(seq_len=1, batch_size=BATCH_SIZE, stateful=True, max_tokens = MAX_TOKENS)
  prediction_model.load_weights('bard-GPU-epoch50.h5') # bard-TPU-epoch10.h5 or bard-GPU-epoch2.h5
  
  # We seed the model with our initial string, copied BATCH_SIZE times
  
  seed_txt = 'selfie'
  
  # Text chars to ascii values np.array.
  seed = transform(seed_txt)
  
  # Repeat seed for batch size times.
  seed = np.repeat(np.expand_dims(seed, 0), BATCH_SIZE, axis=0)
  
  prediction_model.summary()
  
  
  # In[180]:
  
  
  # First, run the seed forward to prime the state of the model.
  prediction_model.reset_states()
  
  # Kick start with the seed text.
  for i in range(len(seed_txt) - 1):
      prediction_model.predict(seed[:, i:i + 1])
  
  
  # In[181]:
  
  
  predictions = [seed[:, -1:]]
  predictions[-1].shape
  
  
  # In[182]:
  
  
  # Last chars in the batch
  last_char = predictions[-1]
  # Predict with only the last chars as input
  next_probits = prediction_model.predict(last_char)[:, 0, :]
  
  
  # In[183]:
  
  
  #import matplotlib.pyplot as plt
  #plt.plot(next_probits[1])
  #plt.xlabel('ascii')
  #plt.ylabel('probs')
  #plt.show()
  
  
  # In[184]:
  
  
  # Now we can accumulate predictions!
  
  # The last char's ascii value for each sequence in seed batch (5 sequences).
  predictions = [seed[:, -1]]
  
  # Predict PREDICT_LEN(250) in number of chars.
  for i in range(PREDICT_LEN):
      # Last chars in the batch
      last_char = predictions[-1]
      
      # Predict with only the last chars as inputs.
      next_probits = prediction_model.predict(last_char)[:, 0, :]
  
      # Sample from output distribution for each sample in the batch.
      next_idx = [
          np.random.choice(len(next_probits[0]), p=next_probits[i])
          for i in range(BATCH_SIZE)
      ]
      # Collect the sampled output char ascii values.
      predictions.append(np.asarray(next_idx, dtype=np.int32))
  
  
  # In[185]:
  
  
  # For each batch, ascii values -> chars.
  for i in range(BATCH_SIZE):
      print('PREDICTION %d\n\n' % i)
      p = [predictions[j][i] for j in range(PREDICT_LEN)]
      generated = ''.join([chr(c) for c in p])
      print(generated)
      print()
      assert len(generated) == PREDICT_LEN, 'Generated text too short'
  
  
  
if __name__ == '__main__':
    main()