import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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
#from keras import backend as K
#K.set_image_dim_ordering('th')
#%matplotlib inline
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



    from collections import Counter
    train_captions_path='/media/raid6/shivam/imagecaption/data/train_data.txt'
    train_img_path='/media/raid6/shivam/imagecaption/data/train_resize/'
    with open(train_captions_path, "r") as data:
        train_data = ast.literal_eval(data.read())
    counter = Counter()
    img_list=[]
    path_list=[]
    cap_list=[]
    unique_words=[]
    dictall={}
    id_list=[]
    word_freq=Counter()
    
    ids = train_data.keys()
    #save_in_file = open('/media/raid6/shivam/imagecaption/simple_cnn_img_attention/test_result100.txt', "w")
    for i, id in enumerate(ids):
        
        image_name=str(train_data[id][0]+'_'+train_data[id][1])
        #id_list.append(image_name)    
        #print("\nimage name : {}".format(train_data[id][1]))
        complete_path=train_img_path+train_data[id][0]+'_'+train_data[id][1]
        #path_list.append(complete_path)
        #print("\nimage path : {}".format(complete_path))
        #caption='<start> '+test_data[id][2]+' <end>'
        caption=train_data[id][2]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        word_freq.update(tokens)
        if len(tokens)>=1:
            id_list.append(image_name) 
            path_list.append(complete_path)
            cap_part=caption.split()
            sub_cap=[]
            for cap in cap_part:
    
                sub_cap.append(str(cap))
                if cap not in unique_words:
                    unique_words.append(cap)
            cap_list.append(sub_cap)
                
            
            
            
            
            
            
            
            
            
    Data={'Id':id_list,
          'Path':path_list,
          'Genre':cap_list
             
        
    }
    for word in unique_words:
        word_list=[]
        
        print(word)
        for cap in cap_list:
            #print(cap)
            if word in cap:
                word_list.append(1)
                
            else:
                word_list.append(0)
        Data[word]=word_list
        print(Data[word])
    from pandas import DataFrame 
    col=['Id','Path','Genre']
    for word in unique_words:
        col.append(str(word))
    train = DataFrame (Data, columns = col)
    
    
    
    train_image = []
    for i in tqdm(range(train.shape[0])):
        img = image.load_img(train['Path'][i],target_size=(227,227,3))
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
    X_train = np.array(train_image)
        
    
    
    y_train = np.array(train.drop(['Id', 'Path','Genre'],axis=1))
    #y.shape
    
    
    
    
    
    
    
    
    test_captions_path='/media/raid6/shivam/imagecaption/data/val_data.txt'
    test_img_path='/media/raid6/shivam/imagecaption/data/val_resize/'
    with open(test_captions_path, "r") as data:
        test_data = ast.literal_eval(data.read())
    #counter = Counter()
    counter = Counter()
    img_list=[]
    path_list=[]
    cap_list=[]
    #unique_words=[]
    dictall={}
    id_list=[]
    
    ids = test_data.keys()
    #save_in_file = open('/media/raid6/shivam/imagecaption/simple_cnn_img_attention/test_result100.txt', "w")
    for i, id in enumerate(ids):
        
        image_name=str(test_data[id][0]+'_'+test_data[id][1])
        #id_list.append(image_name)    
        #print("\nimage name : {}".format(train_data[id][1]))
        complete_path=test_img_path+test_data[id][0]+'_'+test_data[id][1]
        #path_list.append(complete_path)
        #print("\nimage path : {}".format(complete_path))
        #caption='<start> '+test_data[id][2]+' <end>'
        caption=test_data[id][2]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        word_freq.update(tokens)
        if len(tokens)>=1:
            id_list.append(image_name) 
            path_list.append(complete_path)
            cap_part=caption.split()
            sub_cap=[]
            for cap in cap_part:
    
                sub_cap.append(str(cap))
                if cap not in unique_words:
                    unique_words.append(cap)
            cap_list.append(sub_cap)
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    valData={'Id':id_list,
          'Path':path_list,
          'Genre':cap_list
             
        
    }
    for word in unique_words:
        word_list=[]
        
        print(word)
        for cap in cap_list:
            #print(cap)
            if word in cap:
                word_list.append(1)
                
            else:
                word_list.append(0)
        valData[word]=word_list
        print(valData[word])
    from pandas import DataFrame 
    col=['Id','Path','Genre']
    for word in unique_words:
        col.append(str(word))
    test= DataFrame (valData, columns = col)
            
        
        
    test_image = []
    for i in tqdm(range(test.shape[0])):
        img = image.load_img(test['Path'][i],target_size=(227,227,3))
        img = image.img_to_array(img)
        img = img/255
        test_image.append(img)
    X_test = np.array(test_image)
        
    
    
    y_test = np.array(test.drop(['Id','Path', 'Genre'],axis=1))
    #y.shape  
    
    
    
    
    
    
    
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(100*100*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(997))
    model.add(Activation('softmax')) 
      
      
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model.load_weights("weights.hdf5")
    
    model.summary()
    
    '''
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    lr=0.2
    #model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])
    
    '''
    
    history=model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test), batch_size=1024)
    
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    train_acc=history.history['acc']
    val_acc=history.history['val_acc']
    
    numpy_loss_train = np.array(train_loss)
    np.savetxt('/media/raid6/shivam/imagecaption/labelclass/train_loss_history.txt', numpy_loss_train, delimiter=",")
    
    
    numpy_loss_val = np.array(val_loss)
    np.savetxt('/media/raid6/shivam/imagecaption/labelclass/val_loss_history.txt', numpy_loss_val, delimiter=",")
    
    numpy_acc_train = np.array(train_acc)
    np.savetxt('/media/raid6/shivam/imagecaption/labelclass/train_acc_history.txt', numpy_acc_train, delimiter=",")
    
    numpy_acc_val = np.array(val_acc)
    np.savetxt('/media/raid6/shivam/imagecaption/labelclass/val_acc_history.txt', numpy_acc_val, delimiter=",")
    
    
    
    
    
    
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("/media/raid6/shivam/imagecaption/labelclass/alex1_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    
    
    model.save_weights('/media/raid6/shivam/imagecaption/labelclass/alex1_model.h5')
    print("Saved model to disk")
    
    
    
    
    ##scores = model.evaluate(X_train, y_train, verbose=0)
    ##print(model.metrics_names)
    ##print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    ''' 
    # load json and create model
    json_file = open("/media/raid6/shivam/imagecaption/labelclass/alex1_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('/media/raid6/shivam/imagecaption/labelclass/alex1_model.h5')
    print("Loaded model from disk")
     
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    score = loaded_model.evaluate(X_train, y_train, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    '''
    
    
    
    
    #model.save('/media/raid6/shivam/imagecaption/labelclass/alex1_model.h5')
    
    #model = load_weights('/media/raid6/shivam/imagecaption/labelclass/alex1_model.h5')
    '''
    # evaluate the model
    model=loaded_model
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(train_acc)
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(test_acc)
    '''
    '''
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.savefig('/media/raid6/shivam/imagecaption/labelclass/foo.png')
    #pyplot.show()
    '''
    '''
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    classes=unique_words
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    np.set_printoptions(precision=2)
    plt.savefig('/media/raid6/shivam/imagecaption/labelclass/cm.png')
    
    
    #server.launch(model,input_folder='/media/raid6/shivam/imagecaption/labelclass/',temp_folder='/media/raid6/shivam/imagecaption/labelclass/filters')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''    
    
    
    
    
    
    
    
    
    test_img_path='/media/raid6/shivam/imagecaption/data/test_resize/'
    test_captions_path='/media/raid6/shivam/imagecaption/data/test_data.txt'
    with open(test_captions_path, "r") as data:
        test_data = ast.literal_eval(data.read())
    counter = Counter()
    all_test_img_paths=[]
    all_test_captions=[]
    ids = test_data.keys()
    save_in_file = open('/media/raid6/shivam/imagecaption/labelclass/alex_test_result40.txt', "w")
    for i, id in enumerate(ids):
        pred=[]
        image_name=test_data[id][1]
        print("\nimage name : {}".format(test_data[id][1]))
        complete_path=test_img_path+test_data[id][0]+'_'+test_data[id][1]
        print("\nimage path : {}".format(complete_path))
        #caption='<start> '+test_data[id][2]+' <end>'
        caption=test_data[id][2]
        img = image.load_img(complete_path,target_size=(227,227,3))
        img = image.img_to_array(img)
        img = img/255
        classes = np.array(train.columns[2:])
        proba = model.predict(img.reshape(1,227,227,3))
        top_3 = np.argsort(proba[0])[:-4:-1]
        for i in range(3):
            print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
            pred.append(classes[top_3[i]])
        save_in_file.write(complete_path)
        save_in_file.write(',')
        text='original caption : '+str(caption)
        save_in_file.write(text)
        save_in_file.write(',')
        text='predicted caption : '+str(pred)
        save_in_file.write(text)
        save_in_file.write('\n')
        #save_in_file.write('Proba: ')
        #save_in_file.write(proba)
        #save_in_file.write('\n')
    
                
    save_in_file.close()    
        
        
        
        
        
        
     
        
  
    
if __name__ == '__main__':
    main()



            
            
            
        
    
    
    
    
