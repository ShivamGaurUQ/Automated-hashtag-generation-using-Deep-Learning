import nltk
nltk.download('punkt')
import pickle
import argparse
from collections import Counter
import os
import sys
import shutil
import random
from sklearn.model_selection import train_test_split
import sklearn
from PIL import Image







if __name__ == '__main__':
    
    with open('/media/raid6/shivam/imagecaption/data/tag_list.txt', 'r')as file: 
        all_tags=file.readlines() 
    tags_perImg = [x.strip() for x in all_tags]
    
    with open('/media/raid6/shivam/imagecaption/data/data_list.txt', 'r')as file: 
        all_img=file.readlines() 
    path_perImg = [x.strip() for x in all_img]

    
    #shuffling origina data (image path and image caption together in same order)
    shuffle_with_order = list(zip(path_perImg, tags_perImg))
    random.shuffle(shuffle_with_order)
    path_perImg, tags_perImg = zip(*shuffle_with_order)
    
    #creating training and validation sets
    
    img_train, img_val, tags_train, tags_val = train_test_split(path_perImg, 
                                                                    tags_perImg, 
                                                                    test_size=0.2, 
                                                                    random_state=0)
    
    #creating validation and test sets
    img_vald, img_test, tags_vald, tags_test = train_test_split(img_val, 
                                                                    tags_val, 
                                                                    test_size=0.5, 
                                                                    random_state=0)
                                                                    
    
    
    
    
    
    
    
    
    #create train repo
    source='/media/h/harrison_images/HARRISON/instagram_dataset/'
    dest='/home/mobaxterm/Desktop/HARRISON/train/'
    train_dict={}
    i=0
    

    for img, caption in zip(img_train,tags_train):
        image_details=[]
        comp=img.split('/')
        image_details.append(comp[2])
        image_details.append(caption)
        
        
        full_name=source+comp[1]+'/'+comp[2]
        jpgfile = Image.open(full_name)

        print(jpgfile.bits, jpgfile.size, jpgfile.format)
        image_details.append(full_name)
        
        if (full_name.endswith(".jpg")):
            print('yes')
            shutil.copy(full_name, dest)
            train_dict[i]=image_details
        
  
   
    
    


    
    
