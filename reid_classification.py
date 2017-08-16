#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:21:54 2017

@author: luohao
"""

import numpy as np


from keras import optimizers
from keras.utils import np_utils, generic_utils
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.layers.core import Lambda
from sklearn.preprocessing import normalize
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal


import numpy.linalg as la
from IPython import embed

#欧式距离 
def euclidSimilar(query_ind,test_all,top_num):  
    le = len(test_all)
    dis = np.zeros(le)
    for ind in range(le):
        sub = test_all[ind]-query_ind
        dis[ind] = la.norm(sub)
    ii = sorted(range(len(dis)), key=lambda k: dis[k])
#    print(ii[:top_num+1])
    return ii[1:top_num+1]

def euclidSimilar2(query_ind,test_all):  
    le = len(test_all)
    dis = np.zeros(le)
    for ind in range(le):
        sub = test_all[ind]-query_ind
        dis[ind] = la.norm(sub)    
    ii = sorted(range(len(dis)), key=lambda k: dis[k])
#    embed()
#    print(ii[:top_num+1])
    return ii

def get_top_ind(query_all,test_all,top_num):
    query_num = len(query_all)
    query_result_ind = np.zeros([query_num,top_num],np.int32)
    for ind in range(query_num):
        query_result_ind[ind]=euclidSimilar(query_all[ind],test_all,top_num)
#        if np.mod(ind,100)==0:
#            print('query_ind '+str(ind))
    return query_result_ind

def get_top_label(query_result_ind):
    num = len(query_result_ind)
    top_num = len(query_result_ind[0])
    query_top_label = np.zeros([num,top_num],np.int32)
    for ind in range(num):
        for ind2 in range(top_num):
            query_top_label[ind][ind2]= test_label[query_result_ind[ind][ind2]]
#        if np.mod(ind,100)==0:
#            print('query_label '+str(ind))
#        print(query_top_label[ind])
    return query_top_label

def get_top_acc(test_label,query_result_label):
       query_label= test_label
       top1 = 0
       top5 = 0
       top10 = 0 
       for ind in range(len(query_result_label)):
           query_temp_label = query_result_label[ind]-query_label[ind]
#           print(query_temp_label)
           query_temp = np.where(query_temp_label==0)
           if len(query_temp[0] >0):
               if query_temp[0][0]<1:
                   top1 = top1+1
               if query_temp[0][0]<5:
                   top5 = top5+1    
               if query_temp[0][0]<10:
                   top10 = top10+1
       ind = ind +1
       top1 = top1/ind*1.0
       top5 = top5/ind*1.0
       top10 = top10/ind*1.0
       print(str(ind)+' query images')
       return top1,top5,top10


def single_query(query_feature,test_feature,query_label,test_label,test_num):
    test_label_set = np.unique(test_label)
    #single_num = len(test_label_set)
    test_label_dict={}
    topp1=0
    topp5=0
    topp10=0
    for ind in range(len(test_label_set)):
        test_label_dict[test_label_set[ind]]=np.where(test_label==test_label_set[ind])
    for ind in range(test_num):
        query_int = np.random.choice(len(query_label))
        label = query_label[query_int]        
        temp_int = np.random.choice(test_label_dict[label][0],1)
        temp_gallery_ind = temp_int 
        for ind2 in range(len(test_label_set)):
            temp_label = test_label_set[ind2]
            if temp_label != label:
                temp_int = np.random.choice(test_label_dict[temp_label][0],1)
                temp_gallery_ind = np.append(temp_gallery_ind,temp_int)
        single_query_feature =  query_feature[query_int]
        test_all_feature = test_feature[temp_gallery_ind]
        result_ind = euclidSimilar2(single_query_feature,test_all_feature)
        query_temp = result_ind.index(0)
        if query_temp<1:
            topp1 = topp1+1
        if query_temp<5:
            topp5 = topp5+1    
        if query_temp<10:
            topp10 = topp10+1
    topp1 =topp1/test_num*1.0
    topp5 =topp5/test_num*1.0
    topp10 =topp10/test_num*1.0
    print('single query')
    print('top1: '+str(topp1)+'\n')
    print('top5: '+str(topp5)+'\n')
    print('top10: '+str(topp10)+'\n')  
    

"================================"

identity_num = 6273

print('loading data...')
## please add your loading validation data here 


#from load_market_img import get_img
#query_img,test_img,query_label,test_label=get_img()
test_img =preprocess_input(test_img)
query_img = preprocess_input(query_img)

''''''''''''''''''''''''''
datagen = ImageDataGenerator(horizontal_flip=True)

# load pre-trained resnet50
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
x = base_model.output
feature = Flatten(name='flatten')(x)
fc1 = Dropout(0.5)(feature)
preds = Dense(identity_num, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(fc1)  #default glorot_uniform
net = Model(input=base_model.input, output=preds)
feature_model = Model(input=base_model.input, output=feature)

#training IDE model for all layers
for layer in net.layers:
   layer.trainable = True

# train
batch_size = 16
#step 1
adam = optimizers.Adam(lr=0.001)
net.compile(optimizer=adam, loss='categorical_crossentropy',metric ='accuracy')

# your can add a pre-trained model here
#net.load_weights('net_ide.h5')
from load_img_data import get_train_img
  
ind = 0
while(True):
    train_img,train_label = get_train_img(8000) #add your loading training data here
    train_img = preprocess_input(train_img)
    train_label_onehot = np_utils.to_categorical(train_label,identity_num)
    net.fit_generator(datagen.flow(train_img, train_label_onehot, batch_size=batch_size),
                        steps_per_epoch=len(train_img)/batch_size, epochs=1)
    ind = ind+1
    # your can add sth here
    # if np.mod(ind,100) == 0:

