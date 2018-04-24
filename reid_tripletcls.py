#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:00:18 2017

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
import tensorflow as tf


import numpy.linalg as la
from IPython import embed


#欧式距离 


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
        query_result_ind[ind]=euclidSimilar2(query_all[ind],test_all,top_num)
#        if np.mod(ind,100)==0:
#            print('query_ind '+str(ind))
    return query_result_ind


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
    return topp1
"================================"

def triplet_loss(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred,axis=1)
    batch = batch_num
    ref1 = y_pred[0:batch,:]
    pos1 = y_pred[batch:batch+batch,:]
    neg1 = y_pred[batch+batch:3*batch,:]
    dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
    dis_pos = K.sqrt(dis_pos)
    dis_neg = K.sqrt(dis_neg)
    a1 = 0.6
    d1 = K.maximum(0.0,dis_pos-dis_neg+a1)
    return K.mean(d1)

def triplet_hard_loss(y_true, y_pred):
    global SN
    global PN
    feat_num = SN*PN # images num
    y_pred = K.l2_normalize(y_pred,axis=1)
    feat1 = K.tile(K.expand_dims(y_pred,axis = 0),[feat_num,1,1])
    feat2 = K.tile(K.expand_dims(y_pred,axis = 1),[1,feat_num,1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),axis = 2) + K.epsilon() # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
        if i != PN-1:
            negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
        else:
            negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = tf.concat([negetive,negs],axis = 0)
    positive = K.max(positive,axis=1)
    negetive = K.min(negetive,axis=1) 
    a1 = 0.6
    loss = K.mean(K.maximum(0.0,positive-negetive+a1))
    return loss 
 
def msml_loss(y_true, y_pred):
    global SN
    global PN
    feat_num = SN*PN # images num
    y_pred = K.l2_normalize(y_pred,axis=1)
    feat1 = K.tile(K.expand_dims(y_pred,axis = 0),[feat_num,1,1])
    feat2 = K.tile(K.expand_dims(y_pred,axis = 1),[1,feat_num,1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),axis = 2) + K.epsilon() # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
        if i != PN-1:
            negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
        else:
            negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = tf.concat([negetive,negs],axis = 0)
    positive = K.max(positive)
    negetive = K.min(negetive) 
    a1 = 0.6
    loss = K.mean(K.maximum(0.0,positive-negetive+a1))
    return loss 
    
def tf_debug_print(tensor):
    with tf.Session():
        print(tensor.eval())

"================================"
SN = 3
PN =24
identity_num = 751
print('loading data...')

# add your loading validation data here
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
class_triplet_model = Model(input=base_model.input, output=[preds,feature])
#training IDE model for all layers
for layer in net.layers:
   layer.trainable = True

# train
batch_num = PN
#adam = optimizers.Adam(lr=0.00001)
lr = 0.001
adam = optimizers.adam(lr)
#adam = optimizers.SGD(lr=0.0001,momentum=0.9,decay=0.0005)
net.compile(optimizer=adam, loss='categorical_crossentropy',metric ='accuracy')
class_triplet_model.compile(optimizer=adam,loss=['categorical_crossentropy',triplet_hard_loss],loss_weights=[1.0,1.0])
# you can load pre-trained model here
#net.load_weights('triplet_hard_aug_save.h5')
from load_img_data import get_triplet_data,get_triplet_hard_data
from aug import aug_nhw3

while(True):
    train_img,train_label = get_triplet_data(PN)  #the data in a batch: A1 B1 C1 ...PN1 A2 B2 C2 ... PN2 G K S ... Negative(PN) 
    train_img,train_label = get_triplet_hard_data(SN,PN) # the data in a batch : A1 A2 A3... ASN B1 B2 B3... BSN ... PN1 PN2 PN3... PNSN
    train_img = aug_nhw3(train_img)
    train_img = preprocess_input(train_img)
    train_label_onehot = np_utils.to_categorical(train_label,identity_num)
    class_triplet_model.fit(train_img,
                            y=[train_label_onehot,np.ones([PN*SN,2048])],
                            shuffle=False,epochs=1,batch_size=PN*SN)  # for triplet loss: SN = 3

    ind = ind+1
    # you can do sth here
    #if np.mod(ind,1000)==0:
       # test_feature = feature_model.predict(test_img)
       # test_feature = normalize(test_feature)
       # query_feature = feature_model.predict(query_img)
       # query_feature = normalize(query_feature)
       # top1=single_query(query_feature,test_feature,query_label,test_label,test_num=1000)
       # lr = lr*0.9
       # adam = optimizers.adam(lr)
