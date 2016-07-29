# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 07:36:12 2016

@author: rahulkumar

Load the trained model and execute the result.
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import corpus_handel 
import os, json


##Hyper parameters
SCALE_NUM_TRIPS = 100000
numiter = 10000000
modelfile = os.getcwd() +'/tmp/trained_model_test'
npredictors = 46
noutputs = 78 
nhidden = 5


def model(requirement= [ ]):
    
    jsondata = {
               'Topic': ' ', 
               'Prediction' : 0.0
               }
               
#    data = pd.read_pickle(os.getcwd() +'/data')
    data = pd.read_excel('data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
    profile_names = data.columns.tolist()
    profile_names = profile_names[1:]
    
    x, v, _ = corpus_handel.load_data(data)
     

    
    requirement = requirement[0].strip()
    requirement = corpus_handel.clean_str(requirement)
    requirement = requirement.split(" ")
    
    num_padd = x[0].shape[1] - len(requirement)
    requirement = requirement + ["<PAD/>"] * num_padd
    
    
    for word in requirement:
        if not v.has_key(word):
            requirement[requirement.index(word)] = "<PAD/>"
    
#    print 'Processed req=>', requirement
    x = np.array([v[word] for word in requirement])
    
    input = pd.DataFrame(np.array([x]))
    
        
    with tf.Session() as sess:
        filename = modelfile + '-' + str(numiter)
        feature_data = tf.placeholder("float", [None, npredictors])
      
        weights1 = tf.Variable(tf.truncated_normal([npredictors, nhidden], stddev=0.01))
        weights2 = tf.Variable(tf.truncated_normal([nhidden, noutputs], stddev=0.01))
      
        biases1 = tf.Variable(tf.ones([nhidden]))
        biases2 = tf.Variable(tf.ones([noutputs]))
        
        saver = tf.train.Saver({'weights1' : weights1, 'biases1' : biases1, 'weights2' : weights2, 'biases2' : biases2})
    
        saver.restore(sess, filename)
    
        feature_data = tf.placeholder("float", [None, npredictors])
        predict_operation = (tf.matmul(tf.nn.relu(tf.matmul(feature_data, weights1) + biases1), weights2) + biases2) * SCALE_NUM_TRIPS
        predicted = sess.run(predict_operation, feed_dict = {
            feature_data : input.values
          })
    result =[]
    
    thres = (predicted[0].min() + predicted[0].max()  )/2

    for profile, val in enumerate(predicted[0]):
        if val>thres:
            jsondata = {'Topic': profile_names[profile], 'Prediction' : 1}
            result.append(jsondata)
        else:
            jsondata = {'Topic': profile_names[profile], 'Prediction' : 0}
            result.append(jsondata)
    jsonreturn = json.dumps(result)                    
    return jsonreturn

#print model(requirement=['The Solution must have the ability to process the "Inward pain 001" message received from Customer'])
