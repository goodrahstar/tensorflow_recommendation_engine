# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:28:20 2016

@author: rahulkumar

Training Code: This code is used to load the corpus data and perform training using Neural network.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import corpus_handel 



data = pd.read_excel('data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
print 'data loded'

for header in data.columns.values:
    if header == 'Topic':
        pass
    else:
        data[header] = data[header].map({'Applicable':1,'Not Applicable':0})

data = data.fillna(0)


x, v, v_in = corpus_handel.load_data(data)
print 'data encoded'

prod=pd.DataFrame(data=np.array(x[0]))
size = len(prod.columns)

result = pd.concat([prod, data.iloc[0:len(x[0]),1:len(data.columns)]], axis=1)

print 'frame prepared of size = ', size


predictors = result.iloc[:,0:size]
targets = result.iloc[:,len(prod.columns):]
del data, v, v_in,x,prod 


print ' shuffles and input data ready'

##Hyper Parameters

SCALE_NUM_TRIPS = 100000
trainsize = int(len(result['Parties']) * 0.8)
testsize = len(result['Parties']) - trainsize
npredictors = len(predictors.columns)
noutputs = 78 #number of classes to predict
nhidden = 5
numiter = 1000
modelfile = '/tmp/trained_model_test'

with tf.Session() as sess:
  feature_data = tf.placeholder("float", [None, npredictors])
  target_data = tf.placeholder("float", [None, noutputs])
  
  weights1 = tf.Variable(tf.truncated_normal([npredictors, nhidden], stddev=0.01), name='weight1')
  weights2 = tf.Variable(tf.truncated_normal([nhidden, noutputs], stddev=0.01),name='weight2')
  
  biases1 = tf.Variable(tf.ones([nhidden]))
  biases2 = tf.Variable(tf.ones([noutputs]))
  
  model = (tf.matmul(tf.nn.relu(tf.matmul(feature_data, weights1) + biases1), weights2, name= 'output') + biases2) * SCALE_NUM_TRIPS

  cost = tf.nn.l2_loss(model - target_data, name='loss')

  training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
  

  init = tf.initialize_all_variables()
  sess.run(init)

  saver = tf.train.Saver({'weights1' : weights1, 'biases1' : biases1, 'weights2' : weights2, 'biases2' : biases2})
  for iter in xrange(0, numiter):
    sess.run(training_step, feed_dict = {
        feature_data : predictors[:trainsize].values,
        target_data : targets[:trainsize].values.reshape(trainsize, noutputs)
      })

    if iter%1000 == 0:
      print '{0} error={1}'.format(iter, np.sqrt(cost.eval(feed_dict = {
          feature_data : predictors[:trainsize].values,
          target_data : targets[:trainsize].values.reshape(trainsize, noutputs)
      }) / trainsize))
    
  filename = saver.save(sess, modelfile, global_step=numiter)
  print 'Model written to {0}'.format(filename)

  print 'testerror={0}'.format(np.sqrt(cost.eval(feed_dict = {
          feature_data : predictors[trainsize:].values,
          target_data : targets[trainsize:].values.reshape(testsize, noutputs)
      }) / testsize))
      
