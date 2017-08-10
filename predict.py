#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

from tensorflow.contrib.session_bundle import exporter

from nltk.tokenize import word_tokenize

""" This part of the code is to deal with error related to ascii characters in dataset """
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
""" end """

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print ('Start intent classification!!')
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

checkpointdir=sys.argv[1]

print("Checkpoint dir ",checkpointdir)

# CHANGE THIS: Load data. Load your own data here
x_text=[sys.argv[2]]

# CHANGE THIS: To what ever is he max sentence length used while training the model. Usually this is automated when we use vocabulary but here since we are not using vocabulary hence we fix it.
max_document_length = 23

print("Assuming max length of a sentence {}".format(max_document_length))

print("Loading Glove Vectors into a dictionary")
dct={}
#Update glove path here or get it as an argument
with open('/home/cdpai/tensorflow-models/rateresponses/glove_pretrained_word_embedding/glove.6B.100d.txt') as fh:
        for ln in fh:
                ln = ln.replace('\n','')
                tokens = ln.split(" ")
                vector = [float(x) for x in tokens[1:]]
                dct[tokens[0]]=vector

print("converting inference data into array of vectors")
x_inference=[]
zero_vec = list(np.zeros(FLAGS.embedding_dim))
for ln in x_text:
        words = word_tokenize(ln)
        arr_of_vectors=[]
        for i,wrd in enumerate(words):
		wrd = wrd.lower()
		if i + 1 > max_document_length :
			break
                if wrd in dct:
                        vec = dct.get(wrd)
                        arr_of_vectors.append(vec)
                else:
                        arr_of_vectors.append(zero_vec)
	if len(words) < max_document_length:
	        for i in range(max_document_length-len(words)):
        	        arr_of_vectors.append(zero_vec)
        x_inference.append(arr_of_vectors)

x_numpy=np.array(x_inference,dtype='float32')

print (x_inference)

print("\nEvaluating...\n")

# Predicting
# ==================================================
checkpoint_dir=os.path.join('/home/cdpai/tensorflow-models/intent_classification/runs',checkpointdir,'checkpoints')
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
print("Checkpoint file ",checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        print ('import_meta_graph :: '+checkpoint_file+'.meta')
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        
	node_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
	print(node_list)
        print ('saver restore!!')

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("embedding/embedded_chars").outputs[0]
        print(input_x)
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        print(dropout_keep_prob)

        # Tensors we want to evaluate
        print('Start Predictions')
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        print(predictions)

        scores = graph.get_operation_by_name("output/scores").outputs[0]
	print(scores)
        
        # Generate batches for one epoch
        print("Start batches")
        batches = data_helpers.batch_iter(list(x_inference), FLAGS.batch_size, 1, shuffle=False)
        #print(batches)

        # Collect the predictions here
        print('all_predictions!!')
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            batch_scores = sess.run(scores,{input_x: x_test_batch, dropout_keep_prob: 1.0})
            #print(batch_scores)
            siged_scores = tf.sigmoid(batch_scores)
            #soft_scores = tf.nn.softmax(batch_scores)
            siged_scores_val = sess.run(siged_scores)
            #soft_scores_val = sess.run(soft_scores)
            print(siged_scores_val)
            #print(soft_scores_val)
            #print(batch_scores)
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        print('End!!')

predictions_human_readable = np.column_stack((np.array(x_text), all_predictions))
print(predictions_human_readable)

# Data folder contains files with index being the class id, for example if the read file names are as ['class1','class2'] then class1 has index 0 and when the prediction return 0 then class1 is displayed

files=os.listdir('./data')
print(files)
for pred in predictions_human_readable:
    print("Description:",pred[0],", Category:",files[int(float(pred[1]))])
