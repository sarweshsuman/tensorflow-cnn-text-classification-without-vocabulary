import tensorflow as tf
import numpy as np
import sys
import helpers


class TextCNN(object):
    """
    A CNN for text classification.
    This is a special CNN that takes input label and sentenes in form of vectors, this model does not do embedding lookup. embedding lookup is done by train.py
    This has been done to be able to receive vectors for all the words in glove pre-trained data, this will help with similar words giving same predictions.
    If we were to use vocabulary, then in general the words in a sentence is converted to array of ids and the words not in vocabuary are assigned 0, this 0 has no speialized vector
    in glove so lets assume that vector have all 0 items, so they contribute nothing to the prediction. 
    The downside of using it is that there are high chances of prediction completely going off the rails when we become creative with the words in the sentence.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for output and dropout
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.embedding_size=embedding_size

        print("Y Shape")
        print(self.input_y.shape)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_chars = tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name="embedded_chars")
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("Filter shape=",filter_size)
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                conv=tf.Print(conv,[conv],message=" conv2d")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.h_drop=tf.Print(self.h_drop,[self.h_drop],message=" after dropping")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1)) # correct_predictions will be like (1,1,1,0,1,1,1...)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")  
