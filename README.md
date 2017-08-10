# tensorflow-cnn-text-classification-without-vocabulary


A CNN for text classification.

This is a special CNN text classification that takes input label as usual and sentenes in form of vectors, this model does not 
do embedding lookup. Embedding lookup is done by train.py and predict.py

This has been done to be able to receive vectors for all the words in glove pre-trained data, this will help with similar words 
giving same predictions. If we were to use vocabulary, then normally the words in a sentence is converted to array of ids and 
the words not in vocabuary are assigned 0, this 0 has no speialized vector in glove so lets assume that vector has all 0 items, 
so they contribute nothing to the prediction and hence they are not open to different forms of same sentence.

The downside of this approach is that there are high chances of prediction completely going off the rails when 
we become creative with the words in the sentence.

*** Experiment with it.
