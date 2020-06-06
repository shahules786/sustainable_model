from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import pandas as pd
import numpy as np 
import re
import os

from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D


tf.logging.set_verbosity(tf.logging.INFO)

CLASSES = {'financial':0,'cyber':1,'other':2}
MAX_SEQ_LENGTH = 100
VOCAB_FILE_PATH= None
PADWORD='wxz'


def load_train_eval(train_path,eval_path):

    train=pd.read_csv(train_path)
    test =pd.read_csv(eval_path)

    return ((train['text'].values.tolist(),train['target'].map(CLASSES).values),
            (test['test'].values,test['target'].map(CLASSES).values))


def vectorize_input(text):

    words  = tf.string_split(text)
    words  = tf.sparse_tensor_to_dense(words)


    table = tf.contrib.lookup.index_table_from_file(VOCAB_FILE_PATH,num_oov_buckets=0,delimeter=',')

    numbers = table.lookup(words)
    
    return numbers


def pad(feature,labels):

    non_zero_indices=tf.where(tf.not_equal(feature,tf.zeroes_like(feature)))
    non_zero_words = tf.gather(feature,non_zero_indices)
    non_zero_words = tf.squeeze(non_zero_words,axis=1)

    padded = tf.pad(non_zero_words,[MAX_SEQ_LENGTH])
    padded  = padded[-MAX_SEQ_LENGTH:]

    return (padded,labels)




def read_input(text,labels,batch_size,mode):

    x= tf.constant(text)
    x= vectorize_input(x)

    dataset = tf.data.Dataset.from_tensor_slices((x,labels))

    dataset = dataset.map(pad)

    if mode == tf.estimator.Modekeys.TRAIN :
            num_epochs=None
            dataset = dataset.shuffle(2020)
    elif mode==tf.estimator.Modekeys.EVAL:
            num_epochs=1

    return dataset.repeat(num_epochs).batch(batch_size)



def keras_estimator(model_dir,lr,config):


    embedding_matrix = get_embedding_matrix(word_index,embedding_path,embedding_dim)

    model = models.Sequential()

    num_features = len(word_index)+1


    model.add(Embedding(input_dim=num_features,output_dim=embedding_dim,
                        input_length=MAX_SEQ_LENGTH,weights=[embedding_matrix]),
                        trainable=is_trainable)

    model.add(Bidirectional(LSTM(150, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)))
    model.add(GlobalAvgPool1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
  
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model,model_dir=model_dir,config=config)
    return estimator



    def train_and_evaluate(output_dir,hparams):


        (train_texts,train_labels),(test_texts,test_labels)=load_train_eval('train.csv','test.csv')

        tokenizer = text.Tokenizer()
        tokenizer.fit_on_text(train_texts)


        tf.gfile.MkDir(output_dir) # directory must exist before we can use tf.gfile.open
        global VOCAB_FILE_PATH; VOCAB_FILE_PATH = os.path.join(output_dir,'vocab.txt')
        with tf.gfile.Open(VOCAB_FILE_PATH, 'wb') as f:
                f.write("{},0\n".format(PADWORD))  # map padword to 0
                for word, index in tokenizer.word_index.items():
                    if index < TOP_K: # only save mappings for TOP_K words
                        f.write("{},{}\n".format(word, index))


        runconfig = tf.estimator.RunConfig(save_checkpoints_steps=500)



         # Create TrainSpec
        train_steps = hparams['num_epochs'] * len(train_texts) / hparams['batch_size']
        train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:input_fn(
            train_texts,
            train_labels,
            hparams['batch_size'],
            mode=tf.estimator.ModeKeys.TRAIN),
             max_steps=train_steps
            )

        # Create EvalSpec
        exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda:input_fn(
            test_texts,
            test_labels,
            hparams['batch_size'],
            mode=tf.estimator.ModeKeys.EVAL),
            steps=None,
            exporters=exporter,
            start_delay_secs=10,
            throttle_secs=10
            )

        # Start training
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)





