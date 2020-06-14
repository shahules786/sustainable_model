from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from . import model


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        default="gs://risk_model/model_dir/",
        required=False
    )
    parser.add_argument(
        '--train_data_path',
        help='can be a local path or a GCS url (gs://...)',
        default='gs://risk_model/Datasets/risk_train.csv',
        required=False
    )
    parser.add_argument(
        '--eval_data_path',
        help='can be a local path or a GCS url (gs://...)',
        default='gs://risk_model/Datasets/risk_test.csv',
        required=False
    )
    parser.add_argument(
        '--embedding_path',
        help='OPTIONAL: can be a local path or a GCS url (gs://...). \
              Download from: https://nlp.stanford.edu/projects/glove/',
        default="gs://risk_model/Datasets/glove.42B.300d.txt",
        required=False
    )
    parser.add_argument(
        '--embedding_dim',
        help='OPTIONAL: can be a local path or a GCS url (gs://...). \
              Download from: https://nlp.stanford.edu/projects/glove/',
        default=300,
        type=int,
        required=False
    )
    parser.add_argument(
        '--num_epochs',
        help='number of times to go through the data, default=10',
        default=10,
        type=float
    )
    parser.add_argument(
        '--batch_size',
        help='number of records to read during each training step, default=128',
        default=128,
        type=int,
        required=False

    )
    parser.add_argument(
        '--learning_rate',
        help='learning rate for gradient descent, default=.001',
        default=1e-3,
        type=float,
        required=False
    )
    #parser.add_argument(
    #    '--native',
    #    help='use native in-graph pre-processing functions',
    #    type=int,
    #    required=True
    #)

    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    output_dir = hparams.pop('output_dir')



    model.train_and_evaluate(output_dir,hparams)