import numpy as np
import pandas as pd
import json

data_files_dev = '/Users/ankitasinha/UCI_CLASS/Capstone Project/In-House-Search-Engine-main/data/archive/dev-v1.1.json'
data_files_train = '/Users/ankitasinha/UCI_CLASS/Capstone Project/In-House-Search-Engine-main/data/archive/train-v1.1.json'

def squad_json_to_dataframe_train(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                  verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file, record_path)
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values, m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([m[['id', 'question', 'context']].set_index('id'), js.set_index('q_idx')], 1,
                     sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main


def squad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file, record_path)
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    #     ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    #     js['q_idx'] = ndx
    main = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main

# Training data
input_file_path = data_files_train
record_path = ['data','paragraphs','qas','answers']
# train = squad_json_to_dataframe_train(input_file_path=input_file_path,record_path=record_path)

input_file_path = data_files_dev
record_path = ['data','paragraphs','qas','answers']
verbose = 0
dev = squad_json_to_dataframe_dev(input_file_path=input_file_path,record_path=record_path)

import csv

dev.to_csv('dev_data.csv')
# train.to_csv('train_data.csv')

