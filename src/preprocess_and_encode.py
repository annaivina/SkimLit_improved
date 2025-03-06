import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



class Preprocess:

    def __init__(self, data_dir, data_type):
        self.data_dir = data_dir
        self.data_type = data_type
        self.one_hot = OneHotEncoder(sparse_output=False)
        self.label_encode = LabelEncoder()
       
    
    def get_lines(self):
        """
            Takes lines from the raw .txt file and created the array of text 
        """
        with open(self.data_dir, "r") as f:
            lines = f.readlines()
        return lines
    
    @staticmethod
    def load_and_convert(file):
        """
            Uses the arrays of text to create a dataframe with target, text, line_nb, total_lines
        """
        abstract_line =''
        dict_abst = []
        
        for line in file:
            if line.startswith("###"):
                abstract_line = ''
                
            elif line == '\n':
                abstract_line_split = abstract_line.splitlines()
                for line_nb, line in enumerate(abstract_line_split):
                    line_data ={}
                    target_split = line.split("\t")
                    line_data['target'] = target_split[0]#take the first argument aka the target name
                    line_data['text'] = target_split[1]#take the second argument as the text
                    line_data['line_nb'] = line_nb
                    line_data['total_lines'] = len(abstract_line_split)
                    dict_abst.append(line_data)
            else:
                abstract_line +=line #will accumulate the lines after the ### and before \n
                
        return pd.DataFrame(dict_abst)
    
   
    
    @staticmethod
    def convert_sent_to_chars(sentance):
        return " ".join(list(sentance))



def save_tf_dataset(lines, totals, sentances, chars, labels, data_type, out_dir):
    
    def serialize(line, total, sentance, char, label):
        feature_dict = {
            "line_nb": tf.train.Feature(float_list=tf.train.FloatList(value=[line])),  # Float values
            "total_lines": tf.train.Feature(float_list=tf.train.FloatList(value=[total])),  # Float values
            "sentences": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sentance.encode()])),  # sentances and chards are stoires as  bytes
            "chars": tf.train.Feature(bytes_list=tf.train.BytesList(value=[char.encode()])),  # Store as bytes
            "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()]))  # One-hot encoded labels
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()
    
    tfrecord_path = f"{out_dir}/{data_type}.tfrecord"
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i in range(len(sentances)):
            example = serialize(
                line= float(lines[i]),
                total = float(totals[i]),#to convert it to the list otherwise it will not be properly written 
                sentance = sentances[i],
                char = chars[i],
                label = labels[i]
            )
            writer.write(example)
            
        print(f"{data_type} datset is saved")

