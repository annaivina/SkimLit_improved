import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import TextVectorization
import pickle


class DataLoader:

    def __init__(self, dataset_name, num_classes, lable_map, batch_size):
        self.dataset_name= dataset_name
        self.batch_size=batch_size
        self.num_classes = num_classes

        self.one_hot = OneHotEncoder(sparse_output=False)
        # self.char_vectorizer = TextVectorization(max_tokens=96,#Alphabet count + 2
        #                             output_sequence_length = 338,#found out after checking len 
        #                             standardize=None,
        #                             pad_to_max_tokens=True)
    
        #Converting lables to integers and one hot encode them
        self.lable_map = lable_map 
        keys_tensor = tf.constant(lable_map)
        values_tensor = tf.range(len(lable_map), dtype=tf.int64)
        self.label_lookup = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
            num_oov_buckets=1  # Handle unknown labels safely
        )
   


    def load_data(self):
        print("Loading the tfds datasets into the space...")
        dataset = tfds.load(name=self.dataset_name, split=['train', 'valid', 'test'],  shuffle_files=False)
        
       
        train_data = dataset[0].map(lambda x: self.preprocess_examples(x, "train")).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        valid_data = dataset[1].map(lambda x: self.preprocess_examples(x, "valid")).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_data  = dataset[2].map(lambda x: self.preprocess_examples(x, "test")).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_data, valid_data, test_data
    


    def preprocess_examples(self, example):
         

        #Convert text labels to numeric indices using the lookup table
        label_index = self.label_lookup.lookup(example["labels"])

        # One-hot encode the label indices
        label_one_hot = tf.one_hot(label_index, depth=self.num_classes)

        one_hot_line_nb = tf.one_hot(example["line_nb"], depth=15)
        one_hot_total_lines = tf.one_hot(example["total_lines"], depth=21)


        return {
             "line_nb": one_hot_line_nb,
             "total_lines": one_hot_total_lines,
             "sentence": example["sentence"],
             "chars": example["chars"],
        }, label_one_hot


         

    


