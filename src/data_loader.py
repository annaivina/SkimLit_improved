import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization



class DataLoader:

    def __init__(self, dataset_name, num_classes, lable_map, seq_length_chars, batch_size):
        self.dataset_name= dataset_name
        self.batch_size=batch_size
        self.num_classes = num_classes
        self.seq_length_chars=seq_length_chars

        #self.one_hot = OneHotEncoder(sparse_output=False)
        self.char_vectorizer = TextVectorization(max_tokens=96,#Alphabet count + 2
                                     output_sequence_length = seq_length_chars,#found out after checking len 
                                     standardize=None,
                                     pad_to_max_tokens=True)
    
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

        train_chars=[]
        for example in dataset[0].take(10000):
            char_text = example['chars'].numpy().decode("utf-8")
            train_chars.append(char_text)

        self.char_vectorizer.adapt(train_chars)
       
        train_data = dataset[0].map(lambda x: self.preprocess_examples(x)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        valid_data = dataset[1].map(lambda x: self.preprocess_examples(x)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_data  = dataset[2].map(lambda x: self.preprocess_examples(x)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_data, valid_data, test_data
    


    def preprocess_examples(self, example):
         

        #Convert text labels to numeric indices using the lookup table
        label_index = self.label_lookup.lookup(example["labels"])

        # One-hot encode the label indices
        label_one_hot = tf.one_hot(label_index, depth=self.num_classes)

        one_hot_line_nb = tf.one_hot(tf.cast(example["line_nb"], tf.int32), depth=15)
        one_hot_total_lines = tf.one_hot(tf.cast(example["total_lines"], tf.int32), depth=21)

        #Applying char vectorizes 
        print("Vectorizing")
        char_vectorized = self.char_vectorizer(example['chars'])


        return {
             "line_nb": one_hot_line_nb,
             "total_lines": one_hot_total_lines,
             "sentences": example["sentences"],
             "chars": char_vectorized,
        }, label_one_hot


         

    


