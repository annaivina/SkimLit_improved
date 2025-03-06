import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np


class SkimlitDataset(tfds.core.GeneratorBasedBuilder):
    
    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial version with processed SkimLit data. (200k) "
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For the raw input files in tfrecord format. Whiuch have been processed using th processing.py

    The processed tensorflow_dataset can also be downloaded from: git:

    """ 
    ## Use manual instructions if you need to manually provide the path for your raw inputs 

    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
 
    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
        builder=self,
        description="Dataset for classifying medical abstracts",
        features=tfds.features.FeaturesDict({
            "line_nb": tfds.features.Tensor(shape=(), dtype=np.float32),
            "total_lines": tfds.features.Tensor(shape=(), dtype=np.float32),
            "sentences": tfds.features.Text(),#To store it as a raw text
            "chars": tfds.features.Text(),
            "labels": tfds.features.Text()

        }),
        supervised_keys=None,
        )
    
    def _split_generators(self, dl_manager):
        
        raw_data_dir = dl_manager.manual_dir
        print("Loading the datasets for the processing ")
        return {
            "train": self._generate_examples(os.path.join(raw_data_dir, "train.tfrecord")),
            "valid": self._generate_examples(os.path.join(raw_data_dir, "valid.tfrecord")),
            "test": self._generate_examples(os.path.join(raw_data_dir, "test.tfrecord")),
        }
    
    def _generate_examples(self, filepath):
        print(f"processing file {filepath}")
        dataset = tf.data.TFRecordDataset(filepath)

        feature_description = {
            "line_nb": tf.io.FixedLenFeature([], tf.float32),
            "total_lines": tf.io.FixedLenFeature([], tf.float32),
            "sentences": tf.io.FixedLenFeature([], tf.string),
            "chars": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.FixedLenFeature([], tf.string),
        }
        
        for i, example in enumerate(dataset):
            parsed_example=tf.io.parse_single_example(example, feature_description)
            
            yield i, {
            "line_nb": float(parsed_example["line_nb"].numpy()),  
            "total_lines": float(parsed_example["total_lines"].numpy()),
            "sentences": parsed_example["sentences"].numpy().decode("utf-8"),  
            "chars": parsed_example["chars"].numpy().decode("utf-8"),  
            "labels": parsed_example["labels"].numpy().decode("utf-8"),  
            }