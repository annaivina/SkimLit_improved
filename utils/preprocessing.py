from preprocess_and_encode import Preprocess, save_tf_dataset
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import argparse
import pickle
import os 


def main(data_dir="", out_dir=""):

    """
        This function will open raw txt files and convert the sentances and chars.
        Thew TextVectorisation and one hot-encoding will happen in data_loader 
        You shall pass the preprocessing.py as --data_dir and --out_dir
    """

    dev = ["train", "valid", "test"]


    for data_type in dev:
        #Get the lines from the file and convert them to the data frame
        print(f"Processing {data_type} data...")
        file_path = os.path.join(data_dir, f"{data_type}.txt")
        preproc = Preprocess(file_path, data_type)
        lines = preproc.get_lines()
        df = preproc.load_and_convert(lines)
 
        
        #Create sentaces, chars, labels for tfds files 
        sentances = df.text.tolist()
        chars = [preproc.convert_sent_to_chars(sentance) for sentance in sentances]
       

        print("Saving to tfrecord format")
        save_tf_dataset(df.line_nb, df.total_lines, sentances, chars, df.target, data_type, out_dir)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Preprocess and encode the data and also make tfds files")
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--out_dir", type=str, default='data_tfds')
 
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir)
 


