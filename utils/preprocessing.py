from src.preprocess_and_encode import Preprocess, save_tf_dataset
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import argparse
import pickle
import os 


def main(data_dir="", out_dir=""):

    """
        This function will open raw txt files and convert the sentances, chars and lables to the right inputs.
        It will also create final tfds files to be used as inputs to the model training.
        You shall pass the preprocessing.py as --data_dir and --out_dir
    """

    dev = ["train", "valid", "test"]

    char_vectorizer = TextVectorization(max_tokens=96,#Alphabet count + 2
                                    output_sequence_length = 338,#found out after checking len 
                                    standardize=None,
                                    pad_to_max_tokens=True)
    



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
       
 
        #Save the lables encoded in label_encode for the prediciotn checks:
        # if data_type == "test":
        #     print("Saving label encodings for prediction tasks...")
        #     label_path = os.path.join(out_dir, "label_encoded_test.pkl")
        #     number_label = preproc.encode_target(df, encoder="label")
        #     with open(label_path, "wb") as f:
        #         pickle.dump(number_label,f)
        #         print("Saved the test labels")
        

        # #Vectorise chars with text vectorisation
        # vocab_path = os.path.join(out_dir, "char_vectorizer_vocab.pkl")
        # print("Vectorising chars")
        # if data_type == 'train':
        #     char_vectorizer.adapt(chars)
        #     #TODO - store the vectorised versions of label for future prediciotns 
            
        #     # Save vocabulary to use for test and validation
        #     with open(vocab_path, "wb") as f:
        #         pickle.dump(char_vectorizer.get_vocabulary(), f)
        #         print(f"Saved character vocabulary to {vocab_path} for valid and test samples")
        # else:
        #      print(f"Loading saved vocabulary for {data_type}")
        #      with open(vocab_path, "rb") as f:
        #          vocab = pickle.load(f)
        #          char_vectorizer.set_vocabulary(vocab)


        # vectorised_chars = char_vectorizer(chars).numpy()

        print("Saving to tfrecord format")
        save_tf_dataset(df.line_nb, df.total_lines, sentances, chars, df.target, data_type, out_dir)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Preprocess and encode the data and also make tfds files")
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--out_dir", type=str, default='data_tfds')
 
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir)
 


