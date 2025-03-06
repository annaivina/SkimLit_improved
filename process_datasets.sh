#!/bin/bash

# preprocess the txt files
python3 utils/preprocessing.py --data_dir=data/pubmed-rct/PubMed_200k_RCT_numbers_replaced_with_at_sign --out_dir=data/raw_files

#Making the tfds format data for trianing
cd skimlit_dataset #The folder where you defined you dataset class 
tfds build --data_dir=/data/tfds_datasets --manual_dir=data/raw_files/