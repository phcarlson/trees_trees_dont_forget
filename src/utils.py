from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os 
import joblib

RANDOM_SEED = 5525

# The 20 amino acids (+ U ??)
PROTEIN_CATEGORIES = list('ACDEFGHIKLMNPQRSTVWYU')


def load_dataset(set_name):
    if set_name == "BacArch":
        bac_arch = pd.read_parquet("hf://datasets/tattabio/bac_arch_bigene/data/train-00000-of-00001.parquet")
        bac_arch['Label'] = 1 
        return bac_arch
    elif set_name == "ModAC":
        return None
    else:
        print(f"That ({set_name}) isn't a known dataset for the DGEB matching task!")
        return None


def append_negative_samples(dataframe, output_file="bac_arch_neg.parquet"):
    if os.path.exists(output_file):
        return pd.read_parquet(output_file)
    else:
        print("Bach arch negative samples do not already exist, so will be created")
    
    negative_samples = []
    
    # Create negative samples by pairing each Seq1 with all other Seq2
    for i in range(len(dataframe)):
        seq1 = dataframe.iloc[i]['Seq1']
        for j in range(len(dataframe)):
            # except itself
            if i != j:
                seq2 = dataframe.iloc[j]['Seq2']
                # We know these are not similar as there is only 1 true match in the dataset
                negative_samples.append([seq1, seq2, 0])
    
    # Flipping the order, create negative samples by pairing each Seq2 with all other Seq1
    for i in range(len(dataframe)):
        seq2 = dataframe.iloc[i]['Seq2']
        #  except itself
        for j in range(len(dataframe)):
            if i != j:
                seq1 = dataframe.iloc[j]['Seq1']
                negative_samples.append([seq1, seq2, 0])

    # Combine the negative samples with the original dataset's pos ones
    # Per https://www.geeksforgeeks.org/make-a-pandas-dataframe-with-two-dimensional-list-python/
    new_neg_df = pd.DataFrame(negative_samples, columns=['Seq1', 'Seq2', 'Label'])
    merged_samples = pd.concat([dataframe, new_neg_df])
    
    # Adjust ordering
    # Per https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/ then reset the index since that gets shuffled too I think?
    merged_samples = merged_samples.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Read that a parquet better preserves the datatypes for easier loading
    merged_samples.to_parquet(output_file)
    print(f"Negative samples saved to '{output_file}'")
    
    return merged_samples

def pad_sequences(df, max_len, char_for_padding='-'):
    """ Pad sequences to the same length with a specified padding character.
        Once converted to one-hot, where we ignore unknown chars, this will make the one-hot encoded padded Scols all 0! """

    # Per https://www.w3schools.com/python/ref_string_ljust.asp where you just left justify and pad any string, I first trim any too long, then pad any too short
    df["MatchedSeqs"] = df["MatchedSeqs"].apply(lambda matched_seq: matched_seq[:max_len].ljust(max_len, char_for_padding))
    return df


def one_hot_encode_batch(dataframe, max_len, padding_char='-'):
    """
    One-hot encodes a batch of sequence matches concatenated, padded, then encoded using scikit-learn's OneHotEncoder.
    """
    # First, merges the pairs and pad
    df_padded = pad_sequences(dataframe, max_len, padding_char)
    # print(df_padded.iloc[0]["MatchedSeqs"])

    # After padding, our matched seq super string is still... a string. We need it to be an array to put into the models. 
    # Thus this converts each string to a list first, and then we return a new col per char list to plop in each row
    char_cols = df_padded['MatchedSeqs'].apply(lambda matched_seq: pd.Series(list(matched_seq)))

    # Then we can label em as each position
    char_cols.columns = [f"Pos_{i}" for i in range(max_len)]
    # print(char_cols)

    # I have max len cols and each should use the protein categories for encoding
    # categories_per_position = [PROTEIN_CATEGORIES] * max_len
    encoder = OneHotEncoder(categories=[PROTEIN_CATEGORIES] * max_len, handle_unknown="ignore")
    encoder.fit(char_cols)
    df_encoded = encoder.transform(char_cols)  
    # Should be num datapoints * (21 * max seq len)
    # print(df_encoded)
    print(df_encoded.shape)

    return df_encoded

# Per https://stackoverflow.com/questions/20662023/save-python-random-forest-model-to-file
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_encoded_data(seq1_encoded, seq2_encoded, file_name):
    """ Save one-hot encoded data to file to avoid having to redo it every time """
    joblib.dump((seq1_encoded, seq2_encoded), file_name)
    print(f"Encoded data saved to {file_name}")

def load_encoded_data(file_name):
    """ Load one-hot encoded data from file to avoid having to redo it every time """
    try:
        seq1_encoded, seq2_encoded = joblib.load(file_name)
        print(f"One-hot data was found at {file_name}")
        return seq1_encoded, seq2_encoded
    except FileNotFoundError:
        print(f"{file_name} wasn't found.")
        return None
    

def save_model(model, file_name):
    """ Save the trained model using joblib. """
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")


def load_model(file_name):
    """ Load the model """
    try:
        model = joblib.load(file_name)
        print(f"Model loaded from {file_name}")
        return model
    except FileNotFoundError:
        print(f"{file_name} not found. :(")
        return None
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
# class BacArchDataset(Dataset):
#     ''' With inspo from my CS 389 course for custom dataset to work with the dataloader'''
#     def __init__(self):
#         # First we load the raw df from HF
#         raw_dataframe = load_dataset('BacArch')

#         # Now to add the negative samples
#         self.dataframe = append_negative_samples(raw_dataframe)
#         # self.dataframe = raw_dataframe

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         seq1 = self.dataframe.iloc[idx]['Seq1']
#         seq2 = self.dataframe.iloc[idx]['Seq2']
#         label = self.dataframe.iloc[idx]['Label']

#         # By flattening, each should turn into a 21 (because of U?) * max seq length long input
#         seq1_encoded_flattened = one_hot_encode_sequence(seq1).flatten()
#         seq2_encoded_flattened  = one_hot_encode_sequence(seq2).flatten()
        
#         features_for_one_match = np.concatenate([seq1_encoded_flattened, seq2_encoded_flattened])
        
#         # Once flattened concat together to make one big input of 2 * 21 * max seq length
#         return torch.tensor(features_for_one_match, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def main():
    pass

if __name__ == "__main__":
    main()