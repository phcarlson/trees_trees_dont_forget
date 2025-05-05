from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import os 

PROTEIN_CATEGORIES = list('ACDEFGHIKLMNPQRSTVWYU')
RANDOM_SEED = 5525

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

# def one_hot_encode_and_pad(sequence, max_len = 1000):

#     # Init the OneHotEncoder
#     encoder = OneHotEncoder(categories=[PROTEIN_CATEGORIES], dtype=int)
#     # encoder.fit(np.array(PROTEIN_CATEGORIES).reshape(-1, 1))

#     # Raw one hot encoding, which should be of size sequence len x num categories (20)
#     encoded =  encoder.transform(sequence).toarray()
#     padded_or_clipped = encoded

#     encoded = np.zeros((max_len, len(PROTEIN_CATEGORIES)), dtype=int)

def one_hot_encode_sequence(sequence, max_len=1000):
    """
    One-hot encodes the sequence, assuming the sequence contains only characters from the alphabet.
    Pads to max_len if the sequence is shorter.
    """
    # Create a mapping from character to index
    char_positions = {}
    for i, char in enumerate(PROTEIN_CATEGORIES):
        char_positions[char] = i

    # Set up the cols of 0s, which should be of size num categories (20) x  desired/standardized input length
    encoded = np.zeros((len(PROTEIN_CATEGORIES), max_len), dtype=int)
    for i, char in enumerate(sequence[:max_len]):
        # Gets the col and fills 1 in the appropriate element corresponding to that char based on the map
        encoded[char_positions[char], i] = 1

    return encoded

class BacArchDataset(Dataset):
    ''' With inspo from my CS 389 course for custom dataset to work with the dataloader'''
    def __init__(self):
        # First we load the raw df from HF
        raw_dataframe = load_dataset('BacArch')

        # Now to add the negative samples
        self.dataframe = append_negative_samples(raw_dataframe)
        # self.dataframe = raw_dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        seq1 = self.dataframe.iloc[idx]['Seq1']
        seq2 = self.dataframe.iloc[idx]['Seq2']
        label = self.dataframe.iloc[idx]['Label']
        seq1_encoded = one_hot_encode_sequence(seq1)
        seq2_encoded = one_hot_encode_sequence(seq2)
        return torch.tensor(seq1_encoded, dtype=torch.float32), torch.tensor(seq2_encoded, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def main():
    pass

if __name__ == "__main__":
    main()