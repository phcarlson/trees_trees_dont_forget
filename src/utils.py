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


def load_negative_samples(dataframe, output_file="bac_arch_neg.parquet"):
    if os.path.exists(output_file):
        neg_df = pd.read_parquet(output_file)
        return neg_df
    else:
        print("Bach arch negative samples do not already exist, so will be created")
    
    negative_samples = []
    
    # Create negative samples by pairing each Seq1 with all other Seq2, and swapping them too for order invariance
    for i in range(len(dataframe)):
        seq1 = dataframe.iloc[i]['Seq1']
        for j in range(len(dataframe)):
            # except itself
            if i != j:
                seq2 = dataframe.iloc[j]['Seq2']
                # We know these are not similar as there is only 1 true match in the dataset
                negative_samples.append([seq1, seq2, 0])
                negative_samples.append([seq2, seq1, 0])

    
    # Combine the negative samples with the original dataset's pos ones
    # Per https://www.geeksforgeeks.org/make-a-pandas-dataframe-with-two-dimensional-list-python/
    new_neg_df = pd.DataFrame(negative_samples, columns=['Seq1', 'Seq2', 'Label']).reset_index(drop=True)
    # merged_samples = pd.concat([dataframe, new_neg_df])
    
    # Read that a parquet better preserves the datatypes for easier loading
    new_neg_df.to_parquet(output_file)
    print(f"Negative samples saved to '{output_file}'")
    
    return new_neg_df


def load_swapped_positive_samples(dataframe, output_file="bac_arch_swapped_pos.parquet"):
    if os.path.exists(output_file):
        pos_df = pd.read_parquet(output_file)
        return pos_df
    else:
        print("Bach arch swapped pos samples do not already exist, so will be created")
    
    positive_samples = []

    # Just swap the order
    for i in range(len(dataframe)):
        seq1 = dataframe.iloc[i]['Seq1']
        seq2 = dataframe.iloc[i]['Seq2']
        positive_samples.append([seq2, seq1, 1])
        
    new_pos_df = pd.DataFrame(positive_samples, columns=['Seq1', 'Seq2', 'Label']).reset_index(drop=True)
    # merged_samples = pd.concat([dataframe, new_pos_df])
    
    # Adjust ordering
    # Per https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/ then reset the index since that gets shuffled too I think?
    
    # Read that a parquet better preserves the datatypes for easier loading
    new_pos_df.to_parquet(output_file)
    print(f"Pos samples saved to '{output_file}'")
    
    return new_pos_df

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
    except Exception as e:
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
    except Exception as e:
        print(f"{file_name} not found. :(")
        return None
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
# class BacArchDataset(Dataset):
#     ''' With inspo from my CS 389 course for custom dataset to work with the dataloader'''
#     def __init__(self, df):
       
#         self.dataframe = df
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
from scipy.sparse import csr_matrix

class BacArchDataset(Dataset):
    ''' Custom Dataset for BacArch data with sparse matrix support '''
    def __init__(self, X_sparse, y):
        """
        X_sparse: A sparse matrix of shape (n_samples, n_features)
        y: Labels corresponding to the sequences
        """
        self.X_sparse = X_sparse
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Get the sparse row for the current sample
        seq_features = self.X_sparse[idx].toarray().flatten()  # Convert sparse to dense (flattened)
        label = self.y[idx]
        
        # Convert to torch tensors
        features_tensor = torch.tensor(seq_features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return features_tensor, label_tensor
    
def main():
    pass

if __name__ == "__main__":
    main()


    

def eval_set(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq1, seq2, label in data_loader:
            # Move data to the same device as the model
            seq1, seq2, label = seq1.to(device), seq2.to(device), label.to(device)

            # Concatenate the sequences and flatten them
            input_tensor = torch.cat((seq1, seq2), dim=1)
            input_tensor = input_tensor.view(seq1.size(0), -1)

            # Forward pass
            outputs = model(input_tensor).view(-1)

            # Make predictions: threshold at 0.5 for binary classification
            preds = (outputs >= 0.5).float()

            # Store predictions and labels on the CPU for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Compute metrics
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    plot_confusion(all_preds, all_labels)

def plot_confusion(preds, labels):
    # matrix = confusion_matrix(labels, preds)

    # # Plot the confusion matrix using seaborn
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    # plt.show()

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
