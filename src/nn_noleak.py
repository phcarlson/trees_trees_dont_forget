import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from utils import BacArchDataset, one_hot_encode_batch, plot_confusion, load_dataset, load_negative_samples, load_swapped_positive_samples, save_encoded_data, load_encoded_data
from datetime import datetime
import matplotlib.pyplot as plt

# Check if CUDA is available to attach, as using GPU will significantly speed process up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

# Super basic pytorch module, just a linear layer and sigmoid for nonlinearity
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)

def train_model(X_train, y_train, input_dim, batch_size=64, epochs=2):
    # Attaches the model to the GPU instead of CPU speed it up
    model = SimpleNN(input_dim).to(device)

    # Adam is cool beans bc of momentum and stuff
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCELoss()

    # Thank you CS 389
    dataset = BacArchDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses_across_batches = []
    for epoch in range(epochs):
        # Sets it to training mode
        model.train()
        epoch_loss = 0.0

        # For each batch, wrapping in tqdm which indicates progress bar
        for input, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            input, label = input.to(device), label.to(device)

            # Predict batch
            output = model(input)

            # Get avg cross entropy loss from batch, reshaping preds
            loss = criterion(output.view(-1), label.view(-1))

            # Wipes gradient from prev time, calcs it again fresh
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Track losses for this fold for plotting
            losses_across_batches.append(loss.item())

        # Give an intermediate update
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    return model, losses_across_batches

def compute_metrics(model, X_val, y_val, batch_size=128):
    # No longer training
    model.eval()

    # Take the val set from the k fold splits, make a data loader for easily pred batches over it
    val_dataset = BacArchDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    val_labels = []
    val_preds = []

    # No need to calculate the gradient
    with torch.no_grad():
        for input, label in tqdm(val_loader):
            # Makes sure our val set contents is attached to the GPU otherwise I think it'll error out
            input, label = input.to(device), label.to(device)

            # Get val preds
            output = model(input)

            # First we gotta move the val all back to the CPU
            # Make numpy array
            # Flatten the preds for the batch out into our larger arrray of preds and labels for the full val set
            val_labels.extend(label.cpu().numpy())
            val_preds.extend(output.cpu().numpy())

    # Then this makes binary decisions, may want to change to ranked for the other dataset
    val_preds_binary = [1 if p > 0.5 else 0 for p in val_preds]

    # Compute metrics for the positive class (pos_label=1)
    accuracy = accuracy_score(val_labels, val_preds_binary)
    precision = precision_score(val_labels, val_preds_binary, pos_label=1)
    recall = recall_score(val_labels, val_preds_binary, pos_label=1)
    f1 = f1_score(val_labels, val_preds_binary, pos_label=1)  # Only compute F1 for positive class

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    plot_confusion(val_labels, val_preds_binary)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("LOADING BACARCH DATASET, ADDING NEGATIVE SAMPLES, AND SHUFFLING....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    raw_dataframe = load_dataset('BacArch')
    raw_dataframe["pair_id"] = range(len(raw_dataframe))
    raw_dataframe["source"] = "original"

    neg_df_invariant = load_negative_samples(raw_dataframe)
    neg_df_invariant["pair_id"] = -1 
    neg_df_invariant["source"] = "negative"

    pos_df_invariant = load_swapped_positive_samples(raw_dataframe)
    pos_df_invariant["pair_id"] = raw_dataframe["pair_id"]
    pos_df_invariant["source"] = "swapped_positive"

    merged_df = pd.concat([raw_dataframe, neg_df_invariant])
    merged_df = merged_df.sample(frac=1, random_state=42)


    paired_df_no_swapped = merged_df[~merged_df["source"].isin(["swapped_positive"])]


    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("PADDING AND ENCODING SEQ 1 and SEQ 2 COLS....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    encoded_filename = 'encoded_sequences.pkl'
    loaded_encodings = load_encoded_data(encoded_filename)
    if loaded_encodings is None:
        seq1_df = merged_df[['Seq1']].copy()
        seq1_df.columns = ['MatchedSeqs']
        seq2_df = merged_df[['Seq2']].copy()
        seq2_df.columns = ['MatchedSeqs']

        seq1_encoded = one_hot_encode_batch(seq1_df, 1000)
        seq2_encoded = one_hot_encode_batch(seq2_df, 1000)

        save_encoded_data(seq1_encoded, seq2_encoded, encoded_filename)
    else:
        seq1_encoded, seq2_encoded = loaded_encodings

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("CONCATENATING COLS FOR INPUTS....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")   
    # X = hstack([seq1_encoded, seq2_encoded]) 
    # y = np.array(merged_df["Label"])
    X_no_swapped = hstack([seq1_encoded[paired_df_no_swapped.index], seq2_encoded[paired_df_no_swapped.index]])
    y_no_swapped = np.array(paired_df_no_swapped["Label"])

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("TRAINING AND EVALUATING NN WITH K-FOLD CROSS-VALIDATION....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    k_folds = 5
    fold_metrics = []
    fold_losses = []

    k_fold = StratifiedKFold(n_splits = k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X_no_swapped, y_no_swapped)):
        print(f"\nTraining fold {fold+1}/{k_folds}...")
      
        # Now merge the swapped positives based on their pair_id, making sure they are ONLY in the train or val split... NEVER BOTH for leakage
        swapped_pos_train = merged_df[(merged_df["Label"] == 1) & (merged_df["pair_id"].isin(paired_df_no_swapped.iloc[train_idx]["pair_id"]))]
        swapped_pos_val = merged_df[(merged_df["Label"] == 1) & (merged_df["pair_id"].isin(paired_df_no_swapped.iloc[val_idx]["pair_id"]))]
        
        # Assign training and validation sets
        X_train, X_val = X_no_swapped[train_idx], X_no_swapped[val_idx]
        y_train, y_val = y_no_swapped[train_idx], y_no_swapped[val_idx]

        X_train = vstack([X_train, hstack([seq1_encoded[swapped_pos_train.index], seq2_encoded[swapped_pos_train.index]])])
        y_train = np.concatenate([y_train, np.ones(len(swapped_pos_train))])

        X_val = vstack([X_val, hstack([seq1_encoded[swapped_pos_val.index], seq2_encoded[swapped_pos_val.index]])])
        y_val = np.concatenate([y_val, np.ones(len(swapped_pos_val))])

        model, losses_across_batches = train_model(X_train, y_train, X_train.shape[1], batch_size=64, epochs=10)
        fold_losses.append(losses_across_batches)

        metrics = compute_metrics(model, X_val, y_val)
        fold_metrics.append(metrics)

    plt.figure(figsize=(12, 6))
    for fold_idx, losses in enumerate(fold_losses):
        plt.plot(losses, label=f"Fold {fold_idx+1}")

    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Batch for Each Fold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"fold_losses_plot_{datetime.now()}.png")
    plt.show()

    # After cross-validation, calculate average metrics across all folds
    avg_accuracy = np.mean([metrics['accuracy'] for metrics in fold_metrics])
    avg_precision = np.mean([metrics['precision'] for metrics in fold_metrics])
    avg_recall = np.mean([metrics['recall'] for metrics in fold_metrics])
    avg_f1 = np.mean([metrics['f1'] for metrics in fold_metrics])

    print(f"\nAverage Metrics across {k_folds} folds:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    # Convert the fold metrics to a DataFrame
    metrics_df = pd.DataFrame(fold_metrics)
    date = datetime.now()
    # Save the fold metrics to a CSV file
    metrics_df.to_csv(f'logistic_fold_metrics_{date}.csv', index=False)
    print("Fold metrics saved to 'fold_metrics.csv'.")

if __name__ == "__main__":
    main()