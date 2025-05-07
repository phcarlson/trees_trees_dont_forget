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
from sklearn.utils import shuffle

# Check if CUDA is available to attach, as using GPU will significantly speed process up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

# Super basic pytorch module, just a linear layer and sigmoid for nonlinearity
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),       
        )

        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )
        
    def forward(self, x):
        return self.model(x)

def train_model(X_train, y_train, input_dim, batch_size=64, epochs=2):
    # Attaches the model to the GPU instead of CPU speed it up
    model = SimpleNN(input_dim).to(device)

    # Adam is cool beans bc of momentum and stuff
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # num_pos = (y_train == 1).sum()
    # num_neg = (y_train == 0).sum()
    # pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MarginRankingLoss(margin=1.0)

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

def compute_top1_f1_score(model, seq1_encoded, seq2_encoded, merged_df, val_idx, device="cuda"):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i in tqdm(val_idx):
            seq1 = seq1_encoded[i]  # Get the seq1 encoding for the current index

            seq_2s_to_score_with_seq_1 = []
            true_labels = []

            scores_against_seq1 = []
            # Pair seq1 with all other seq2 (skip the same pair for the true match)
            for j in val_idx:
                seq2 = seq2_encoded[j]
                
                label = 1 if i == j else 0  # True match for i == j, else negative
                true_labels.append(label)

                # Concatenate seq1 and seq2 as in training phase
                pair_input = hstack([seq1, seq2])  # Combine seq1 and seq2 for each pair
                seq_2s_to_score_with_seq_1.append(pair_input)


            dataset = BacArchDataset(seq_2s_to_score_with_seq_1, true_labels)
            var_loader = DataLoader(dataset, batch_size=128, shuffle=False)

            for input, label in var_loader:
                input, label = input.to(device), label.to(device)
                
                scores = model(input)
                scores_against_seq1.extend(scores.cpu().numpy())

            # Find top-1 prediction (index with the highest output value)
            top1_idx = np.argmax(scores_against_seq1)
            predicted_label = 1 if true_labels[top1_idx] == 1 else 0

            # Append true label (always 1, because seq1 has exactly one match)
            y_true.append(1)
            y_pred.append(predicted_label)

    # Compute precision, recall, and F1 score based on the true vs predicted labels
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Top-1 Precision: {precision:.4f}")
    print(f"Top-1 Recall: {recall:.4f}")
    print(f"Top-1 F1 Score: {f1:.4f}")
    plot_confusion(y_pred, y_true )
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("LOADING BACARCH DATASET, ADDING NEGATIVE SAMPLES, AND SHUFFLING....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    raw_dataframe = load_dataset('BacArch')
    raw_dataframe["pair_id"] = range(len(raw_dataframe))
    raw_dataframe["source"] = "original"

    merged_df = raw_dataframe.sample(frac=1, random_state=42)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("PADDING AND ENCODING SEQ 1 and SEQ 2 COLS....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    encoded_filename = 'encoded_sequences.pkl'
    loaded_encodings = None
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
    X = hstack([seq1_encoded, seq2_encoded]) 
    y = np.array(merged_df["Label"])

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("TRAINING AND EVALUATING NN WITH K-FOLD CROSS-VALIDATION....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    k_folds = 5
    fold_metrics = []
    fold_losses = []

    k_fold = StratifiedKFold(n_splits = k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
        print(f"\nTraining fold {fold+1}/{k_folds}...")

        # Assign training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        flipped_pos_samples = []
        for i in train_idx:
            seq1 = seq1_encoded[i]
            seq2 = seq2_encoded[i]
            flipped_pos_samples.append(hstack([seq2, seq1]))

        # Create negative samples for the train set by pairing each Seq1 with all other Seq2
        neg_samples = []
        for i in train_idx:
            seq1 = seq1_encoded[i]
            for j in train_idx:
                # except itself
                if i != j:
                    seq2 = seq2_encoded[j]
                    # We know these are not similar as there is only 1 true match in the dataset
                    neg_samples.append(hstack([seq1, seq2]))

        X_neg = vstack(neg_samples)
        y_neg = np.zeros(X_neg.shape[0])
        # X_neg, y_neg = shuffle(X_neg, y_neg, random_state=42)
        # X_neg = X_neg[:250]
        # y_neg = y_neg[:250]

        X_pos_flipped = vstack(flipped_pos_samples)
        y_pos_flipped = np.ones(X_pos_flipped.shape[0])

        repeat_factor = 3
        X_pos_repeated = vstack([X_train[y_train == 1]] * repeat_factor)
        y_pos_repeated = np.ones(X_pos_repeated.shape[0])



        X_train = vstack([X_train, X_neg, X_pos_flipped, X_pos_repeated])
        y_train = np.concatenate([y_train, y_neg, y_pos_flipped, y_pos_repeated])

        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        model, losses_across_batches = train_model(X_train, y_train, X_train.shape[1], batch_size=128, epochs=5)
        fold_losses.append(losses_across_batches)


        metrics = compute_top1_f1_score(model, seq1_encoded, seq2_encoded, merged_df, val_idx, device="cuda")
        fold_metrics.append(metrics)
        
        plt.figure(figsize=(12, 6))
        plt.plot(losses_across_batches, label=f"Fold {fold+1}")

        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Batch for Each Fold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"fold_{fold+1}_losses_plot_{datetime.now()}.png")
        plt.show()
        break 

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
    avg_precision = np.mean([metrics['precision'] for metrics in fold_metrics])
    avg_recall = np.mean([metrics['recall'] for metrics in fold_metrics])
    avg_f1 = np.mean([metrics['f1'] for metrics in fold_metrics])

    print(f"\nAverage Metrics across {k_folds} folds:")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    # Convert the fold metrics to a DataFrame
    metrics_df = pd.DataFrame(fold_metrics)
    date = datetime.now()
    # Save the fold metrics to a CSV file
    metrics_df.to_csv(f'nn_fold_metrics_{date}.csv', index=False)

if __name__ == "__main__":
    main()