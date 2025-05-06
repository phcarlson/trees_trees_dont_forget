
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
from scipy.sparse import hstack, vstack
import numpy as np

from utils import append_negative_samples, plot_confusion, load_dataset, load_encoded_data, save_encoded_data, load_model, save_model, one_hot_encode_batch

def train_random_forest(X_train, y_train, n_estimators = 10, max_depth = 30):
    # Following the seed tradition here?? lol https://www.reddit.com/r/datascience/comments/17kxd5s/data_folks_of_reddit_how_do_you_choose_a_random/
    # When n jobs is -1, utilizes all cores to train ASAP
    random_forest_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=30, n_jobs=-1)
    random_forest_model.fit(X_train, y_train)
    return random_forest_model

def compute_metrics(random_forest_model, all_seq1_encoded, all_seq2_encoded, y_test, k_values=[1, 50]):
    """ Just like in the DGEB github, they only perform the metrics on the OG dataset. """

    recall_at_k_values = []
    precision_values = []
    f1_values = []

    # So lets get the matches where they are postive
    pos_indices = np.where(y_test == 1)[0] 
    all_seq2_pos = all_seq2_encoded[pos_indices]

    for top_k in k_values:
        true_labels = []
        predicted_labels = []

        # Across the OG dataset's seq 1
        for pos_index_value in pos_indices:
            s1 = all_seq1_encoded[pos_index_value]

            # Creates a batch where first we duplicate the s1 to find a match for as many times as there are pairs to "query against"
            s1_repeated = vstack([s1] * len(pos_indices))
            # Once we have this col of the s1 encoding, we merge with the s2 encoding col to get a new matrix of encoded pairs as inputs
            pair_inputs = hstack([s1_repeated, all_seq2_pos]) 
            # Get scores for the batch
            scores = random_forest_model.predict(pair_inputs)

            # Get top k indices to check whether the label 1 is inside (first get indices that sort the array, reverse to get descending indices, then take top k)
            top_k_indices = np.argsort(scores)[::-1][:top_k]

            # Now that we have the indices of top scores, 
            # # First get the actual index of the current pos score position in pos scores
            index_of_true_match = np.where(pos_indices == pos_index_value)[0]
            # Is the actual index of the curr true pair in the top k indices picked?
            is_hit = len(index_of_true_match) > 0 and index_of_true_match[0] in top_k_indices

            true_labels.append(1)
            predicted_labels.append(1 if is_hit else 0)

        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)

        print(f"Recall@{top_k}: {recall:.4f}")
        print(f"Precision@{top_k}: {precision:.4f}")
        print(f"F1 Score@{top_k}: {f1:.4f}")

        plot_confusion(predicted_labels, true_labels)

        recall_at_k_values.append(recall)
        precision_values.append(precision)
        f1_values.append(f1)

    return recall_at_k_values, precision_values, f1_values

def main():
    # To tweak what I want saved or rerun
    retrain_model_anyway = False
    encode_again_anyway = False

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("LOADING BACARCH DATASET, ADDING NEGATIVE SAMPLES, AND SHUFFLING....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    raw_dataframe = load_dataset('BacArch')
    dataframe = append_negative_samples(raw_dataframe)
    shuffled = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("PADDING AND ENCODING SEQ 1 and SEQ 2 COLS....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    encoded_filename = 'encoded_sequences.pkl'
    loaded_encodings = load_encoded_data(encoded_filename)

    # If data is not found, re-run encoding
    if loaded_encodings is None or encode_again_anyway:
        seq1_df = shuffled[['Seq1']].copy()
        seq1_df.columns = ['MatchedSeqs']
        seq2_df = shuffled[['Seq2']].copy()
        seq2_df.columns = ['MatchedSeqs']
        
        seq1_encoded = one_hot_encode_batch(seq1_df, 1000)
        seq2_encoded = one_hot_encode_batch(seq2_df, 1000)

        save_encoded_data(seq1_encoded, seq2_encoded, 'encoded_sequences.pkl')
    else:
        seq1_encoded, seq2_encoded = loaded_encodings

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("CONCATENATING COLS FOR INPUTS....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")   
    # Take both encoded sequences in the pairing (not necessarily label = 1 due to neg samples) and concat together 
    X = hstack([seq1_encoded, seq2_encoded]) 
    y = np.array(shuffled["Label"])
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("TRAINING RANDOM FOREST....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    model_filename = 'random_forest_model.pkl'
    rf_model = load_model(model_filename)

    # If model is not found, retrain and save it
    if rf_model is None or retrain_model_anyway:
        print("Must retrain!")
        rf_model = train_random_forest(X, y)
        save_model(rf_model, model_filename)
    else:
        print("Pretrained model loaded")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("EVALUATING RANDOM FOREST....")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
    compute_metrics(rf_model=rf_model, all_seq1_encoded=seq1_encoded, all_seq2_encoded=seq2_encoded, y_test=y, k_values=[1, 50])

if __name__ == "__main__":
    main()
