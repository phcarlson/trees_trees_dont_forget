## Reproducability

# Random Forests

- Refer to src/random_forest_k_fold.py
- The seeds are already fixed
- First, it will try to load the one-hot encoded data as though you have saved it
- If not found, it will recreate the seq1 and seq2 encodings and save it for future runs
- Pass your desired hyperparams into train_random_forest function in main and set the number of folds you want to run
    - If you don't want to run all folds, add a break after the first iteration
- Plots will be produced, metrics will be saved to csvs

# Basic NN

- Refer to src/nn_split_on_pos_pairs.py
- The seeds are already fixed
- First, it will try to load the one-hot encoded data as though you have saved it
- If not found, it will recreate the seq1 and seq2 encodings and save it for future runs
- Pass your desired batch size, lr, num epochs in train_model in main
- For the punishing false negatives, you can swap the criterion between regular BCE and the one with pos_weight 
- Additionally, for  repeating positives, you can add X_pos_repeated, y_pos_repeated to the concatenated final set with a repeat_factor of 3
- Plots will be produced, metrics will be saved to csvs
