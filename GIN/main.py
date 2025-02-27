import os
import torch
import argparse
import json
import optuna
from data_utils import load_dataset, k_fold_split
from model import Net
from train import train_with_early_stopping
import numpy as np
import torch.multiprocessing as mp

def preprocess_and_cache_splits(dataset, dataset_name, n_folds=10, seed=0):
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.getcwd(), "cache_new")
    os.makedirs(cache_dir, exist_ok=True)

    # Use the existing k_fold_split function from data_utils
    fold_splits = k_fold_split(dataset, n_splits=n_folds, seed=seed)

    # Preprocess and cache each fold split
    cached_folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        cache_file = os.path.join(cache_dir, f'cached_fold_{dataset_name}_{fold_idx}.pt')

        if os.path.exists(cache_file):
            print(f"Loading existing cache for fold {fold_idx} of dataset {dataset_name}")
        else:
            print(f"Creating cache for fold {fold_idx} of dataset {dataset_name}")
            train_data = [dataset[i] for i in train_idx]
            test_data = [dataset[i] for i in test_idx]
            torch.save({'train': train_data, 'test': test_data}, cache_file)

        cached_folds.append(cache_file)

    return cached_folds


def objective(trial, dataset, device, args):
    # Define the hyperparameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_channels = trial.suggest_int("hidden_channels", 16, 128, step=16)

    # Initialize the model
    num_node_features = dataset[0].x.size(1)
    num_classes = dataset[0].y.size(0)
    model = Net(
    num_layers=3,                # Total GIN layers
    num_mlp_layers=2,            # Layers in each MLP
    input_dim=num_node_features,
    hidden_dim=hidden_channels,               # Hidden dimension size
    output_dim=num_classes,
    final_dropout=0.5,           # Dropout probability
    learn_eps=True               # Enable learnable epsilon
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use the first fold for optimization
    train_idx, test_idx = k_fold_split(dataset, n_splits=10, seed=0)[0]
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]

    # Train and validate
    mean_score, _ = train_with_early_stopping(
        train_data, test_data, model, optimizer, device, 
        mixup=args.mixup, alpha=args.alpha, patience=500, 
        max_epochs=200, batch_size=batch_size
    )

    return mean_score  # Maximize validation accuracy


def run_single_fold(fold_idx, cached_file, device, args, best_params):
    print(f'Loading cached data for fold {fold_idx + 1}')
    data = torch.load(cached_file)
    train_data, test_data = data['train'], data['test']

    # Extract num_node_features and num_classes from train_data
    num_node_features = train_data[0].x.size(1)
    num_classes = train_data[0].y.size(0)

    model = Net(
    num_layers=3,                # Total GIN layers
    num_mlp_layers=2,            # Layers in each MLP
    input_dim=num_node_features,
    hidden_dim=best_params['hidden_channels'],               # Hidden dimension size
    output_dim=num_classes,
    final_dropout=0.5,           # Dropout probability
    learn_eps=True               # Enable learnable epsilon
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    # Pass fold_idx to train_with_early_stopping
    mean_score, std_score = train_with_early_stopping(
        train_data, test_data, model, optimizer, device, args.mixup, args.alpha, 
        fold_idx=fold_idx, batch_size=best_params['batch_size']
    )
    return mean_score, std_score



if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Graph classification with Mixup')
    parser.add_argument('--mixup', action='store_true', default=False, help='Whether to use Mixup')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        choices=['MUTAG', 'DD', 'NCI1', 'PROTEINS', 'COLLAB', 'IMDB-MULTI', 'REDDIT-MULTI-5K'],
                        help='Name of the dataset (default: MUTAG)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for Mixup (default: 1.0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Step 1: Load dataset
    dataset = load_dataset(args.dataset)

    # Step 2: Preprocess and cache folds (Make sure folds exist before hyperparameter search)
    cached_folds = preprocess_and_cache_splits(dataset, args.dataset, n_folds=10)

    # Step 3: Check for existing best hyperparameters
    hyper_dir = os.path.join(os.getcwd(), "hyper_gin")
    os.makedirs(hyper_dir, exist_ok=True)
    best_params_file = os.path.join(hyper_dir, f"best_hyperparameters_{args.dataset}.json")

    if os.path.exists(best_params_file):
        print(f"Loading best hyperparameters from {best_params_file}")
        with open(best_params_file, "r") as f:
            best_params = json.load(f)
    else:
        # Use Optuna to find best hyperparameters
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, dataset, device, args), n_trials=30)

        best_params = study.best_params
        print("Best hyperparameters:", best_params)

        # Save the best parameters to the hyper directory
        with open(best_params_file, "w") as f:
            print(f"Saving best hyperparameters to {best_params_file}")
            json.dump(best_params, f)

    # Step 4: Run the best hyperparameter setting on fold 0 first
    print("\n>>> Running best hyperparameters on Fold 0 for evaluation <<<")
    fold_0_result = run_single_fold(0, cached_folds[0], device, args, best_params)
    print(f"Fold 0 result: {fold_0_result[0]:.4f} ± {fold_0_result[1]:.4f}\n")

    # Step 5: Run remaining folds in batches of 3
    print("\n>>> Running remaining folds in batches of 3 <<<")
    fold_results = [fold_0_result]  # Initialize results with fold 0

    for i in range(1, 10, 5):  # Iterate in steps of 3
        pool = mp.Pool(processes=5)  # Only run 3 folds at a time

        folds_to_run = [
            (fold_idx, cached_folds[fold_idx], device, args, best_params)
            for fold_idx in range(i, min(i + 5, 10))  # Ensure we don't go beyond fold 9
        ]

        results = [pool.apply_async(run_single_fold, args=fold) for fold in folds_to_run]
        pool.close()
        pool.join()

        fold_results.extend([res.get() for res in results])

        # Print results for the current batch
        for fold_idx, (mean, std) in zip(range(i, min(i + 5, 10)), fold_results[-len(folds_to_run):]):
            print(f'Fold {fold_idx}: {mean:.4f} ± {std:.4f}')

    # Final results (including fold 0)
    fold_means = [result[0] for result in fold_results]
    fold_stds = [result[1] for result in fold_results]

    final_mean = np.mean(fold_means)
    final_std = np.std(fold_means)
    print(f'\nFinal average test accuracy (including fold 0): {final_mean:.4f} ± {final_std:.4f}')

