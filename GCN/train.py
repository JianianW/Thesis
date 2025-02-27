import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import copy
from tqdm import tqdm
from data_utils import split_train_val
import os
import matplotlib.pyplot as plt


def mixup_graph_embeddings(data, model, alpha):
    """
    Generate mixed graph embeddings using the Mixup method.
    """
    device = next(model.parameters()).device  # Get the device of the model
    data = data.to(device)  # Move data to the same device as the model

    # Obtain graph-level embeddings from the GNN model, without MLP layers
    h1 = model.get_graph_embedding(data.x, data.edge_index, data.batch)

    # Randomly shuffle h1 to generate h2
    indices = torch.randperm(h1.size(0), device=device)
    h2 = h1[indices]

    # Generate lambda from a Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Mix embeddings and return the associated label pairs
    mixed_h = lam * h1 + (1 - lam) * h2
    y_a = data.y
    y_b = data.y[indices]

    return mixed_h, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the Mixup loss for the mixed embeddings.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def early_stopping(val_loss, patience=500, min_delta=0):
    """
    Early stopping logic to monitor validation loss.
    """
    if len(val_loss) > patience:
        if all(val_loss[-i - 1] - val_loss[-i] <= min_delta for i in range(patience)):
            return True
    return False


def train_model(train_data, val_data, model, optimizer, device, mixup=False, alpha=1.0, patience=500, max_epochs=1000, batch_size=32):
    """
    Train the model with optional Mixup and early stopping.
    """
    criterion = torch.nn.CrossEntropyLoss()  # Use Cross-Entropy Loss
    best_val_loss = float('inf')
    best_model_state = None
    val_loss_history = []
    train_loss_history = []

    # Load all data into GPU once
    train_data = [data.to(device) for data in train_data]
    val_data = [data.to(device) for data in val_data]

    # Set up DataLoader with the fixed batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    num_classes = model.mlp_output.out_features
    # Begin training with early stopping
    for epoch in range(max_epochs):
        model.train()

        total_loss = 0

        for data in train_loader:
            data.y = data.y.view(-1, num_classes)
            optimizer.zero_grad()

            if mixup:
                mixed_h, y_a, y_b, lam = mixup_graph_embeddings(data, model, alpha)
                mixed_h = model.mlp_forward(mixed_h)
                loss = mixup_criterion(criterion, mixed_h, y_a, y_b, lam)
            else:
                target = data.y
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, target)


            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Record training loss
        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation loss calculation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for val_data_batch in val_loader:

                val_data_batch.y = val_data_batch.y.view(-1, num_classes)
                target = val_data_batch.y
                val_out = model(val_data_batch.x, val_data_batch.edge_index, val_data_batch.batch)
                val_loss += criterion(val_out, target).item()

        val_loss_history.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # Early stopping logic
        if early_stopping(val_loss_history, patience=patience):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_model_state, train_loss_history, val_loss_history


def test_model(test_data, model, device):
    """
    Test the model on the test dataset and calculate accuracy.
    """
    correct = 0
    total = 0
    model.eval()

    test_loader = DataLoader(test_data, batch_size=32, num_workers=0)
    num_classes = model.mlp_output.out_features
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.y = data.y.view(-1, num_classes)
            target = data.y.argmax(dim=-1)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=-1)  # Predict class indices
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    return accuracy


def train_with_early_stopping(train_data, test_data, model, optimizer, device, mixup=False, alpha=1.0, patience=500, max_epochs=1000, batch_size=32, fold_idx=0):
    """
    Perform 50 independent trials per fold and save loss curves for specific trials into the 'loss' folder.
    """
    fold_scores = []  # Store results of 50 independent trials

    # Ensure the 'loss' directory exists
    loss_dir = os.path.join(os.getcwd(), "loss")
    os.makedirs(loss_dir, exist_ok=True)

    for trial in tqdm(range(50), desc="Independent Trials"):
        print(f"Starting trial {trial + 1}")
        model.reset_parameters()

        # Dynamically split train_data into train and val for this trial
        dynamic_seed = np.random.randint(0, 10000)
        train_split, val_split = split_train_val(train_data, val_size=0.1, seed=dynamic_seed)

        # Train the model
        best_model_state, train_loss_history, val_loss_history = train_model(
            train_split, val_split, model, optimizer, device, mixup, alpha, patience, max_epochs, batch_size
        )

        # Save loss curves for trials 1, 25, and 50
        if trial + 1 in [1, 25, 50]:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(train_loss_history)), train_loss_history, label='Train Loss', marker='o')
            plt.plot(range(len(val_loss_history)), val_loss_history, label='Validation Loss', marker='x')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve: Fold {fold_idx + 1}, Trial {trial + 1}')
            plt.legend()
            plt.grid(True)

            # Save the plot in the 'loss' folder
            save_path = os.path.join(loss_dir, f"loss_curve_fold_{fold_idx + 1}_trial_{trial + 1}.png")
            plt.savefig(save_path)
            plt.close()

        # Load best model state and test
        model.load_state_dict(best_model_state)
        accuracy = test_model(test_data, model, device)
        fold_scores.append(accuracy)

    return np.mean(fold_scores), np.std(fold_scores)
