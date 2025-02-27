from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

def load_dataset(name):
    dataset = TUDataset(root=f'/tmp/{name}', name=name)
    print(f'Dataset: {name}, Number of graphs: {len(dataset)}')

    # Filter out invalid graphs (graphs with no nodes)
    valid_graphs = [data for data in dataset if data.num_nodes > 0]
    print(f"Number of valid graphs after filtering: {len(valid_graphs)}")
    
    # Update the dataset with valid graphs
    dataset = valid_graphs

    # Use node degree as scalar feature for all graphs
    for data in dataset:
        if data.x is None:
            deg = degree(data.edge_index[0], dtype=torch.float)
            data.x = deg.view(-1, 1)  # Keep scalar degree as feature

    # One-hot encode the labels
    num_classes = len(torch.unique(torch.tensor([data.y.item() for data in dataset])))
    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]

    # Check if all labels are one-hot encoded
    for i, data in enumerate(dataset):

        assert data.y.shape[0] == num_classes, f"Graph {i} label size mismatch: {data.y.shape}"
        assert (data.y.sum().item() == 1), f"Graph {i} label is not one-hot: {data.y}"

    print("All labels are one-hot encoded correctly.")


    # Record the first occurrence of each class label in the dataset
    first_occurrence = defaultdict(int)
    printed_labels = set()

    print("Labels of the first graph for each class:")

    for i, data in enumerate(dataset):
        # Get the label of the graph
        label = torch.argmax(data.y).item()  # Convert one-hot label to class index

        # If this class has not been printed yet, record and print it
        if label not in printed_labels:
            first_occurrence[label] = i
            print(f"Class {label} first appears in graph {i}, label: {data.y}")
            printed_labels.add(label)

    # Print the indices of the first graph for each class
    print("\nIndices of the first graph for each class:", dict(first_occurrence))


    # Check if one-hot to index conversion matches original labels
    original_labels = [data.y.item() if data.y.dim() == 0 else data.y.argmax().item() for data in dataset]  # Ensure labels are scalars
    for i, data in enumerate(dataset):
        one_hot_label = F.one_hot(torch.tensor(original_labels[i]), num_classes=num_classes).float()
        recovered_label = one_hot_label.argmax().item()
        assert recovered_label == original_labels[i], f"Mismatch in label for graph {i}: {recovered_label} vs {original_labels[i]}"
    
    print("All labels passed the consistency check!")

    return dataset


# Function for k-fold split using stratified splitting
def k_fold_split(dataset, n_splits=10, seed=0):
    # Extract labels for stratified splitting
    labels = [torch.argmax(data.y).item() for data in dataset]  # Use one-hot encoded labels
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


def split_train_val(train_data, val_size=0.1, seed=None):
    seed = np.random.randint(0, 10000) if seed is None else seed
    train_indices, val_indices = train_test_split(np.arange(len(train_data)), test_size=val_size, random_state=seed)

    # Use list comprehensions to extract train and validation splits
    train_split = [train_data[i] for i in train_indices]
    val_split = [train_data[i] for i in val_indices]
    
    return train_split, val_split

