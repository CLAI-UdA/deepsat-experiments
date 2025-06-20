"""
Contains functions for training and testing a PyTorch model.
"""
import copy  

import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

import pandas as pd
from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch

from logic_utils import CustomTokenizer


# --- Setup Device Agnostic-Code ---
device = "cuda" if torch.cuda.is_available else "cpu"

# --- Vocabulary size ---
def compute_vocab_size(tokenizer :CustomTokenizer):
    """
    Computes the total vocabulary size used by the tokenizer.

    The vocabulary includes:
        - All unique token indices assigned to formulas
        - Token indices for logical connectives 
        - Special tokens (e.g., padding)
        - The token assigned to falsity

    The returned size is equal to the maximum assigned index + 1 for
    padding int 0.

    Args:
        tokenizer (CustomTokenizer): A fitted tokenizer instance containing mappings
                                     from formula components to token indices.

    Returns:
        int: The vocabulary size (i.e., number of distinct token indices).
    """
    all_token_indices = (
        list(tokenizer.formula_to_token.values()) +
        list(tokenizer.connective_map.values()) +
        list(tokenizer.special_map.values()) +
        [tokenizer.falsity_token]
    )
    return max(all_token_indices) + 1


# --- Functions for training and testing a PyTorch model ---
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on.

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy).
    
  """
  # Put model in train mode
  model.train()
  
  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0
  
  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_logits = model(X) 
      y_preds = torch.round(torch.sigmoid(y_logits))

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_logits, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      train_acc += (y_preds == y).sum().item()/len(y_preds)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on.

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). 
    
  """
  # Put model in eval mode
  model.eval() 
  
  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0
  
  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)
  
          # 1. Forward pass
          test_pred_logits = model(X) 
          preds = torch.round(torch.sigmoid(test_pred_logits))

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
          
          # Calculate and accumulate accuracy
          test_pred_labels = preds
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on.

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 

  """
  # Create empty results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []
  }
  
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs), desc="Training Epochs"):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      
      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results

# --- Reproducibility ----
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    #torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# --- Function to save a model ---
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print("\n" + "=" * 80)
  print(f"[INFO] Saving model to: {model_save_path}")
  print("=" * 80 + "\n")

  torch.save(obj=model.state_dict(),
             f=model_save_path)

# --- Function to save Model's results ---
def save_results(results: Dict[str, List[float]], 
                 target_dir: str, 
                 filename: str = "training_results.csv") -> None:
    """
    Saves training and testing results to a CSV file.

    Args:
        results: A dictionary containing training and testing metrics.
        target_dir: The directory where the CSV file will be saved.
        filename: The name of the CSV file (default is 'training_results.csv').

    Returns:
        None
    """
    # Create the target directory if it doesn't exist
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Convert the results dictionary to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    file_path = target_dir_path / filename
    results_df.to_csv(file_path, index_label="epoch")
    
    print("\n" + "=" * 80)
    print(f"[INFO] Results saved to: {file_path}")
    print("=" * 80 + "\n")



# --- Functions for training and testing the Tree LSTM ---
def tree_train_step(model, dataloader, loss_fn, optimizer, device):
    """
    Trains a TreeLSTM model for one epoch using unbatched tree structures.

    For each batch (a list of root nodes), the model is called individually on
    each tree due to structural variability. 

    Args:
        model: A TreeLSTM model implementing forward(root).
        dataloader: A DataLoader yielding batches of (FormulaTreeNode, label).
        loss_fn: The loss function used for evaluation.
        optimizer: Optimizer used to update model parameters.
        device: Target device for tensor computation. 

    Returns:
        Tuple[float, float]: Average training loss and accuracy over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # Note 1 — Batching: Although batch_size > 1, each tree is processed individually.
    # This approach avoids tensor batching because trees have variable structure
    # and cannot be stacked into a single tensor.
    
    # Note 2 — Device: Tree roots are custom Python objects (FormulaTreeNode), not tensors,
    # so we don't move them to device. The tensors (e.g., embeddings) are created
    # on the correct device inside the TreeLSTM model.

    for roots, labels in dataloader:
        labels = labels.to(device)

        batch_logits = []
        for root in roots:
            root_logits = model(root)                    # shape [1]
            batch_logits.append(root_logits)

        logits = torch.stack(batch_logits).squeeze(1)    # (after stach:) [batch_size, 1] -> (after squezze(1):) [batch_size]
        preds = torch.round(torch.sigmoid(logits))

        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (preds == labels).sum().item()
        total_examples += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy

def tree_test_step(model, dataloader, loss_fn, device):
    """
    Evaluates a TreeLSTM model on a test dataset for one epoch.

    This function processes each tree individually (no batch parallelism),
    accumulates loss and accuracy, and returns average metrics over the dataset.

    Args:
        model: A TreeLSTM model accepting a single FormulaTreeNode as input.
        dataloader: A DataLoader yielding batches of (List[FormulaTreeNode], labels).
        loss_fn: The loss function used for evaluation.
        device: Target device for tensor computation.

    Returns:
        Tuple[float, float]: A tuple containing the average test loss and test accuracy.
    """
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for roots, labels in dataloader:
            labels = labels.to(device)

            batch_logits = []
            for root in roots:
                root_logits = model(root)                 # shape [1]
                batch_logits.append(root_logits)

            logits = torch.stack(batch_logits).squeeze(1) # (after stach:) [batch_size, 1] -> (after squezze(1):) [batch_size]
            preds = torch.round(torch.sigmoid(logits))

            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_examples += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def train_tree(model, train_loader, test_loader, loss_fn, optimizer, epochs, device,
               save_best: bool = True,
               save_dir: str = "models",
               model_name: str = None ):
    """
    Trains and evaluates a TreeLSTM model over multiple epochs.

    For each epoch:
        - Trains the model using `tree_train_step`.
        - Evaluates the model using `tree_test_step`.
        - Tracks loss and accuracy.
        - Optionally saves the best model (based on test accuracy).

    This function is designed for TreeLSTM models that operate on one tree at a time
    and cannot be batch-parallelized in the usual tensor-based way.

    Args:
        model: The TreeLSTM model to train.
        train_loader (DataLoader): Dataloader yielding training examples.
        test_loader (DataLoader): Dataloader yielding test examples.
        loss_fn: The loss function used for evaluation.
        optimizer: Optimizer used to update model parameters.
        epochs (int): Number of epochs to train the model.
        device (torch.device): Target device for computation.
        save_best (bool): Whether to save the model with highest test accuracy.
        save_dir (str): Directory where the best model will be saved.
        model_name (str, optional): Filename for saving the best model.

    Returns:
        Dict[str, List[float]]: Dictionary containing training and test metrics per epoch:
            {
                "train_loss": [...],
                "train_acc": [...],
                "test_loss": [...],
                "test_acc": [...]
            }
    """
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    best_test_acc = 0.0 
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        train_loss, train_acc = tree_train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = tree_test_step(model, test_loader, loss_fn, device)

        print(f"Epoch {epoch+1} |"
              f"train_loss: {train_loss:.4f} |"
              f"train_acc={train_acc:.4f} |"
              f"test_loss={test_loss:.4f} |"
              f"test_acc={test_acc:.4f}"
        )

        if save_best and test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())

            print("\n" + "=" * 80)
            print(f"[BEST MODEL SAVED] Epoch {epoch+1} | test_acc = {test_acc:.4f}")
            print("=" * 80 + "\n")

            if model_name is not None:
                model_to_save = copy.deepcopy(model)
                model_to_save.load_state_dict(best_model_state)
                save_model(model_to_save, target_dir=save_dir, model_name=model_name)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


# --- Function to count trainable parameters ---
def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.

    Note:
    We cannot use torchinfo.summary() for models like TreeLSTMClassifier,
    because they accept complex objects (e.g., FormulaTreeNode) as input, not standard tensors.
    summary() expects tensors (e.g., input_size=(batch_size, input_dim)), but in this case
    the model works recursively on tree structures, which are not compatible with the required format.

    This function is therefore an alternative solution to obtain useful information
    about the number of trainable parameters in the model, without having to artificially adapt the input.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# --- Model Evaluation ---
def eval_model(model, dataloader, loss_fn, device):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    with torch.inference_mode():
        for roots, labels in tqdm(dataloader):
            labels = labels.to(device)
            
            batch_logits = []
            for root in roots:
                root_logits = model(root)                 # shape [1]
                batch_logits.append(root_logits)

            logits = torch.stack(batch_logits).squeeze(1) # (after stach:) [batch_size, 1] -> (after squezze(1):) [batch_size]
            preds = torch.round(torch.sigmoid(logits))

            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_examples += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_examples
        
    return {"model_name": model.__class__.__name__, 
            "model_loss": avg_loss, 
            "model_acc": accuracy}


# --- Confusion Matrix ---
def evaluate_confusion_matrix(model, test_loader, device):
    """
    Evaluates a TreeLSTM model on the test set and plots the confusion matrix.

    This function processes each tree individually, collecting predicted and true labels,
    and then computes and visualizes the confusion matrix using scikit-learn utilities.

    Args:
        model: A trained TreeLSTM model that takes a FormulaTreeNode as input.
        test_loader: A DataLoader yielding batches of (List[FormulaTreeNode], labels).
        device: The torch device to perform inference on.

    Returns:
        numpy.ndarray: The confusion matrix as a 2D array.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for roots, labels in tqdm(test_loader):
            for root, label in zip(roots, labels):
                label = int(label.cpu().item())  # safe cast
                output = model(root)
                pred = torch.round(torch.sigmoid(output)).cpu().item()  # move pred to CPU
                all_preds.append(int(pred))
                all_labels.append(label)

    # Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Taut", "Taut"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return cm



# --- Function to analyze model errors ---
def analyze_model_errors(model, dataloader, device, threshold=0.5):
    """
    Analyzes misclassifications made by a TreeLSTM model on a dataset.

    For each example, the model predicts a label (based on a probability threshold),
    and compares it to the true label. Misclassified formulas are collected and returned,
    along with their string representation, true label, predicted label, and prediction probability.

    Args:
        model: A trained TreeLSTM model that accepts a FormulaTreeNode as input.
        dataloader: A DataLoader yielding batches of (List[FormulaTreeNode], labels).
        device: The torch device used for model computation.
        threshold (float): Classification threshold for converting probabilities into binary labels.

    Returns:
        Tuple[List[Dict], List[Dict]]: A tuple containing two lists:
            - false_positives: list of misclassified examples where true = 0 and pred = 1
            - false_negatives: list of misclassified examples where true = 1 and pred = 0

        Each entry is a dictionary with:
            {
                "formula": str,        # Fully parenthesized string representation
                "true_label": int,     # Ground truth label 
                "pred_label": int,     # Predicted label 
                "prob": float          # Raw sigmoid probability from the model
            }
    """
    model.eval()
    false_positives = []
    false_negatives = []

    with torch.inference_mode():
        for roots, labels in tqdm(dataloader):
            for root, true_label in zip(roots, labels):
                true_label = int(true_label.cpu().item())  # fix: move to CPU and cast to int

                # Predizione
                output = model(root)
                pred_prob = torch.sigmoid(output).item()
                pred_label = int(pred_prob >= threshold)

                if pred_label != true_label:
                    formula_str = root.formula.to_fully_parenthesized_str()
                    error_info = {
                        "formula": formula_str,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "prob": pred_prob
                    }
                    if pred_label == 1:
                        false_positives.append(error_info)
                    else:
                        false_negatives.append(error_info)

    return false_positives, false_negatives

