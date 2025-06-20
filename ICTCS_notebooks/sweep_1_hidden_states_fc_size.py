import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import pandas as pd

from models import TreeLSTMClassifier, AsymmetricFocalLoss
from train_utils import compute_vocab_size, device, CustomTokenizer
from data_setup import prepare_balanced_tree_dataloaders, prepare_formula_dataset, TreeFormulaDataset
from torch.utils.data import DataLoader


# Parametri base
TEST_SIZE = 0.2
BATCH_SIZE = 16
SEED = 42

# Carica il dataset e prepara i dati
datapath = "datasets/extended_dataset_with_tautologies.csv"
dataset = pd.read_csv(datapath)  # Modifica il path se necessario

tree_train_loader, tree_test_loader, tokenizer = prepare_balanced_tree_dataloaders(dataset=dataset,
                                                                                   train_size=1000,
                                                                                   test_size=200,
                                                                                   positive_ratio=0.26,
                                                                                   batch_size=16,
                                                                                   test_split_ratio=0.2,
                                                                                   seed=42
                                                                                  )

VOCAB_SIZE = compute_vocab_size(tokenizer)


# Calcolo vocabolario
VOCAB_SIZE = compute_vocab_size(tokenizer)


def tree_train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for roots, labels in dataloader:
        labels = labels.to(device)
        batch_logits = [model(root) for root in roots]
        logits = torch.stack(batch_logits).view(-1)
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
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for roots, labels in dataloader:
            labels = labels.to(device)
            batch_logits = [model(root) for root in roots]
            logits = torch.stack(batch_logits).view(-1)
            preds = torch.round(torch.sigmoid(logits))

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_examples += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy

def main():
    run = wandb.init(project="theorem_prover", group="Step_1_hidden_states_fc_size")
    config = wandb.config

    run.name = f"TreeLSTM_h{config.hidden_size}_fc{config.fc_size}"

    model = TreeLSTMClassifier(vocab_size=VOCAB_SIZE,
                               embedding_dim=config.embedding_dim,
                               hidden_size=config.hidden_size,
                               fc_size=config.fc_size).to(device)

    loss_fn = AsymmetricFocalLoss(
        alpha_pos=config.alpha_pos,
        alpha_neg=config.alpha_neg,
        gamma_pos=config.gamma_pos,
        gamma_neg=config.gamma_neg
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in tqdm(range(config.num_epochs), desc="Training Epochs"):
        train_loss, train_acc = tree_train_step(model, tree_train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = tree_test_step(model, tree_test_loader, loss_fn, device)

        print(f"Epoch {epoch+1} | train_loss: {train_loss:.4f} | train_acc={train_acc:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

    wandb.finish()

sweep_config = {
    "method": "grid",
    "name": "Sweep_Step_1_hidden_states_fc_size",
    "metric": {"goal": "maximize", "name": "test_acc"},
    "parameters": {
        "hidden_size": {"values": [32, 64, 128]},
        "fc_size": {"values": [16, 32, 64]},
        "embedding_dim": {"value": 32},
        "alpha_pos": {"value": 0.3},
        "alpha_neg": {"value": 0.7},
        "gamma_pos": {"value": 3.0},
        "gamma_neg": {"value": 1.5},
        "learning_rate": {"value": 0.0005},
        "num_epochs": {"value": 5}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="theorem_prover")
    wandb.agent(sweep_id, function=main, count=9)

