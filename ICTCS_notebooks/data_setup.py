"""
Contains functionality for creating Formula Dataset and PyTorch DataLoaders for 
formulas classification data.
"""

# --- Importing Libraries ---
import sys
import os
import random
from typing import Dict, Tuple, List, Set, Union, Type, Literal
from itertools import product
from dataclasses import dataclass
from collections import Counter
import re

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from train_utils import CustomTokenizer
from logic_utils import parse_formula_string


from pathlib import Path

from logic_utils import generate_normalized_random_formula, is_tautology

# --- Importing Formula Class ---
# Go two levels up: from ICTCS_notebooks → theorem_prover_core → project root
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from theorem_prover_core.formula import (Formula, Letter, Falsity, Conjunction, Disjunction, Implication,
                                         Negation, BinaryConnectiveFormula, UnaryConnectiveFormula, bottom)

from logic_utils import Normalizer, CustomTokenizer, instantiate_random_formulas, parse_formula_string, FormulaTreeNode, assign_embedding_indices


# --- Generate a Dataset ---
def generate_normalized_dataset(num_formulas: int, 
                                max_depth: int, 
                                num_letters: int) -> pd.DataFrame:
    """
    Generates a DataFrame containing random normalized formulas and their tautology status.

    Args: 
        num_formulas (int): Number of formulas to generate.
        max_depth (int): maximum depth of the generated formula's syntax tree.
        num_letters (int): Number of propositional letters that can be used.
    
    Returns: 
        pd.DataFrame: A DataFrame with columns:
            - 'formula': Fully parenthesized string representation,
            - 'is_tautology': Boolean indicating if the formula is a tautology.
    """
    data = []
    seen_formulas = set()

    while len(data) < num_formulas:
        formula = generate_normalized_random_formula(max_depth=max_depth, num_letters=num_letters)

        formula_str = formula.to_fully_parenthesized_str()

        if formula_str in seen_formulas:
            continue

        seen_formulas.add(formula_str)

        tautology_status = is_tautology(formula)

        data.append({
            "formula": formula_str,
            "is_tautology": tautology_status
        })

    return pd.DataFrame(data)



# --- Adding new tatologies to the dataset ---
def add_new_tautologies_to_dataset(dataset: pd.DataFrame, 
                                   tautologies: List[Formula],
                                   num_samples: int,
                                   max_depth: int,
                                   num_letters: int,
                                   seed: int = None) -> pd.DataFrame:
    """
    Add a fixed number of unique tautologies to the dataset.

    Returns: Updated and shuffled DataFrame.
    """
    existing_formulas = set(dataset['formula'].tolist())
    new_data = []
    attempts = 0
    batch_size = 500
    seed_base = seed if seed is not None else random.randint(0, 10000)
    normalizer = Normalizer()

    while len(new_data) < num_samples and attempts < num_samples * 5:
        fresh = instantiate_random_formulas(batch_size, 
                                            max_depth,
                                            num_letters,
                                            tautologies, 
                                            seed_base + attempts)

        for formula in fresh:
            formula = normalizer.normalize(formula)
            formula_str = formula.to_fully_parenthesized_str()

            if formula_str not in existing_formulas:
                assert is_tautology(formula), f"Generated formula is not a tautology: {formula_str}"

                new_data.append({
                    'formula': formula_str,
                    'is_tautology': True
                })
                existing_formulas.add(formula_str)

                if len(new_data) >= num_samples:
                    break
        attempts += 1

    if len(new_data) < num_samples:
        print(f"[Warning] Only {len(new_data)} unique tautologies added out of {num_samples} requested.")

    new_df = pd.DataFrame(new_data)
    updated_dataset = pd.concat([dataset, new_df], ignore_index=True)
    return updated_dataset.sample(frac=1, random_state=seed).reset_index(drop=True)


# --- Create a PyTorch Dataset ---
class FormulaDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.classes = ['non-tautology', 'tautology']
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return (
            f"Dataset FormulaDataset\n"
            f"  Number of datapoints: {len(self)}\n"
            f"  Input shape: {self.X[0].shape if len(self.X) > 0 else 'N/A'}\n"
            f"  Target type: {self.y.dtype}\n"
        )
        

# --- Converting a dataset of formulas into PyTorch dataloaders for training/testing ---
def prepare_formula_dataset(dataset: pd.DataFrame,
                            test_size: float,
                            batch_size: int,
                            seed: int = 42
                            ) -> Tuple[DataLoader, DataLoader, CustomTokenizer,
                                       List[Formula], List[Formula],
                                       List[bool], List[bool], List[int]]:
    """
    Converts a dataset of logical formulas into PyTorch DataLoaders for training and testing.

    Args:
        dataset (pd.DataFrame): A DataFrame with columns 'formula' and 'is_tautology'.
        test_size (float): Fraction of data to use as test set (e.g., 0.2).
        batch_size (int): Batch size for DataLoaders.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple containing:
            - train_dataloader (DataLoader): Dataloader for training set.
            - test_dataloader (DataLoader): Dataloader for test set.
            - tokenizer (CustomTokenizer): Fitted tokenizer on training formulas.
            - X_train (List[Formula]): Parsed training formulas.
            - X_test (List[Formula]): Parsed test formulas.
            - y_train (List[bool]): Training labels.
            - y_test (List[bool]): Test labels.
            - idx_test (List[int]): Original indices of test samples.
    """
    # Parse formulas
    parsed_formulas = [parse_formula_string(f) for f in dataset['formula']]
    truth_values = dataset['is_tautology'].tolist()

    if 'original_index' in dataset.columns:
        original_indices = dataset['original_index'].tolist()
    else:
        original_indices = list(range(len(dataset)))

    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        parsed_formulas, truth_values, original_indices,
        test_size=test_size,
        random_state=seed
    )

    # Tokenizer
    tokenizer = CustomTokenizer()
    tokenizer.fit(X_train)

    # Tokenize and convert to tensor
    X_train_seq = [tokenizer.tokenize(f) for f in X_train]
    X_test_seq = [tokenizer.tokenize(f) for f in X_test]

    X_train_tensors = [torch.tensor(seq, dtype=torch.long) for seq in X_train_seq]
    X_test_tensors = [torch.tensor(seq, dtype=torch.long) for seq in X_test_seq]

    # Pad sequences
    X_train_padded = pad_sequence(X_train_tensors, batch_first=True, padding_value=0)
    X_test_padded = pad_sequence(X_test_tensors, batch_first=True, padding_value=0)

    # Convert labels to tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Build datasets
    train_data = FormulaDataset(X_train_padded, y_train_tensor)
    test_data = FormulaDataset(X_test_padded, y_test_tensor)

    # Dataloaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, tokenizer, X_train, X_test, y_train, y_test, idx_test


# --- Extending Dataset with SATLIB - Benchmark Problems ---
def parse_dimacs_files(base_dir: str):
    """
    Parses DIMACS CNF files from a directory and returns a labeled dataset of formulas.
    
    SATLIB - Benchmark Problems: (https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)
    Formulas Downloaded from SATLIB: 
    - uf20-91: 20 variables, 91 clauses - 1000 instances, all satisfiable
    - uf50-218 / uuf50-218: 50 variables, 218 clauses - 1000 instances, all sat/unsat

    This function scans subdirectories under the specified base directory to identify 
    and process CNF files representing propositional logic formulas in DIMACS format. 
    It labels the formulas as tautology or not tautology based on directory naming 
    conventions (`uf*` = satisfiable, `uuf*` = unsatisfiable).

    Args:
        base_dir (str): The path to the top-level directory containing `uf*` and `uuf*` folders
                        with .cnf files inside.

    Returns:
        pd.DataFrame: A DataFrame containing two columns:
                      - 'formula': the parsed `Formula` object
                      - 'is_tautology': a boolean indicating whether the formula is satisfiable.
    """
    formulas_data = []
    base_path = Path(base_dir)

    for folder in base_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name.lower()
            if folder_name.startswith("uf") and not folder_name.startswith("uuf"):
                label = True  # satisfiable
            elif folder_name.startswith("uuf"):
                label = False  # unsatisfiable
            else:
                print(f"Skipping unknown folder: {folder_name}")
                continue

            for file_path in folder.glob("*.cnf"):
                with open(file_path, 'r') as file:
                    dimacs_lines = file.readlines()
                try:
                    formula = Formula.from_dimacs(dimacs_lines)
                    formulas_data.append({
                        'formula': formula,
                        'is_tautology': label
                    })
                except Exception as e:
                    print(f"Error parsing {file_path.name}: {e}")

    return pd.DataFrame(formulas_data)


# --- Formul Tree Representation ---
class FormulaTreeNode:
    """
    Represents a node in the syntax tree of a logical formula.
    """
    def __init__(self, formula):
        self.formula = formula
        self.children = []
        self.embedding_index = None  # will be assigned by the tokenizer

        # Costruzione ricorsiva dei figli
        if isinstance(formula, UnaryConnectiveFormula):
            self.children.append(FormulaTreeNode(formula.formula))

        elif isinstance(formula, BinaryConnectiveFormula):
            self.children.append(FormulaTreeNode(formula.left))
            self.children.append(FormulaTreeNode(formula.right))

    def __repr__(self):
        return f"Node({self.formula}, children={len(self.children)})"
   
    def to_formula(self):
        """
        Ricostruisce la Formula (oggetto) dal syntax tree.
        """
        if not self.children:
            return self.formula  # caso base: lettera o ⊥

        if isinstance(self.formula, UnaryConnectiveFormula):
            return type(self.formula)(self.children[0].to_formula())

        if isinstance(self.formula, BinaryConnectiveFormula):
            return type(self.formula)(
                self.children[0].to_formula(),
                self.children[1].to_formula()
            )
        
        raise ValueError("Tipo di formula non riconosciuto")


# --- Dataset based on tree ---
class TreeFormulaDataset(torch.utils.data.Dataset):
    def __init__(self, formulas: List[Formula], labels: List[float], tokenizer: CustomTokenizer):
        self.formulas = formulas
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        formula = self.formulas[idx]
        root = FormulaTreeNode(formula)
        assign_embedding_indices(root, self.tokenizer)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return root, label


# --- Custom Tree collate function ---
def tree_collate_fn(batch: List[Tuple[FormulaTreeNode, torch.Tensor]]) -> Tuple[List[FormulaTreeNode], torch.Tensor]:
    """
    Custom collate function for TreeLSTM:
    - Receives the DataLoader batch: a list of (root, label) tuples
    - Returns:
        * a list of roots (not tensorizable)
        * a stacked tensor of labels

    Args:
        batch: List of tuples (FormulaTreeNode, label)

    Returns:
        roots: list of FormulaTreeNode
        labels: scalar tensors (float), stacked into a batch tensor of shape [batch_size]
    """
    # Extract tree roots and labels separately:
    roots, labels = zip(*batch)
    
    return list(roots), torch.stack(labels)



# --- Function to create a balanced subset for experiments ---
def prepare_balanced_tree_dataloaders(dataset: pd.DataFrame,
                                      train_size: int,
                                      test_size: int,
                                      positive_ratio: float,
                                      batch_size: int,
                                      test_split_ratio: float = 0.2,
                                      seed: int = 42):
    """
    Splits, balances, and prepares dataloaders for training TreeLSTM with a consistent tokenizer.

    Args:
        dataset (pd.DataFrame): Contains 'formula' and 'is_tautology'.
        train_size (int): Number of examples in the balanced training set.
        test_size (int): Number of examples in the balanced test set.
        positive_ratio (float): Desired percentage of tautologies in the subsets.
        batch_size (int): Batch size.
        test_split_ratio (float): Proportion for the initial test split.
        seed (int): Random seed.

    Returns:
        train_loader, test_loader, tokenizer
    """
    
    # 1. Parsing formulas
    parsed_formulas = [parse_formula_string(f) for f in dataset["formula"]]
    dataset = dataset.copy()
    dataset["parsed_formula"] = parsed_formulas

    # 2. Train/test split
    df_train, df_test = train_test_split(
        dataset,
        test_size=test_split_ratio,
        stratify=dataset["is_tautology"],
        random_state=seed
    )

    # 3. Proportional balanced sampling
    def sample_balanced(df, total_size, positive_ratio, seed):
        pos_count = int(total_size * positive_ratio)
        neg_count = total_size - pos_count

        pos_pool = df[df["is_tautology"] == 1]
        neg_pool = df[df["is_tautology"] == 0]

        if len(pos_pool) < pos_count or len(neg_pool) < neg_count:
            raise ValueError("Not enough examples in one of the two classes to sample from")

        pos_sample = pos_pool.sample(pos_count, random_state=seed)
        neg_sample = neg_pool.sample(neg_count, random_state=seed)

        balanced_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)
        return balanced_df

    train_balanced = sample_balanced(df_train, train_size, positive_ratio, seed)
    test_balanced = sample_balanced(df_test, test_size, positive_ratio, seed)

    # 4. Output distribution
    def print_distribution(name, df):
        counter = Counter(df["is_tautology"])
        total = len(df)
        ratio_true = counter[1] / total * 100
        ratio_false = counter[0] / total * 100
        print(f"{name} set ({total} samples): False = {ratio_false:.2f} % and True = {ratio_true:.2f} %")

    print_distribution("Train", train_balanced)
    print_distribution("Test", test_balanced)

    # 5. Tokenizer (fit only on the train set!)
    tokenizer = CustomTokenizer()
    tokenizer.fit(train_balanced["parsed_formula"])

    # 6. Dataloaders
    train_dataset = TreeFormulaDataset(
        formulas=train_balanced["parsed_formula"].tolist(),
        labels=train_balanced["is_tautology"].tolist(),
        tokenizer=tokenizer
    )

    test_dataset = TreeFormulaDataset(
        formulas=test_balanced["parsed_formula"].tolist(),
        labels=test_balanced["is_tautology"].tolist(),
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tree_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tree_collate_fn)

    return train_loader, test_loader, tokenizer


# --- Tree DataLoaer ---
def prepare_tree_dataloaders(dataset: pd.DataFrame,
                             test_size: float,
                             batch_size: int,
                             seed: int = 42):
    """
    Prepares dataloaders for TreeLSTM with a tokenizer fitted only on the training set.

    Args:
        dataset (pd.DataFrame): contains 'formula' and 'is_tautology' columns
        test_size (float): proportion of data to use for testing
        batch_size (int): batch size
        seed (int): seed for reproducibility

    Returns:
        train_loader, test_loader, tokenizer: PyTorch DataLoaders and fitted tokenizer
    """
    assert "formula" in dataset.columns and "is_tautology" in dataset.columns

    # 1. Parsing formulas
    formulas = [parse_formula_string(f) for f in dataset["formula"]]
    labels = dataset["is_tautology"].tolist()

    # 2. Train/test split
    train_formulas, test_formulas, train_labels, test_labels = train_test_split(
        formulas, labels, test_size=test_size, random_state=seed
    )

    # 3. Tokenizer fitted only on the training set
    tokenizer = CustomTokenizer()
    tokenizer.fit(train_formulas)

    # 4. PyTorch datasets
    train_dataset = TreeFormulaDataset(train_formulas, train_labels, tokenizer)
    test_dataset = TreeFormulaDataset(test_formulas, test_labels, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=tree_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=tree_collate_fn
    )

    return train_loader, test_loader, tokenizer























































