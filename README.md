# deepsat-experiments
Experiments related to the DeepSAT project.

# Project Structure Explained
The repositoey in organized into modules and folders that separate the core logic, experimental tools, datasets, and trained models. 


## `theorem_prover_core/` - Core Logic and Proof Search Modules
This is the main Python package containing all the essential components for the DeepSAT project and the experimental setup presented in the paper. 

**Key Files**:
 - `formula.py`: Defines the internal representation of porpositional logic formulas and their parsing and manipulation.
 - `sequent.py` - Provides tools to represent and manilupate sequents, used to represent logical statements during proof construction.
 - `proofgraph.py` - Implements the central data structure for proof searh: a hypergraph of sequents and inference rules.
 - `mcts.py` - Contains the Monte Carlo Tree Search (MCTS) algorithm adapted for explore the space of logical proofs. 
 - `show_structure.py` - Utility for debugging or visualizing the structure of sequents or proof steps (optional).

## **`ITCTS_notebooks/` - Experiments and Models (as used in the paper)**
This folder includes Jupyter notebooks, training scripts, dataset, and trained models used during the experiments documented in the paper. 

### **Notebooks**: 
These notebooks guide the process of training, evaluating, and tracking the models: 
 - `rnn_experiment_tracking.ipynb`
 - `mcts_experiments-tracking.ipynb`
 - `hardware_info.ipynb` - Logs hardware configuration used during training. 

### **Scripts**
Python scripts for data processing, training utilities, and parameter sweeps: 
 - `logic_utils.py` - Logic-specific helper functions. 
 - `data_setup.py` - Dataset loading and propocessing.
 - `train_utils.py` - Training utilities for training models.
 - `sweep_1_hidden_states_fc_size.py` - Explores different hidden state sizes and fully connected layer dimensions. 
 - `sweep_2_focal_loss.py` - Tests different parameter configurations for the Asymmetric Focal Loss. 
 - `sweep_3_lr.py` - Sweeps learning rates to optimize model performance.

### **`datasets/` - Experimental Data**
CVS files used for training and evaluation: 
 - `normalized_formulas_dataset.cvs` - Dataset of normalized propositional formulas.
 - `extended_dataset_with_tautologies.cvs` - Extended dataset including instatiated tautology schemata.
 - `wand_*.cvs` - Exported Weight & Biases logs from hyperparameter sweeps.
 - `mcts_sweep_results.cvs`, `uniform_poliy_sweep_results.cvs` - Results from experiments with different MCTS configurations. 

### **`models/` - Pretrained Neural Models**
Trained PyTorch models used in experiments: 
 - `Bidirectional_GRU_*.pth` - GRU-based models.
 - `Tree_lstm.pth`, `Tree_lstm_without_dropout.pth` - Tree-LSTM models. 
