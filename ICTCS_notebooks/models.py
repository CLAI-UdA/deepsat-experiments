import torch
import torch.nn as nn

from logic_utils import FormulaTreeNode



# --- Asymmetric Focal Loss for Binary Classification ---

# Due to the class imbalance in our dataset (~26% tautologies), the model may bias toward 
# predicting the majority class (non-tautologies). This leads to high accuracy but poor recall 
# on the minority class, which is undesirable in many reasoning or safety-critical settings.

# To address this, we use an Asymmetric Focal Loss, a refined version of the standard focal loss.
# The core idea is to:
# - Assign higher weight (α) to the minority class (tautologies) to penalize false negatives more.
# - Apply a modulating factor (1 - p)^γ to focus learning on hard examples.
# - Use separate α and γ values for each class for better control.

# Loss formula:
# L(y, ŷ) = 
#   - α_pos * y * (1 - ŷ)^γ_pos * log(ŷ)
#   - α_neg * (1 - y)^γ_neg * log(1 - ŷ)
# where:
#   - y is the true label (0 or 1)
#   - ŷ is the predicted probability (after sigmoid)
#   - α_pos/neg control class weighting
#   - γ_pos/neg control the focus on hard examples


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for binary classification with class imbalance.

    This loss is designed to handle imbalanced datasets (e.g., tautology detection),
    where false negatives are more critical than false positives.

    Args:
        alpha_pos (float): Class weight for the positive class (e.g., tautologies).
        alpha_neg (float): Class weight for the negative class (e.g., non-tautologies).
        gamma_pos (float): Focusing parameter for the positive class.
        gamma_neg (float): Focusing parameter for the negative class.
        reduction (str): Reduction method to apply to the output: 'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha_pos :int, alpha_neg :int, gamma_pos :int, gamma_neg :int, reduction='mean'):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Computes the Asymmetric Focal Loss between predicted logits and true targets.

        Args:
            logits (Tensor): Raw output logits from the model, of shape [batch_size].
            targets (Tensor): Binary ground truth labels, of shape [batch_size].

        Returns:
            Tensor: Computed loss value.
        """
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)  # Avoid log(0)

        # Loss for positive (tautology)
        pos_loss = self.alpha_pos * (1 - probs) ** self.gamma_pos * torch.log(probs)
        # Loss for negative (non-tautology)
        neg_loss = self.alpha_neg * (probs) ** self.gamma_neg * torch.log(1 - probs)

        # Full loss
        loss = -targets * pos_loss - (1 - targets) * neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# --- Binary Tree LSTM Cell ---
class BinaryTreeLSTMCell(nn.Module):
    """
    Binary TreeLSTM cell for processing tree-structured data with binary branching.

    This implementation extends the standard LSTM cell to binary trees by:
    - Using two hidden/cell state pairs (left and right children).
    - Computing separate forget gates for each child.
    - Combining them with the input through standard LSTM gating mechanisms.

    Reference: Tai et al., "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks" (2015)

    Args:
        input_size (int): Dimensionality of the input embedding x.
        hidden_size (int): Dimensionality of the hidden and cell states (h, c).
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size

        # Linear transformations for i (input), o (output), and u (update) gates
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)
        self.U_iou = nn.Linear(2 * hidden_size, 3 * hidden_size)

        # Linear transformations for forget gates (left and right children) 
        self.W_f = nn.Linear(input_size, 2 * hidden_size)
        self.U_f = nn.Linear(2 * hidden_size, 2 * hidden_size)

    def forward(self, x, left_state, right_state):
        h_l, c_l = left_state  # left hidden states and cell states,  [1, hidd_size], [1, hidden_size]
        h_r, c_r = right_state # right hidden states and cell states, [1, hidd_size], [1, hidden_size]

        h_cat = torch.cat([h_l, h_r], dim=1)                 # [1, 2xhidden_size]

        # Input, Output, Update gates
        iou = self.W_iou(x) + self.U_iou(h_cat)              # [1, 3xhidden_size]             
        i, o, u = torch.chunk(torch.sigmoid(iou), 3, dim=1)  # each [1, hidden_size]
        u = torch.tanh(u)                                    # [1, hidden_size]

        # Forget gates
        f = self.W_f(x) + self.U_f(h_cat)                    # [1, 2xhidden_size]
        f_l, f_r = torch.chunk(torch.sigmoid(f), 2, dim=1)   # each [1, hidden_size]

        # Cell state
        c = i * u + f_l * c_l + f_r * c_r                    # [1, hidden_size]
        h = o * torch.tanh(c)                                # [1, hidden_size]
 
        return h, c


# --- Tree LSTM Encoder ---
class TreeLSTMEncoder(nn.Module):
    """
    A TreeLSTM encoder for binary tree-structured data.

    This module recursively encodes a formula tree by applying a BinaryTreeLSTMCell
    bottom-up, starting from the leaves and propagating hidden and cell states upward.

    Supports:
    - Leaf nodes (no children): initialized with zero states.
    - Unary nodes: the same child state is passed as both left and right.
    - Binary nodes: recursive calls for left and right children.

    Args:
        vocab_size (int): Size of the input vocabulary (number of unique token indices).
        embedding_dim (int): Dimension of input embeddings.
        hidden_size (int): Dimension of TreeLSTM hidden and cell states.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        # Note: This encoder operates on a single root node at a time.
        # It computes the embedding from a single index, shaped [1] -> [1, embedding_dim].
        # Native batch processing is not supported in this architecture.
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.cell = BinaryTreeLSTMCell(input_size=embedding_dim, hidden_size=hidden_size)

    def forward(self, node: FormulaTreeNode):
        # Ottieni embedding del nodo corrente
        x = self.embedding(torch.tensor([node.embedding_index], device=self.embedding.weight.device))

        # Caso foglia: nessun figlio
        if len(node.children) == 0:
            zero_state = (
                torch.zeros(1, self.cell.hidden_size, device=x.device), # zero_state = (h_zero, c_zero)
                torch.zeros(1, self.cell.hidden_size, device=x.device)  # where h_zero: initial hidden state filled whit zeros
                                                                        # c_zero: initial cell state filled with zeros
            )
            h, c = self.cell(x, zero_state, zero_state)                 # [1, hidden_size], [1, hidden_size]
            return h, c

        # Caso unary: un figlio (es. Negazione)
        elif len(node.children) == 1:
            child_state = self.forward(node.children[0])
            h, c = self.cell(x, child_state, child_state) 
            return h, c                                                 # [1, hidden_state]

        # Caso binary: due figli
        elif len(node.children) == 2:
            left_state = self.forward(node.children[0])
            right_state = self.forward(node.children[1])
            h, c = self.cell(x, left_state, right_state)                # [1, hidden_size], [1, hidden_size]
            return h, c
            
        else:
            raise ValueError(f"Unexpected number of children: {len(node.children)}")


# --- Tree LSTM Classifier V1---
class TreeLSTMClassifierV1(nn.Module):
    """
    A binary classifier for tree-structured formulas using a TreeLSTM encoder.

    This model uses a TreeLSTM to recursively encode a propositional formula tree
    into a hidden representation, followed by a feedforward network to output a binary logit.

    Architecture:
    - TreeLSTMEncoder: computes the hidden state of the root node.
    - Fully-connected layer (fc1) with ReLU: projects hidden state to intermediate features.
    - Fully-connected layer (fc2): maps to a single logit for binary classification.

    Args:
        vocab_size (int): Number of unique token indices in the input vocabulary.
        embedding_dim (int): Size of the input embeddings.
        hidden_size (int): Dimension of TreeLSTM hidden states.
        fc_size (int): Size of the intermediate fully-connected layer.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, fc_size):
        super().__init__()

        # Note: this model operates on a single tree at a time.
        # It does not support batched inputs like sequential models (e.g., GRU).
        # The batch dimension is never used in input/output tensors: each forward pass
        # receives a single root `FormulaTreeNode`, not a tensor of shape [batch_size, seq_len, embed_dim].
        #
        # Tensors inside the model (e.g., hidden states, embeddings) always have shape [1, hidden_size]
        # and are created dynamically during the recursive forward pass.
        # This design is compatible with PyTorch's DataLoader,
        # but it prevents the kind of parallelism typically used in batch training.

        self.encoder = TreeLSTMEncoder(vocab_size, embedding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, 1)  

    def forward(self, root_node: FormulaTreeNode):
        h, _ = self.encoder(root_node)  # root's hidden state (ignore the cell state)
        x = self.relu(self.fc1(h))
        output = self.fc2(x)
        output = output.squeeze(1)  # [batch_size] 
        return output 


# --- Tree LSTM Classifier V2 (with Dropout) ---
class TreeLSTMClassifierV2(nn.Module):
    """
    A TreeLSTM-based binary classifier with dropout regularization.

    This version of the classifier extends the basic TreeLSTMClassifier by introducing
    dropout after the first fully connected layer to reduce overfitting.

    Architecture:
    - TreeLSTMEncoder: recursively encodes the input formula tree.
    - Fully-connected layer with ReLU: projects hidden representation to intermediate space.
    - Dropout: applied after ReLU to regularize the model.
    - Output layer: maps to a single logit for binary classification.

    Args:
        vocab_size (int): Size of the input vocabulary (number of unique token indices).
        embedding_dim (int): Dimensionality of input embeddings.
        hidden_size (int): Dimensionality of TreeLSTM hidden states.
        fc_size (int): Size of the intermediate fully connected layer.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, fc_size):
        super().__init__()
        self.encoder = TreeLSTMEncoder(vocab_size, embedding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(fc_size, 1) 

    def forward(self, root_node: FormulaTreeNode):
        h, _ = self.encoder(root_node)  
        x = self.relu(self.fc1(h))
        x = self.dropout(x)
        output = self.fc2(x)
        output = output.squeeze(1)  
        return output




# =========================== 
#   NEW MODEL V3 DROPOUT=0.4 
# ===========================

# --- Tree LSTM Classifier V3 (with NEW Dropout) ---
class TreeLSTMClassifierV3(nn.Module):
    """
    A TreeLSTM-based binary classifier with dropout regularization.

    This version of the classifier extends the basic TreeLSTMClassifier by introducing
    dropout after the first fully connected layer to reduce overfitting.

    Architecture:
    - TreeLSTMEncoder: recursively encodes the input formula tree.
    - Fully-connected layer with ReLU: projects hidden representation to intermediate space.
    - Dropout: applied after ReLU to regularize the model.
    - Output layer: maps to a single logit for binary classification.

    Args:
        vocab_size (int): Size of the input vocabulary (number of unique token indices).
        embedding_dim (int): Dimensionality of input embeddings.
        hidden_size (int): Dimensionality of TreeLSTM hidden states.
        fc_size (int): Size of the intermediate fully connected layer.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, fc_size):
        super().__init__()
        self.encoder = TreeLSTMEncoder(vocab_size, embedding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(fc_size, 1) 

    def forward(self, root_node: FormulaTreeNode):
        h, _ = self.encoder(root_node)  
        x = self.relu(self.fc1(h))
        x = self.dropout(x)
        output = self.fc2(x)
        output = output.squeeze(1)  
        return output






























