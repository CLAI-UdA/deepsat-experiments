import numpy as np
from theorem_prover_core.sequent import Sequent 
from theorem_prover_core.formula import (Formula, Letter, Falsity, Conjunction, Disjunction, Implication,
                                         Negation, BinaryConnectiveFormula, UnaryConnectiveFormula, bottom)

number_simulations = (
    1600  # Following AlphaGo Zero model, but PROVISORY!
)
c_puct_value = 5 #PROVISORY (it works well with chain test)

"""
Monte Carlo Tree Search for DNN-guided proof search.

Each node represents a logical sequent (proof state). The MCTS
navigates the space of inference steps (Sequent.Move) to construct
proof trees, guided by neural network policy and value.

"""


def softmax(x):
    """
    Compute the softmax of a vector.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        np.ndarray: Probability distribution.
    """
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def formula_applied_to(sequent: Sequent, move: Sequent.Move) -> Formula: 
    """
    Retrieves the formula (prmise or conclusion) to which the move was applied.

    Args: 
        sequent (Sequent): The sequent before applying the rule.
        move (Sequent.Move): The inference rule to be applied.

    Returns: 
        Formula: The formula on which the rule operates.
    """
    return (
        sequent.conclusion if move.pos == -1
        else sequent.premises[move.pos]
    )


def aggregate_by_connective(move, child_nodes, mode="value"):   
    """
    Aggregates values or proof statuses from child nodes,
    using logical semantics of the rule (conjunction, disjunction, implication, etc.).

    Args:
        move (Sequent.Move): The inference move applied.
        child_nodes (list): Either ProofTreeNode or ProofGraph.Node objects.
        mode (str): "value" or "proved"
        parent_sequent (Sequent): Sequent from which the move was generated.

    Returns:
        float or bool: Aggregated result based on mode.
    """
    # Compatibility: detect node type
    def get_q(node):
        return getattr(node, "_Q", 1.0)  # Default to 1.0 if no Q present

    def is_proved(node):
        if hasattr(node, "pg_node"):
            return node.pg_node.num_moves.n == 0
        return node.num_moves.n == 0

    Qs = [get_q(c) for c in child_nodes if get_q(c) is not None]
    subgoals_proved = [is_proved(c) for c in child_nodes]

    return min(Qs) if mode == "value" else all(subgoals_proved) 



# --- ProofTreeNode Class ---
class ProofTreeNode:
    """A node in the MCTS proof tree, corresponding to a ProofGraph.Node."""

    def __init__(self, pg_node, parent, prior_p):
        self.pg_node = pg_node  # Underlying ProofGraph node
        self.sequent = pg_node.sequent  
        self._parent = parent  
        self._children = dict()  # maps move to list of ProofTreeNode
        self._n_visits = 0  
        self._Q_total = 0  # Total accumulated value
        self._Q = 0        # Mean value
        self._u = 0        # Exploration bonus
        self._P = prior_p  # Prior from neural policy
        self.id = pg_node.id  # For matching nodes across MCTS tree and proof graph

    def expand(self, move_priors, proof_graph):
        """
        Expands the node by applying rules to generate children.

        Args:
            move_priors (list): List of (Sequent.Move, prior_prob) tuples.
            proof_graph: The underlying proof graph to update.
        """
        for move, prob in move_priors:
            if move not in self._children:
                children_sequents = self.sequent.rule(move)
                if children_sequents:
                    child_pg_nodes = proof_graph.add_children(self.pg_node, move, children_sequents)
                    child_nodes = [
                        ProofTreeNode(pg_node, self, prob)
                        for pg_node in child_pg_nodes
                    ]
                    self._children[move] = child_nodes

    def select(self, c_puct):
        """
        Selects the best move based on logical aggregation of UCB values.

        Args:
            c_puct (float): PUCT exploration constant.

        Returns:
            (Sequent.Move, list[ProofTreeNode]): best move and its children.
        """
        best_move = None
        best_nodes = None
        best_value = -float("inf")

        for move, child_nodes in self._children.items():
            # Compute the Q + U values for each child
            Qs_with_U = [child.get_value(c_puct) for child in child_nodes if child.get_value(c_puct) is not None]

            if not Qs_with_U:
                continue  # Skip this move if no valid children

            # Retrieve the formula on which the rule acts
            f = formula_applied_to(self.sequent, move)

            # Use logic-aware aggregation
            if isinstance(f, Conjunction):
                total_value = min(Qs_with_U)
            elif isinstance(f, Disjunction) and move.pos == -1:
                total_value = max(Qs_with_U)
            elif isinstance(f, Disjunction) and move.pos >= 0:
                total_value = min(Qs_with_U)  
            elif isinstance(f, (Implication, Negation)):
                total_value = Qs_with_U[0]
            else: 
                print(f"[DEBUG] Unrecognized formula: {f} (type: {type(f)})")
                raise ValueError("Unknown connective type.")
            
            # Update the best move
            if total_value > best_value: 
                best_value = total_value
                best_move = move
                best_nodes = child_nodes
        
        return best_move, best_nodes

    def update(self, leaf_value):
        """
        Updates visit count and Q-value with the new evaluation.

        Args:
            leaf_value (float): Value estimate from simulation [0, 1].
        """
        self._n_visits += 1
        self._Q_total += leaf_value
        self._Q = self._Q_total / self._n_visits

    def update_recursive(self, leaf_value):
        """
        Backpropagates the value up the tree using logical aggregation.

        Args:
            leaf_value (float): Simulation value for the current node.
        """
        # --- Update leaf node first ---
        self.update(leaf_value)

        # --- If this node has a parent, update it too ---
        if self._parent:
            move_values = []

            for move, child_nodes in self._parent._children.items():
                move_value = aggregate_by_connective(
                    move,
                    child_nodes,
                    mode="value"
                )
                
                
                if move_value is not None:
                    move_values.append(move_value)

            filtered_values = [v for v in move_values if v is not None]
             
            if filtered_values:
                best_value = max(filtered_values)
            else:
                raise ValueError("[update_recursive] No valid move values for parent; cannot backpropagate.")

                
            self._parent.update_recursive(best_value)

    def get_value(self, c_puct):
        """
        Computes PUCT value for this node.

        Args:
            c_puct (float): PUCT exploration parameter.

        Returns:
            float: Value used for action selection (Q + U).
        """
        if self._parent is None:
            return self._Q

        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def get_child_ucb_value(self, child, c_puct):
        """
        Computes UCB value for a child node, used in 2nd-level PUCT.

        Args:
            child (ProofTreeNode): The child node.
            c_puct (float): Exploration constant.

        Returns:
            float: UCB value.
        """
        u = c_puct * child._P * np.sqrt(self._n_visits) / (1 + child._n_visits)
        return child._Q + u

    def is_leaf(self):
        """Returns True if the node has no expanded children."""
        return len(self._children) == 0

    def is_root(self):
        """Returns True if the node has no parent."""
        return self._parent is None

    def is_fully_proved(self):
        """
        Checks whether the node is logically proved, i.e., whether at least one move
        generated only proved subgoals (according to logical rule semantics).
        """
        if len(self._children) == 0:
            return False

        for move, child_nodes in self._children.items():
            if aggregate_by_connective(
                move,
                child_nodes,
                mode="proved"
            ):
                return True

        return False

    def print_tree(self, indent=0):
        """Recursively prints the proof tree for inspection."""
        prefix = "  " * indent
        status = "[proved]" if self.pg_node.num_moves.n == 0 else ""
        print(f"{prefix}- {self.sequent} {status} (Q={self._Q:.2f}, N={self._n_visits})")

        for move, child_nodes in self._children.items():
            print(f"{prefix}  |_ Move: {move}")
            for child in child_nodes:
                child.print_tree(indent + 2)


# --- MCTS Class ---
class MCTS:
    """Monte Carlo Tree Search guided by a policy-value neural network."""

    def __init__(
        self,
        policy_value_fn,
        proof_graph,
        c_puct=c_puct_value,
        n_playout=number_simulations,
        verbose=False,
    ):
        self._root = None  # Root will be set upon first search
        self._policy = policy_value_fn  # Neural net: maps sequent to (priors, value)
        self._proof_graph = proof_graph
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._proof_complete = False
        self.verbose = verbose

    def set_root(self, sequent):
        """Initializes the root of the search tree."""
        if self._root is None:
            root_pg_node = self._proof_graph.root
            self._root = ProofTreeNode(root_pg_node, parent=None, prior_p=1.0)
            if sequent.is_axiom():
                self._proof_graph.set_proved(root_pg_node)
                self._proof_complete = True

    def _uniform_priors(self, move_priors, threshold=1e-6):
        """
        If all prior values are close to zero, fallback to uniform priors.

        This prevents early collapse in case the network is untrained.
        """
        total = sum(p for _, p in move_priors)
        if total < threshold:
            n = len(move_priors)
            return [(m, 1.0 / n) for m, _ in move_priors]
        return move_priors

    def playout(self, sequent):
        """
        Performs one full MCTS simulation (selection - expansion - backprop).

        Returns:
            path: sequence of visited nodes (used for root update)
            proof_complete: whether the root has been proved
        """
        if self._root is None:
            raise ValueError("Root not set. Call `set_root(sequent)` first.")

        if self.verbose: 
            print(f"[MCTS start] root sequent ≡ {self._root.sequent}")

        node = self._root
        path = []

        # --- Selection ---
        while not node.is_leaf():
            move, children = node.select(self._c_puct)

            selected_child = max(
                children,
                key=lambda child: node.get_child_ucb_value(child, self._c_puct)
            )

            path.append((node, move, selected_child))
            node = selected_child

        # --- Evaluation ---
        if node.sequent.is_axiom():
            self._proof_graph.set_proved(node.pg_node)
            leaf_value = 1.0
            node.update_recursive(leaf_value)

            if self._root.pg_node.num_moves.n == 0:
                self._proof_complete = True
            return [], self._proof_complete

        # Expansion with neural guidance
        move_probs, leaf_value = self._policy(node.sequent)
        
        if leaf_value is None or not isinstance(leaf_value, (float, int)):
            print(f"[DEBUG] leaf_value not valid: {leaf_value} per sequent: {node.sequent}")
            return [], False

        move_probs = self._uniform_priors(move_probs)

        node.expand(move_probs, self._proof_graph)

        if self.verbose:
            print(f"[expand] Node: {node.sequent}, Moves: {move_probs}")

        if node.is_fully_proved():
            self._proof_graph.set_proved(node.pg_node)

        # --- Backpropagation ---
        node.update_recursive(leaf_value)

        for ancestor, _, _ in reversed(path):
            if ancestor.is_fully_proved():
                self._proof_graph.set_proved(ancestor.pg_node)
                
        if self._root.is_fully_proved():
            self._proof_graph.set_proved(self._root.pg_node)
            self._proof_complete = True
            if self.verbose:
                print("Root sequent has been proved (via is_fully_proved).")
    
        if self._root.pg_node.num_moves.n == 0:
            self._proof_complete = True
            if self.verbose:
                print("Root sequent has been proved (via num_moves).")

        return path, self._proof_complete

    def get_move_probs(self, sequent, tau=1e-3):
        """
        Performs n simulations and returns softmax-normalized move probabilities.

        Args:
            sequent: Starting sequent
            tau: Temperature for softmax (low → greedier)

        Returns:
             List of (move, prob) tuples: the policy over moves.
        """
        if self._root is None:
            raise ValueError("Root not initialized. Call `set_root(sequent)` first.")

        last_path = None
        for _ in range(self._n_playout):
            last_path, proved = self.playout(sequent)
            if proved and self.verbose:
                print("Complete proof found during MCTS simulation.")
                break

        self._last_path = last_path

        act_visits = [
            (move, max(child._n_visits for child in children))
            for move, children in self._root._children.items()
        ]

        if not act_visits:
            return []


        moves, visits = zip(*act_visits)
        probs = softmax(np.log(np.array(visits) + 1e-10) / tau)

        if self._root.is_fully_proved() or self._proof_graph.root.num_moves.n == 0:
            self._proof_graph.set_proved(self._root.pg_node)
            self._proof_complete = True

        return list(zip(moves, probs))

    def update_with_move(self, last_move):
        """
        Advances the tree after making a move, reusing the corresponding subtree.

        This enables efficiency between consecutive MCTS calls.
        """
        if hasattr(self, "_last_path"):
            for _, move, selected_child in self._last_path:
                if move == last_move:
                    children = self._root._children.get(move, [])
                    for child in children:
                        if child.pg_node.id == selected_child.pg_node.id:
                            self._root = child
                            self._root._parent = None
                            return

        children = self._root._children.get(last_move, [])
        self._root = children[0] if children else None

        if self._root:
            self._root._parent = None


__all__ = ["MCTS", "ProofTreeNode"]