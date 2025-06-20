"""
Contains functionality to manipulate Formula objects
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

from pathlib import Path

# --- Importing Formula Class ---
# Go two levels up: from ICTCS_notebooks → theorem_prover_core → project root
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from theorem_prover_core.formula import (Formula, Letter, Falsity, Conjunction, Disjunction, Implication,
                                         Negation, BinaryConnectiveFormula, UnaryConnectiveFormula, bottom)


# --- Truth Assignment Function ---
# Generate random truth values for letters
def truth_assignment(letters: int, seed: int) -> Dict[int, bool]:
    """
    Randomly assigns True or False to each propositional letter in a formula.

    Args:
        letters (int): An integer that represents the number of propositional letters 
    
    Returns: 
        Dict[int, bool]: A dictonary where keys are integers representing the propositional 
        letters, and values are booleans (True or False), representing the truth value 
        assigned to each variable.
    """
    if seed is not None:
        random.seed(seed)
    return {i: random.choice([True, False]) for i in range(letters)}


# --- Evaluate the truth value of a formula ---
def evaluate_formula(formula: Formula, truth_assignment: Dict[int, bool]) -> bool:
    """
    Evaluates the truth value of a formula using the given truth assignment 
    for propositional letters.

    Args: 
        formula: The propositional formula to evaluate.
        truth_assignment: A dictionary with the propositional letters values.
    
    Returns: 
        bool: A truth value.
    """
    if isinstance(formula, Letter):
        return truth_assignment[formula.n]
    elif isinstance(formula, Falsity):
        return False
    elif isinstance(formula, Negation):
        return not evaluate_formula(formula.formula, truth_assignment)
    elif isinstance(formula, Conjunction):
        return evaluate_formula(formula.left, truth_assignment) and evaluate_formula(formula.right, truth_assignment)
    elif isinstance(formula, Disjunction):
        return evaluate_formula(formula.left, truth_assignment) or evaluate_formula(formula.right, truth_assignment)
    #elif isinstance(formula, ExclusiveDisjunction):
    #    return evaluate_formula(formula.left, truth_assignment) != evaluate_formula(formula.right, truth_assignment)
    elif isinstance(formula, Implication):
        return not evaluate_formula(formula.left, truth_assignment) or evaluate_formula(formula.right, truth_assignment)
    else:
        raise Exception("Unknown formula type")


# --- Derive letters from a given formula ---
def derive_letters(formula: Formula) -> Set[int]:
    """
    Recursively derives the set of propositional letters (by their index) in the given formula.
       
    Args: 
        formula: A propositional formula.
       
    Returns: 
        A set of letters indices. 
    """
    if isinstance(formula, Letter):
        return {formula.n}
    elif isinstance(formula, Falsity):
        return set()
    elif isinstance(formula, UnaryConnectiveFormula):
        return derive_letters(formula.formula)
    elif isinstance(formula, BinaryConnectiveFormula):
        left_letters = derive_letters(formula.left)
        right_letters = derive_letters(formula.right)
        return left_letters.union(right_letters)
    else:
        raise Exception("Unknown formula type")


# --- Check tautology status of a given formula ---
def is_tautology(formula: Formula) -> bool:
    """
    Checks if a given formula is a tautology.

    Args: 
        formula: A propositional formula.
       
    Returns: 
        bool: The tautology status (True or False).
    """
    letters = derive_letters(formula)
    num_letters = len(letters)

    for values in product([False, True], repeat=num_letters):
        assignment = dict(zip(sorted(letters), values))
        if not evaluate_formula(formula, assignment):
            return False

    return True


# --- Generator of letter sequence ---
def generate_letter_sequence(num_letters: int) -> None:
    """"
    Generate sequence of letter wrapping around when reaching 
    the max number of letters.

    Args:
        num_letters (int): Number of total propositional letters to generate.
    """
    current_letter = 0
    while True:
        yield Letter(current_letter)
        current_letter += 1
        if current_letter > num_letters:
            current_letter = 0  


# --- Normalizer Class ---
class Normalizer:
    """
    This class is used to normalize formulas by renaming propositional letters to 
    follow a consistent index order starting from 0 before use them as input.
    Normalization maintains letter consistency, ensuring that the same propositional 
    letter could appears with the same index throughout the formula.

    Example:
        A2 ∧ A7   ->   A0 ∧ A1
        A7 ∧ A2   ->   A0 ∧ A1  (same as above)

    The same original letter index is always mapped to the same normalized index.
    
    """

    def __init__(self):
        self.__dict_letters = {}
        self.__last_letter = 0

    def normalize(self, data: Formula) -> Formula:
        """
        Recursively normalizes a formula so that propositional letters are renamed
        with consistent normalized indices.

        Args:
            data (Formula): The propositional formula to normalize.

        Returns:
            Formula: A new formula with normalized letter indices.

        Raises:
            Exception: If the input formula is of an unknown type.
        """
        
        if isinstance(data, Letter):
            # If the letter has already been encountered, use the same normalized index
            n = data.n
            if n not in self.__dict_letters:
                # Assign a new letter if it hasn't been seen before
                self.__dict_letters[n] = self.__last_letter
                self.__last_letter += 1
            # Return the letter with its normalized index
            return Letter(self.__dict_letters[n])

        elif isinstance(data, Falsity):
            return Falsity()

        elif isinstance(data, UnaryConnectiveFormula):
            formula = self.normalize(data.formula)
            return data.__class__(formula)

        elif isinstance(data, BinaryConnectiveFormula):
            left = self.normalize(data.left)
            right = self.normalize(data.right)
            return data.__class__(left, right)

        else:
            raise Exception("Unknown formula type")


# --- Random Formula Generator ---
def generate_random_formula(max_depth: int = None, 
                            letter_generator = None, 
                            seed :int = None ) -> Formula:
    """
    Generates a random formula of a certain depth.
    
    Args: 
        max_depth (int): Controls the maximum depth of the generated formula's syntax tree.
        letter_generator: A generator of letter sequence. 
        seed (int): Integer for reproducibility.
       
    Returns: 
        A random propositional formula.

    """
      
    if seed is not None:
        random.seed(seed)


    if max_depth == 0 or random.random() > 0.5:
        # If max depth is 0 or by random chance, return either a letter or falsity
        return random.choices([next(letter_generator), Falsity()], weights=[0.95, 0.05])[0]  # Priority given to random letter

    formula_type = random.choice([Conjunction, Disjunction, Implication, Negation])

    if formula_type == Negation:
        subformula = generate_random_formula(max_depth - 1, letter_generator)
        return Negation(subformula)
    else:
        left_subformula = generate_random_formula(max_depth - 1, letter_generator)
        right_subformula = generate_random_formula(max_depth - 1, letter_generator)
        return formula_type(left_subformula, right_subformula)


# --- Generate normalized random formulas ---
def generate_normalized_random_formula(max_depth: int = None, 
                                       num_letters: int = None, 
                                       seed :int = None) -> Formula:
    """
    Generates a normalized random formula where letters are renumbered in ascending order.

    Args: 
        max_depth (int): Controls the maximum depth of the generated formula's syntax tree.
        num_letters (int): Number of propositional letter that can be used.
        seed (int): integer for reproducibility.
      
    Returns: 
        A random normalized formula.
        
    """

    if seed is not None:
        random.seed(seed)

    # Initialize a generator for propositional letters in ascending order
    letter_generator = generate_letter_sequence(num_letters)

    # Generate a random formula
    random_formula = generate_random_formula(max_depth, letter_generator, seed=seed)

    # Normalize the formula using the Normalizer class
    normalizer = Normalizer()
    normalized_formula = normalizer.normalize(random_formula)

    return normalized_formula


# --- Checking if a formula is normalized --- 
def is_normalized(formula: Formula) -> bool:
    """
    Checks if a given formula is normalized.
    
    """
    encountered_letters = set()
    last_index = -1  # Initialize to an invalid index to check ascending order
                     # when the first letter is checked, its index will always be greater than -1

    def check_formula(f: Formula):
        nonlocal last_index
        if isinstance(f, Letter):
            # Check if the letter index is in ascending order
            if f.n in encountered_letters:
                return True  # Already seen, valid
            if f.n <= last_index:
                return False  # Not in ascending order
            encountered_letters.add(f.n)
            last_index = f.n  # Update last_index to current letter's index
            return True

        elif isinstance(f, Falsity):
            return True

        elif isinstance(f, UnaryConnectiveFormula):
            return check_formula(f.formula)

        elif isinstance(f, BinaryConnectiveFormula):
            return check_formula(f.left) and check_formula(f.right)

        else:
            raise Exception("Unknown formula type")

    return check_formula(formula)


# --- Metavariable and Instantiation classes ---
Position = Literal["left", "right"]  
Associativity = Literal["left", "right"]  

@dataclass(frozen=True)
class Metavariable(Formula):
    """
    A class representing a metavariable in a formula.
    Metavariables will be replaced with actual formulas during instantiation.
    """
    __slots__ = ('name',)
    name: str

    def _make_str(self, outer_class: Union[Type[Formula], None], position: Union[Position, None]) -> str:
        return self.name

    def _make_fully_parenthesized_str(self) -> str:
        return self.name  


# --- Metavariables Instantiator ---
class Instantiator:
    """
    This class handles the instantiation of metavariables within formulas.

    """
    def __init__(self, num_letters: int):
        self.num_letters = num_letters
        self.letter_generator = generate_letter_sequence(num_letters)
        self.normalizer = Normalizer()

    def instantiate(self, formula: Formula, metavariable_map: Dict[str, Formula]) -> Formula:
        """
        Recursively replaces metavariables in a formula with actual formulas from the map.

        """
        if isinstance(formula, Metavariable):
            if formula.name not in metavariable_map:
                raise Exception(f"Metavariable '{formula.name}' not found in the map.")
            return self.normalizer.normalize(metavariable_map[formula.name])

        elif isinstance(formula, Letter):
            return formula

        elif isinstance(formula, Falsity):
            return formula

        elif isinstance(formula, UnaryConnectiveFormula):
            instantiated_formula = self.instantiate(formula.formula, metavariable_map)
            return formula.__class__(instantiated_formula)

        elif isinstance(formula, BinaryConnectiveFormula):
            left_instantiated = self.instantiate(formula.left, metavariable_map)
            right_instantiated = self.instantiate(formula.right, metavariable_map)
            return formula.__class__(left_instantiated, right_instantiated)

        else:
            raise Exception("Unknown formula type")


# --- Common Tautologies Istantiator --- 
def instantiate_random_formulas(num_samples: int, 
                                max_depth: int,
                                num_letters: int,
                                tautologies: List[Formula], 
                                seed: int = None) -> List[Formula]:
    """
    Generate a list of unique instantiated tautological formulas by randomly substituting
    metavariables in known tautologies with normalized randomly generated subformulas.

    Each tautology may contain metavariables like 'A', 'B', 'C', which are replaced with
    randomly generated formulas. The result is normalized to ensure consistent variable naming.

    Args:
        num_samples (int): The number of unique instantiated tautologies to generate.
        max_depth (int): Maximum depth for randomly generated subformulas.
        num_letters (int): Number of distinct propositional letters to use in random generation.
        tautologies (List[Formula]): A list of tautology templates (with metavariables A, B, C).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        List[Formula]: A list of unique instantiated and normalized tautological formulas.
    
    Notes:
        - If the required number of unique formulas can't be generated within the allowed attempts,
          a warning is printed and fewer formulas may be returned.
        - Each formula is normalized to canonicalize variable indices.
    """
    if seed is not None:
        random.seed(seed)

    instantiated_tautologies = []
    seen_formulas = set()
    instantiator = Instantiator(num_letters=num_letters)
    normalizer = Normalizer()

    attempts = 0
    max_attempts = num_samples * 5  # Increased to ensure enough unique samples

    while len(instantiated_tautologies) < num_samples and attempts < max_attempts:
        try:
            random_formulas = [
                generate_normalized_random_formula(max_depth=max_depth, 
                                                   num_letters=num_letters, 
                                                   seed=(seed + attempts + i))
                for i in range(3)
            ]
        except Exception:
            attempts += 1
            continue

        metavariable_map = {
            "A": random_formulas[0],
            "B": random_formulas[1],
            "C": random_formulas[2]
        }

        tautology = tautologies[attempts % len(tautologies)]

        try:
            instantiated_formula = instantiator.instantiate(tautology, metavariable_map)
            instantiated_formula = normalizer.normalize(instantiated_formula)
            instantiated_str = instantiated_formula.to_fully_parenthesized_str()

            if instantiated_str not in seen_formulas:
                instantiated_tautologies.append(instantiated_formula)
                seen_formulas.add(instantiated_str)
        except Exception:
            pass

        attempts += 1

    if len(instantiated_tautologies) < num_samples:
        print(f"[Warning] Only {len(instantiated_tautologies)} unique formulas generated out of {num_samples} requested.")

    return instantiated_tautologies



# ---Parses a string representation of a formula ---
def tokenize(formula: str) -> List[str]:
    """
    Splits a logical formula string into a list of tokens.

    Supported tokens:
        - Propositional letters: A0, A1, ...
        - Connectives: ¬, ∧, ∨, ⊻, →
        - Falsity: ⊥
        - Parentheses: (,)

    Args:
        formula (str): A logical formula in string format.

    Returns:
        List[str]: A list of string tokens (e.g. ['¬', '(', 'A0', '∧', 'A1', ')'])

    Raises:
        ValueError: If the formula contains an unrecognized character.  
    """
    tokens = []
    i = 0
    # Iterate over each character
    while i < len(formula):
        c = formula[i]
        # Skip whitespace
        if c.isspace():
            i += 1
        elif c in '()¬∧∨⊻':
            tokens.append(c)
            i += 1
        elif formula[i:i+1] == '→':  
            tokens.append('→')
            i += 1
        elif c == 'A':
            j = i + 1
            while j < len(formula) and formula[j].isdigit():
                j += 1
            tokens.append(formula[i:j])
            i = j
        elif c == '⊥':
            tokens.append('⊥')
            i += 1
        else:
            raise ValueError(f"Unexpected character: {c}")
    return tokens


# --- Formula Parser ---
def parse_formula_string(formula_string: str) -> Formula:
    """
    Parses a fully parenthesized string representation of a propositional logic formula.

    Only accepts formulas where all operations are explicitly parenthesized.

    Args:
        formula_string (str): The input formula string.

    Returns:
        Formula: The corresponding Formula object.

    Raises:
        ValueError: If the input is not well-formed or not fully parenthesized.
    """
    
    tokens = tokenize(formula_string)

    def parse(tokens: List[str]) -> Formula:
        if not tokens:
            raise ValueError("Unexpected end of input")

        token = tokens.pop(0)

        if token == '(':
            if not tokens:
                raise ValueError("Unexpected end after '('")
            
            # Check for unary negation
            if tokens[0] == '¬':
                tokens.pop(0)
                subformula = parse(tokens)
                if not tokens or tokens.pop(0) != ')':
                    raise ValueError("Expected closing ')' after negation")
                return Negation(subformula)

            # Otherwise, must be binary operation
            left = parse(tokens)
            if not tokens:
                raise ValueError("Expected binary operator")
            op = tokens.pop(0)
            right = parse(tokens)
            if not tokens or tokens.pop(0) != ')':
                raise ValueError("Expected closing ')' after binary formula")

            if op == '∧':
                return Conjunction(left, right)
            elif op == '∨':
                return Disjunction(left, right)
            elif op == '⊻':
                return ExclusiveDisjunction(left, right)
            elif op == '→':
                return Implication(left, right)
            else:
                raise ValueError(f"Unknown operator: {op}")

        elif token.startswith('A') and token[1:].isdigit():
            return Letter(int(token[1:]))
        elif token == '⊥':
            return Falsity()
        else:
            raise ValueError(f"Unexpected token: {token}")

    result = parse(tokens)
    if tokens:
        raise ValueError(f"Extra tokens after parsing: {tokens}")
    return result


# --- Convert symbols in numbers ---
class CustomTokenizer:
    """
    Custom tokenizer class for logical formulas.
    This class converts formulas into tokenized integer representations,
    and supports detokenizing back into Formula objects.
    """

    def __init__(self):
        self.token_to_formula: Dict[int, Formula] = {}
        self.formula_to_token: Dict[Formula, int] = {}

        self.connective_map = {
            'Conjunction': 100,
            'Disjunction': 101,
            'Negation': 102,
            'Implication': 103,
            #'Exclusive Disjunction': 104
        }

        self.special_map = {
            '(': 106,
            ')': 107
        }

        self.falsity_token = 105

    def fit(self, formulas: List[Formula]):
        """
        Fit the tokenizer on a list of formulas, deriving tokens for each formula.
        """
        for formula in formulas:
            self._derive_tokens(formula)

    def _derive_tokens(self, formula: Formula):
        """
        Recursively derive tokens for all subcomponents of a formula.
        """
        if isinstance(formula, Falsity):
            if formula not in self.formula_to_token:
                self.formula_to_token[formula] = self.falsity_token
                self.token_to_formula[self.falsity_token] = formula

        elif isinstance(formula, Letter):
            letter_index = formula.n
            if formula not in self.formula_to_token:
                token = letter_index + 1  # Start letters from 1
                self.formula_to_token[formula] = token
                self.token_to_formula[token] = formula

        elif isinstance(formula, UnaryConnectiveFormula):
            self._derive_tokens(formula.formula)

        elif isinstance(formula, BinaryConnectiveFormula):
            self._derive_tokens(formula.left)
            self._derive_tokens(formula.right)

    def tokenize(self, formula: Formula) -> List[int]:
        """
        Convert a formula into a list of integer tokens.
        """
        tokens = []
        self._tokenize_helper(formula, tokens)
        return tokens

    def _tokenize_helper(self, formula: Formula, tokens: List[int]):
        """
        Helper method for recursive token generation.
        """
        if formula in self.formula_to_token:
            tokens.append(self.formula_to_token[formula])
            return

        if isinstance(formula, BinaryConnectiveFormula):
            tokens.append(self.special_map['('])
            self._tokenize_helper(formula.left, tokens)
            tokens.append(self.connective_map[type(formula).__name__])
            self._tokenize_helper(formula.right, tokens)
            tokens.append(self.special_map[')'])

        elif isinstance(formula, UnaryConnectiveFormula):
            tokens.append(self.special_map['('])
            tokens.append(self.connective_map[type(formula).__name__])
            tokens.append(self.special_map['('])
            self._tokenize_helper(formula.formula, tokens)
            tokens.append(self.special_map[')'])
            tokens.append(self.special_map[')'])

        elif isinstance(formula, Falsity):
            tokens.append(self.falsity_token)

        elif isinstance(formula, Letter):
            tokens.append(self.formula_to_token[formula])

    def detokenize(self, tokens: List[int]) -> Formula:
        """
        Convert a list of tokens (possibly padded) back into a Formula object.
        """
        # Remove trailing padding
        tokens = [t for t in tokens if t != 0]

        def parse_expr(pos: int) -> Tuple[Formula, int]:
            token = tokens[pos]

            if token == self.falsity_token:
                return Falsity(), pos + 1

            elif token in self.token_to_formula:
                return self.token_to_formula[token], pos + 1

            elif token == self.special_map['(']:
                next_token = tokens[pos + 1]

                # Handle unary connective
                if next_token in self.connective_map.values() and tokens[pos + 2] == self.special_map['(']:
                    op_token = next_token
                    inner_formula, new_pos = parse_expr(pos + 3)
                    assert tokens[new_pos] == self.special_map[')']
                    assert tokens[new_pos + 1] == self.special_map[')']
                    connective_class = self._connective_class(op_token)
                    return connective_class(inner_formula), new_pos + 2

                # Handle binary connective
                else:
                    left_formula, pos_left = parse_expr(pos + 1)
                    op_token = tokens[pos_left]
                    right_formula, pos_right = parse_expr(pos_left + 1)
                    assert tokens[pos_right] == self.special_map[')']
                    connective_class = self._connective_class(op_token)
                    return connective_class(left_formula, right_formula), pos_right + 1

            raise ValueError(f"Unexpected token at position {pos}: {tokens[pos:]}")

        return parse_expr(0)[0]

    def _connective_class(self, token: int):
        """
        Resolve token ID back to the appropriate connective class.
        """
        for name, code in self.connective_map.items():
            if token == code:
                return {
                    'Conjunction': Conjunction,
                    'Disjunction': Disjunction,
                    'Negation': Negation,
                    'Implication': Implication,
                    #'Exclusive Disjunction': ExclusiveDisjunction
                }[name]
        raise ValueError(f"Unknown connective token: {token}")


# --- Formula syntax tree ---
class FormulaTreeNode:
    """
    Represents a node in the binary syntax tree of a logical formula.

    Each node contains a formula object and recursively stores its children,
    corresponding to the structure of the formula.

    Attributes:
        formula (Formula): The formula (or subformula) represented by this node.
        children (List[FormulaTreeNode]): Child nodes (1 for unary, 2 for binary connectives).
        embedding_index (Optional[int]): Token index for embedding (set externally).
    """
    def __init__(self, formula):
        self.formula = formula
        self.children: List[FormulaTreeNode] = []
        self.embedding_index: Optional[int] = None  # Set by tokenizer externally

        # Recursively build the tree
        if isinstance(formula, UnaryConnectiveFormula):
            self.children.append(FormulaTreeNode(formula.formula))

        elif isinstance(formula, BinaryConnectiveFormula):
            self.children.append(FormulaTreeNode(formula.left))
            self.children.append(FormulaTreeNode(formula.right))
        
        # Leaves (e.g., Letter, Falsity) naturally have no children

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self):
        return f"Node({self.formula}, children={len(self.children)})"


# --- Embedding tree node ---
def assign_embedding_indices(node: FormulaTreeNode, tokenizer: CustomTokenizer):
    """
    Recursively assigns an embedding index to each node in a FormulaTreeNode tree,
    based on a pre-fitted tokenizer.

    Each node in the tree corresponds to a subformula. This function uses the tokenizer
    to map each subformula (or its type) to a unique integer index, which will be used
    for embedding lookup during model input preparation.

    Args:
        node (FormulaTreeNode): The root node of the formula tree to process.
        tokenizer (CustomTokenizer): A tokenizer that has already been fitted on a set
                                     of formulas and provides formula-to-token mappings.

    Raises:
        ValueError: If a Letter is not found in the tokenizer,
                    or if the formula type is unknown and cannot be handled.
    """
    
    if node.formula in tokenizer.formula_to_token:
        node.embedding_index = tokenizer.formula_to_token[node.formula]
    else:
        # fallback: try to assign based on type (for connectives)
        if isinstance(node.formula, Falsity):
            node.embedding_index = tokenizer.falsity_token
        elif isinstance(node.formula, Letter):
            # If it is a letter but it is not in the tokenizer, it raises an error
            raise ValueError(f"Using an unknown letter is not accepted: {node.formula}")
        elif isinstance(node.formula, UnaryConnectiveFormula):
            node.embedding_index = tokenizer.connective_map[type(node.formula).__name__]
        elif isinstance(node.formula, BinaryConnectiveFormula):
            node.embedding_index = tokenizer.connective_map[type(node.formula).__name__]
        else:
            raise ValueError(f"Unknown formula: {node.formula}")

    for child in node.children:
        assign_embedding_indices(child, tokenizer)


# --- Function: Print tree with embedding indexes ---
def print_tree_with_embeddings(node: FormulaTreeNode, prefix: str = "", is_last: bool = True):
    """
    Recursively prints the formula tree structure along with each node's embedding index.

    The output is formatted as a visual tree using connectors (├──, └──),
    and includes:
        - the type of formula (e.g., Letter, And, Not, etc.)
        - the formula string
        - the embedding index assigned to that node

    Args:
        node (FormulaTreeNode): The current node in the formula tree to print.
        prefix (str): Internal prefix used to format tree depth and branches (used recursively).
        is_last (bool): Whether the current node is the last child of its parent (used for formatting).
    """
    
    connector = "└── " if is_last else "├── "
    formula_str = str(node.formula)
    formula_type = type(node.formula).__name__
    print(f"{prefix}{connector}[{formula_type}] {formula_str} (embedding_index={node.embedding_index})")

    # Preparazione del prefisso per i figli
    new_prefix = prefix + ("    " if is_last else "│   ")
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        is_child_last = i == (child_count - 1)
        print_tree_with_embeddings(child, new_prefix, is_child_last)
        