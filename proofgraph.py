"""
This module implements the `ProofGraph` class, which represents a state in the search space of the proof.
It builds a hypergraph of sequents, tracking rule applications and proof progress.
"""

from dataclasses import dataclass, field
from typing import NamedTuple, Union, List
from uuid import uuid4

from .formula import *
from .sequent import *
from .mcts import aggregate_by_connective


# --- intinf: Integer or Infinity for tracking proof cost ---
@dataclass(frozen=True)
class intinf:
    """
    Represents an integer or positive infinity (∞).
    Used to count the number of steps to a proof.
    """

    n: Union[int, None] = None

    # Less-than comparison
    def __lt__(self, other: "intinf") -> bool:  
        if self.n is None:
            return False  
        elif other.n is None:
            return True  
        else:
            return self.n < other.n
    
    # Minimum between two intinf values
    def __min__(self, other: "intinf") -> "intinf":  
        if self.n is None:
            return other
        if other.n is None:
            return self
        return intinf(min(self.n, other.n))
    
    # Addition of two intinf values
    def __add__(self, other: "intinf") -> "intinf":  
        if self.n is None:
            return self
        if other.n is None:
            return other
        return intinf(self.n + other.n)

    def __str__(self) -> str:  
        return "∞" if self.n is None else str(self.n)

# --- ProofGraph Class ---
class ProofGraph:
    
    # --- Nestes Class:  ProofGraph.HyperEdge ---
    class HyperEdge(NamedTuple):
        """
        Represents a move applied to a node, producing multiple child nodes (subgoals).
        Each edge stores the applied rule and is resulting sequents.
        """

        move: Sequent.Move  
        sequents: List["ProofGraph.Node"]  
    
    # --- Nested Class: ProofGraph.Node ---
    @dataclass
    class Node:
        """
        A node in the proof graph. Represents a sequent to prove.
        Links to parents and children for graph traversal and updating.
        """

        sequent: Sequent  
        parents: List["ProofGraph.Node"] = field(
            default_factory=list
        )  
        children: List["ProofGraph.HyperEdge"] = field(
            default_factory=list
        )  
        num_moves: intinf = intinf()  
        id: str = field(
            default_factory=lambda: str(uuid4()), init=False
        )  

#        """DESIGN NOTE: see proofgraph_class_doc"""

        def __str__(self) -> str:
            return f"{self.sequent} ({self.num_moves})"

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            return isinstance(other, ProofGraph.Node) and self.id == other.id

    def __init__(self, sequent: Sequent):
        self.root = ProofGraph.Node(sequent)
        self._nodes_by_sequent = {sequent: self.root}  
    
    def node_of(self, sequent: Sequent) -> Node:
        if sequent in self._nodes_by_sequent:
            return self._nodes_by_sequent[sequent]
        
        node = ProofGraph.Node(sequent=sequent)
        self._nodes_by_sequent[sequent] = node
        return node

    def recompute_moves(self, nodes: list["ProofGraph.Node"]):
        """Recalculates num_moves – the cost to prove – for a node and propagate changes to parents."""
        while nodes:
            current = nodes.pop()
            
            if current.children:
                min_cost = intinf()

                for edge in current.children:
                    # Compute the aggregated cost using logical semantics
                    value = aggregate_by_connective(
                        move=edge.move,
                        child_nodes=edge.sequents,
                        mode="value",
                        parent_sequent=current.sequent
                    )
                    
                    if value is None:
                        continue  # Skip moves that cannot be completed
                        
                    # +1 because it is an inference step
                    total_cost = intinf(value + 1)
                    min_cost = min(min_cost, total_cost)
                
                new_num_moves = min_cost
            else:
                new_num_moves = intinf()  # No applicable move: infinite cost

            if new_num_moves != current.num_moves:
                current.num_moves = new_num_moves
                nodes.extend(current.parents)


    def add_children(
        self, parent: Node, move: Sequent.Move, children: list[Sequent]
    ) -> list[Node]:
        """
        Applies a move to a parent node, generating new child nodes (subgoals).
        Adds a HyperEdge and updates move counts.
        """
        nodes = [self.node_of(s) for s in children]  
        for n in nodes:
            n.parents.append(parent)  
        parent.children.append(ProofGraph.HyperEdge(move, nodes))  
        self.recompute_moves([parent])  
        return nodes

    def set_proved(self, node):
        """Marks a node as proved and propagates the result upwards."""

        if node.num_moves.n == 0:
            return  # Already proved

        # Base case: axiom
        if not node.children:
            if node.sequent.is_axiom():
                node.num_moves = intinf(0)
                for parent in node.parents:
                    self.set_proved(parent)
            return
        
        # Check whether at least one rule has all its children proved
        for move, children in node.children:
            if aggregate_by_connective(move, children, mode="proved", parent_sequent=node.sequent):
                node.num_moves = intinf(0)
                for parent in node.parents:
                    self.set_proved(parent)
                return


    def to_sequent_calculus(self) -> str:
        """Converts the proof graph into a sequent calculus derivation tree."""

        def escape_dot(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        # Converts a Sequent object to a string format: [A, B, ...] ⊢ C
        def format_sequent(sequent: Sequent) -> str:
            premises = ", ".join(str(p) for p in sequent.premises)
            return f"[{premises}] ⊢ {sequent.conclusion}"

        # Assigns a rule label and whether it's left or right.
        def rule_label(sequent: Sequent, move: Sequent.Move) -> str:
            if move.pos == -1:
                f = sequent.conclusion
            elif 0 <= move.pos < len(sequent.premises):
                f = sequent.premises[move.pos]
            else:
                return "..."
            if isinstance(f, Conjunction):
                return "∧L" if move.pos >= 0 else "∧R"
            if isinstance(f, Disjunction):
                return "∨L" if move.pos >= 0 else "∨R"
            if isinstance(f, Implication):
                return "→L" if move.pos >= 0 else "→R"
            if isinstance(f, Negation):
                return "¬L" if move.pos >= 0 else "¬R"
            return "Ax"

        def style_rule(rule: str) -> str:
            return f"[{rule}]"

        # Start the graph.
        lines = [
            "digraph SequentCalc {",
            "  rankdir=TB;",
            '  node [shape=box, fontsize=12, fontname="Latin Modern Roman"];',
            "  edge [arrowhead=none];",
        ]

        visited = set()  
        id_counter = [0]  

        # Assigns or retrieves a unique ID for a node
        def get_id(node):
            if not hasattr(node, "_dot_id"):
                node._dot_id = f"n{id_counter[0]}"
                id_counter[0] += 1
            return node._dot_id

        # Recursively walks the proof tree, starting from the root,
        # adding nodes and edges.
        def walk(node):
            if node in visited:
                return
            visited.add(node)

            nid = get_id(node)

            # Label: Sequent + number of moves.
            label_text = f"{format_sequent(node.sequent)}      ({node.num_moves})"
            label = escape_dot(label_text)
            # Add the current sequent as a node
            lines.append(f'  {nid} [label="{label}", color="black"];')

            for edge in node.children:
                # Create a node for the rule application
                rule_id = f"rule_{nid}_{id_counter[0]}"
                id_counter[0] += 1
                rule_name = style_rule(rule_label(node.sequent, edge.move))
                lines.append(
                    f'  {rule_id} [label="{rule_name}", shape=plaintext, fontname="monospace"];'
                )
                # Connect rule node to the parent sequent node
                lines.append(f"  {rule_id} -> {nid};")

                child_ids = []
                for child in edge.sequents:
                    cid = get_id(child)
                    child_ids.append(cid)
                    # Connect each child sequent to the rule node
                    lines.append(f"  {cid} -> {rule_id};")
                    # Recursively walk the child node
                    walk(child)

                # Align premises horizontally
                if len(child_ids) > 1:
                    lines.append(f'  {{ rank = same; {"; ".join(child_ids)}; }}')

        # Start walking from the root of the proof graph
        walk(self.root)
        # Close the graph
        lines.append("}")
        # Return the graph as a single string
        return "\n".join(lines)


__all__ = ["ProofGraph", "intinf"]