"""
This module implements the `Formula` class, which is the base class for all formulas in the propositional logic.
It defines both the logical structure and the printing mechanics.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, Union, Type, Iterable


# --- Position and Associativity helpers ---
Position = Literal["left", "right"]  
Associativity = Literal["left", "right"]  



# --- Base Class: Formula ---            
class Formula(metaclass=ABCMeta):
    """Abstract base class for all formulas in propositional logic."""

    __slots__ = ()

    priority: ClassVar[int]
    """
    An integer that represents the priority of the formula. 
    The higher the number, the higher the priority.
    """

    @abstractmethod
    def _make_str(
        self, outer_class: Union[Type["Formula"], None], position: Union[Position, None]
    ) -> str:
        """
        This method is used to generate the string printing representation of the formula.
        """
        pass
        
    def to_fully_parenthesized_str(self) -> str:
        return self._make_fully_parenthesized_str()

    @abstractmethod
    def _make_fully_parenthesized_str(self) -> str:
        pass

    # --- Logical connectives using operator overloading ---
    def __and__(self, f: "Formula") -> "Formula":
        return Conjunction(self, f)

    def __or__(self, f: "Formula") -> "Formula":
        return Disjunction(self, f)

    def __xor__(self, f: "Formula") -> "Formula":
        return ExclusiveDisjunction(self, f)

    def __invert__(self) -> "Formula":
        return Negation(self)

    def implies(self, f: "Formula") -> "Formula":
        return Implication(self, f)

    def __str__(self) -> str:
        return self._make_str(None, "left")
    """
    This method returns the string representation of the formula by calling the _make_str method.
    """

    @staticmethod
    def from_dimacs(iter: Iterable[str]) -> 'Formula':
        """
        This method is used to create a formula expressed in DIMACS CNF format.
        """
        spec_vars = None
        spec_clauses = None
        num_clauses = 0
        letters = []
        formula = None
        clause = None
        header = True

        for line in iter:
            line = line.strip()

            # Skip comments, empty lines, % symbols, and stray zeros
            if not line or line.startswith('c') or line.startswith('%'):
                continue

            # Handle problem line
            if header and line.startswith('p cnf'):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError('Invalid DIMACS CNF format: malformed problem line')
                spec_vars, spec_clauses = int(parts[2]), int(parts[3])
                letters = [Letter(i) for i in range(spec_vars)]
                header = False
                continue

            # Handle clause lines
            values = line.split()
            for value in values:
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError(f'Invalid DIMACS CNF format: non-integer value "{value}"')

                if value == 0:
                    if clause is not None:
                        formula = clause if formula is None else formula & clause
                        num_clauses += 1
                        clause = None
                    continue

                if abs(value) > len(letters):
                    raise ValueError('Invalid DIMACS CNF format: variable index out of range')

                literal = letters[abs(value) - 1]
                if value < 0:
                    literal = ~literal
                clause = literal if clause is None else clause | literal

        # Final clause check
        if clause is not None:
            formula = clause if formula is None else formula & clause
            num_clauses += 1

        if formula is None:
            raise ValueError('Invalid DIMACS CNF format: no clauses found')

        if spec_clauses is not None and num_clauses != spec_clauses:
            print(f"[Warning] Clauses counted: {num_clauses}, specified: {spec_clauses}")

        return formula
  


# --- Atomic Formulas ---                 
@dataclass(frozen=True)
class Letter(Formula):
    """Represents a propositional letter."""

    __slots__ = ("n",)

    n: int

    def _make_str(
        self, outer_class: Union[Type[Formula], None], position: Union[Position, None]
    ) -> str:
        return f"A{self.n}"

    def _make_fully_parenthesized_str(self) -> str:
        return f"A{self.n}"
    
    def __eq__(self, other):
        return isinstance(other, Letter) and self.n == other.n

    def __hash__(self):
        return hash(("Letter", self.n))


    
@dataclass(frozen=True)
class Falsity(Formula):
    """This class represents the falsity constant."""

    __slots__ = ()

    def _make_str(
        self, outer_class: Union[Type[Formula], None], position: Union[Position, None]
    ) -> str:
        return "⊥"
    
    def _make_fully_parenthesized_str(self) -> str:
        return "⊥"

    def __eq__(self, other):
        return isinstance(other, Falsity)

    def __hash__(self):
        return hash("Falsity")


# --- Unary Connectives ---              
@dataclass(frozen=True)
class UnaryConnectiveFormula(Formula):
    """ Base class for formulas with one operand, i.e. negation."""

    __slots__ = ("formula",)

    formula: Formula

    symbol: ClassVar[str]

    def _make_str(
        self, outer_class: Union[Type[Formula], None], position: Union[Position, None]
    ) -> str:
        s = f"{self.symbol}{self.formula._make_str(self.__class__, None)}"
        if outer_class is None:
            return s
        parenthesis = outer_class.priority > self.priority
        return f"({s})" if parenthesis else s

    def _make_fully_parenthesized_str(self) -> str:
        return f"({self.symbol}{self.formula._make_fully_parenthesized_str()})"

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            self.formula == other.formula
        )

    def __hash__(self):
        return hash((self.__class__.__name__, self.formula))



# --- Subclass for negation ---
class Negation(UnaryConnectiveFormula):
    """Represents negation."""

    __slots__ = ()
    priority = 30
    symbol = "¬"



# --- Binary Connectives ---                  
@dataclass(frozen=True)
class BinaryConnectiveFormula(Formula):
    """Base class for formulas with two operands."""

    __slots__ = ("left", "right")
    left: Formula
    right: Formula

    symbol: ClassVar[str]
    commutativity: ClassVar[bool]
    associativity: ClassVar[Associativity]

    def _make_str(
        self, outer_class: Union[Type[Formula], None], position: Union[Position, None]
    ) -> str:
        lefts = self.left._make_str(self.__class__, "left")
        rights = self.right._make_str(self.__class__, "right")
        s = f"{lefts} {self.symbol} {rights}"
        if outer_class is None:
            return s
        parenthesis = (outer_class.priority > self.priority) or (
            outer_class.priority == self.priority
            and (
                outer_class != self.__class__
                or (self.associativity != position and not self.commutativity)
            )
        )
        return f"({s})" if parenthesis else s

    def _make_fully_parenthesized_str(self) -> str:
        left = self.left._make_fully_parenthesized_str()
        right = self.right._make_fully_parenthesized_str()
        return f"({left} {self.symbol} {right})"

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            self.left == other.left and self.right == other.right
        )

    def __hash__(self):
        return hash((self.__class__.__name__, self.left, self.right))



# --- Subclasses for specific binary connectives ---
class Conjunction(BinaryConnectiveFormula):
    """Represents conjunction."""

    __slots__ = ()
    priority = 20
    symbol = "∧"
    associativity = "left"
    commutativity = True


class Disjunction(BinaryConnectiveFormula):
    """Represents disjunction."""

    __slots__ = ()
    priority = 20
    symbol = "∨"
    associativity = "left"
    commutativity = True


class Implication(BinaryConnectiveFormula):
    """Represents implication."""

    __slots__ = ()
    priority = 10
    symbol = "→"
    associativity = "right"
    commutativity = False


class ExclusiveDisjunction(BinaryConnectiveFormula):
    """Represents exclusive disjunction."""

    __slots__ = ()
    priority = 20
    symbol = "⊻"
    associativity = "left"
    commutativity = True


# --- Constant: bottom ---
bottom: Final[Falsity] = Falsity()
"""
The falsity constant.
"""

__all__ = [
    "Formula",
    "Letter",
    "Falsity",
    "Conjunction",
    "Disjunction",
    "Implication",
    "Negation",
    "BinaryConnectiveFormula",
    "UnaryConnectiveFormula",
    "bottom",
]
