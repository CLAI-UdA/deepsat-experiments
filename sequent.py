"""
This module implements the `Sequent` class, which represents a sequent in the intuitionistic propositional logic.
"""

from dataclasses import dataclass
from typing import Literal, Union

from .formula import *



# --- Sequent Class ---                  
@dataclass(frozen=True)
class Sequent:
    """Represents a sequent with a list of premises and a single conclusion."""

    premises: tuple[Formula, ...]
    conclusion: Formula


    # --- Nested Class: Sequent.Move             
    @dataclass(frozen=True)
    class Move:
        """Represents a possible proof step: inference rule application."""
        

        pos: int  
        # -1: apply the move to the conclusion
        # 0 to len(premises)-1 : apply the move to a premise

        param: Literal["left", "right"]  

    def __str__(self) -> str:
        return f'{", ".join(map(str, self.premises))} ⊢ {self.conclusion}'

    def __eq__(self, other):
        if not isinstance(other, Sequent):
            return False
        return self.premises == other.premises and self.conclusion == other.conclusion
    
    def __hash__(self):
        return hash((self.premises, self.conclusion))


    def is_axiom(self) -> bool:
        """
        Returns True if the sequent is an axiom, i.e., 
        the conclusion is in the premises or the premises contain `⊥`.
        """
        return self.conclusion in self.premises or bottom in self.premises

    def moves(self) -> list[Move]:
        """
        Returns the list of all applicable moves to the sequent. Each move is represented by a `Move` object:
        - Move(-1, 'right'): apply right-side rule on conclusion, choose right formula
        - Move(-1, 'left') : apply right-side rule on conclusion, choose left formula
        - Move(i, 'left')  : apply left-side rule to a premise (param ignored)
        REMARK: param is only meaningful when pos == -1 and conclusion is a disjunction.
        """
        moves = []

        # Add moves on conclusion
        c = self.conclusion
        if isinstance(c, Disjunction):
            # ∨R: two different applications, depending on param
            moves.append(Sequent.Move(-1, "left"))
            moves.append(Sequent.Move(-1, "right"))
        elif isinstance(c, (Conjunction, Implication, Negation)):
            # Only one rule can be applied, param is ignored
            moves.append(Sequent.Move(-1, "right"))  # arbitrary, consistent choice
        elif self.is_axiom():
            # Axioms do not generate any moves
            pass

        # Add moves on premises
        for i in range(len(self.premises)):
            moves.append(Sequent.Move(i, "left"))  # param not used

        return moves

    def rule(self, move: Move) -> Union[list["Sequent"], None]:
        """
        Applies a logical rule corresponding to the move.
        Returns the resulting list of new sequents (premises of the rule).
        """
        pos, param = move.pos, move.param
        if pos < 0:
            # Apply rule to the conclusion
            conclusion = self.conclusion
            if isinstance(conclusion, Conjunction):
                # ∧-Right: Γ ⊢ A ∧ B  ~>  Γ ⊢ A  and  Γ ⊢ B
                return [
                    Sequent(self.premises, conclusion.left),
                    Sequent(self.premises, conclusion.right),
                ]

            elif isinstance(conclusion, Disjunction):
                # ∨-Right: Γ ⊢ A ∨ B  ~>  Γ ⊢ A or Γ ⊢ B (based on param)
                chosen = conclusion.left if param == "left" else conclusion.right
                return [Sequent(self.premises, chosen)]

            elif isinstance(conclusion, Implication):
                # →-Right: Γ ⊢ A → B  ~>  Γ, A ⊢ B
                return [Sequent(self.premises + (conclusion.left,), conclusion.right)]

            elif isinstance(conclusion, Negation):
                # ¬-Right: Γ ⊢ ¬A  ~>  Γ, A ⊢ ⊥
                return [Sequent(self.premises + (conclusion.formula,), bottom)]
            else:
                return None
        else:
            # Rule applies to a premise
            if pos >= len(self.premises):
                return None  # Invalid position

            formula = self.premises[pos]
            before = self.premises[:pos]
            after = self.premises[pos + 1 :]

            if isinstance(formula, Conjunction):
                # ∧-Left: A ∧ B, Γ ⊢ C  ~>  A, B, Γ ⊢ C
                return [
                    Sequent(
                        before + (formula.left, formula.right) + after, self.conclusion
                    )
                ]

            elif isinstance(formula, Disjunction):
                # ∨-Left: A ∨ B, Γ ⊢ C  ~>   A, Γ ⊢ C  and  B, Γ ⊢ C
                return [
                    Sequent(before + (formula.left,) + after, self.conclusion),
                    Sequent(before + (formula.right,) + after, self.conclusion),
                ]

            elif isinstance(formula, Implication):
                # →-Left: A → B, Γ ⊢ C  ~>   A → B, Γ ⊢ A  and   B, Γ ⊢ C
                return [
                    Sequent((formula,) + before + after, formula.left),
                    Sequent(before + (formula.right,) + after, self.conclusion),
                ]

            elif isinstance(formula, Negation):
                # ¬-Left: ¬A, Γ ⊢ C  ~>  Γ ⊢ A
                return [Sequent(before + after, formula.formula)]
            else:
                return None

    def oracle(self) -> int:
        """
        Returns the probability that the sequent is true.
        """
        # return 0
        raise NotImplementedError("Oracle is not implemented yet.")


__all__ = ["Sequent"]
