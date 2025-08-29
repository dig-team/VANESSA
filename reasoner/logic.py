from collections import defaultdict
from typing import Any
import string

class Formula:
    pass

class Literal(Formula):
    def __init__(self, char):
        self.char = char

    def negate(self):
        return Not(self)

    def __eq__(self, other):
        return isinstance(other, Literal) and self.char == other.char

    def __hash__(self):
        return hash(self.char)

    def __repr__(self):
        return self.char
    
    def copy(self):
        return Literal(self.char)
    
    def is_universal(self):
        return False
    
    def instantiate(self, first_allowed_letter, character_list):
        return self, first_allowed_letter, []

    def remove_quantif(self):
        return self

    def get_variables(self):
        return [self.char]

    def get_variables_implications(self):
        return [self.char], [self.char]

    def clean(self, correspondance_dict):
        return self

    def equal_sentence(self, other, correspondance_dict):
        return isinstance(other, Literal) and correspondance_dict[self.char] == correspondance_dict[other.char]

    def get_xor_pairs(self):
        return []

    def transmute(self, correspondance_dict):
        return Literal(correspondance_dict[self.char])

class UnaryPredicate(Formula):
    def __init__(self, name, arg):
        self.name = name
        self.arg = arg

    def negate(self):
        return Not(self)

    def __eq__(self, other):
        return isinstance(other, UnaryPredicate) and self.name == other.name and self.arg == other.arg

    def __hash__(self):
        return hash((self.name, self.arg))

    def __repr__(self):
        return f"{self.name}({self.arg})"
    
    def copy(self):
        return UnaryPredicate(self.name, self.arg)
    
    def is_universal(self):
        return self.arg[0] == "X"
    
    def instantiate(self, first_allowed_letter, character_list):
        char = self.name
        self = Literal(first_allowed_letter)
        return self, character_list[character_list.index(first_allowed_letter)+1], [char]
        #return self, chr(ord(first_allowed_letter)+1), [char]
    
    def remove_quantif(self):
        return Literal(self.name)

    def get_variables(self):
        return [self.name]

    def clean(self, correspondance_dict):
        return self

    def equal_sentence(self, other, correspondance_dict):
        return isinstance(other, UnaryPredicate) and correspondance_dict[self.name] == correspondance_dict[other.name]

    def transmute(self, correspondance_dict):
        return UnaryPredicate(correspondance_dict[self.name], self.arg)

class And(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __eq__(self, other):
        return isinstance(other, And) and self.left == other.left and self.right == other.right
    
    def __hash__(self) -> int:
        return hash((self.left, self.right))
    
    def __repr__(self):
        return f"({self.left}∧{self.right})"
    
    def copy(self):
        left = self.left.copy()
        right = self.right.copy()
        return And(left, right)

    def is_universal(self):
        return (self.left.is_universal() or self.right.is_universal())

    def negate(self):
        return Or(self.left.negate(), self.right.negate())

    def instantiate(self, first_allowed_letter, character_list):
        left, first_allowed_letter, charsl = self.left.instantiate(first_allowed_letter, character_list)
        right, first_allowed_letter, charsr = self.right.instantiate(first_allowed_letter, character_list)
        print(charsl, charsr)
        chars = charsl + charsr
        print(chars)
        return And(left, right), first_allowed_letter, chars
    
    def remove_quantif(self):
        return And(self.left.remove_quantif(), self.right.remove_quantif())

    def get_variables(self):
        return self.left.get_variables() + self.right.get_variables()

    def get_variables_implications(self):
        left = self.left.get_variables_implications()
        right = self.right.get_variables_implications()
        return list(set(left[0]+right[0])), list(set(left[1]+right[1]))

    def clean(self, correspondance_dict):
        if isinstance(self.left, Implies) and isinstance(self.right, Implies):
            left_prem = self.left.left
            right_prem = self.right.left
            if left_prem.equal_sentence(right_prem, correspondance_dict):
                return Implies(left_prem, And(self.left.right, self.right.right)).clean(correspondance_dict)

        return And(self.left.clean(correspondance_dict), self.right.clean(correspondance_dict))

    def equal_sentence(self, other, correspondance_dict):
        return isinstance(other, And) and ((self.left.equal_sentence(other.left, correspondance_dict) and self.right.equal_sentence(other.right, correspondance_dict)) or (self.left.equal_sentence(other.right, correspondance_dict) and self.right.equal_sentence(other.left, correspondance_dict)))

    def get_xor_pairs(self):
        return self.left.get_xor_pairs() + self.right.get_xor_pairs()

    def transmute(self, correspondance_dict):
        return And(self.left.transmute(correspondance_dict), self.right.transmute(correspondance_dict))

class Or(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def copy(self):
        left = self.left.copy()
        right = self.right.copy()
        return Or(left, right)

    def __repr__(self):
        return f"({self.left}∨{self.right})"
    
    def __hash__(self) -> int:
        return hash((self.left, self.right))
    
    def __eq__(self, other):
        return isinstance(other, Or) and self.left == other.left and self.right == other.right

    def negate(self):
        return And(self.left.negate(), self.right.negate())
    
    def is_universal(self):
        return (self.left.is_universal() or self.right.is_universal())

    def instantiate(self, first_allowed_letter, character_list):
        left, first_allowed_letter, charsl = self.left.instantiate(first_allowed_letter, character_list)
        right, first_allowed_letter, charsr = self.right.instantiate(first_allowed_letter, character_list)
        chars = charsl+charsr
        return Or(left, right), first_allowed_letter, chars

    def remove_quantif(self):
        return Or(self.left.remove_quantif(), self.right.remove_quantif())

    def get_variables(self):
        return self.left.get_variables() + self.right.get_variables()

    def get_variables_implications(self):
        left = self.left.get_variables_implications()
        right = self.right.get_variables_implications()
        return list(set(left[0]+right[0])), list(set(left[1]+right[1]))

    def clean(self, correspondance_dict):
        if isinstance(self.left, Implies) and isinstance(self.right, Implies):
            left_prem = self.left.left
            right_prem = self.right.left
            if left_prem.equal_sentence(right_prem, correspondance_dict):
                return Implies(left_prem, Or(self.left.right, self.right.right)).clean(correspondance_dict)
                
        return Or(self.left.clean(correspondance_dict), self.right.clean(correspondance_dict))

    def equal_sentence(self, other, correspondance_dict):
        return isinstance(other, Or) and ((self.left.equal_sentence(other.left, correspondance_dict) and self.right.equal_sentence(other.right, correspondance_dict)) or (self.left.equal_sentence(other.right, correspondance_dict) and self.right.equal_sentence(other.left, correspondance_dict)))

    def get_xor_pairs(self):
        return self.left.get_xor_pairs() + self.right.get_xor_pairs()

    def transmute(self, correspondance_dict):
        return Or(self.left.transmute(correspondance_dict), self.right.transmute(correspondance_dict))
    
class Implies(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def copy(self):
        left = self.left.copy()
        right = self.right.copy()
        return Implies(left, right)

    def __repr__(self):
        return f"({self.left}→{self.right})"
    
    def __hash__(self) -> int:
        return hash((self.left, self.right))
    
    def __eq__(self, other):
        return isinstance(other, Implies) and self.left == other.left and self.right == other.right
    
    def negate(self):
        return Implies(self.left, self.right.negate())

    def is_universal(self):
        return (self.left.is_universal() or self.right.is_universal())

    def instantiate(self, first_allowed_letter, character_list):
        print(self.left, self.right)
        left, first_allowed_letter, charsl = self.left.instantiate(first_allowed_letter, character_list)
        right, first_allowed_letter, charsr = self.right.instantiate(first_allowed_letter, character_list)
        chars = charsl+charsr
        return Implies(left, right), first_allowed_letter, chars

    def remove_quantif(self):
        return Implies(self.left.remove_quantif(), self.right.remove_quantif())
    
    def get_variables(self):
        return self.left.get_variables() + self.right.get_variables()

    def get_variables_implications(self):
        left1, left2 = self.left.get_variables_implications()
        right1, right2 = self.right.get_variables_implications()
        left, right = list(set(left1+left2)), list(set(right1+right2))
        if right1 != right2:
            left = list(set(left+right1))
        if left1 != left2:
            right = list(set(right+left2))
        return left, right

    def clean(self, correspondance_dict):
        return Implies(self.left.clean(correspondance_dict), self.right.clean(correspondance_dict))

    def equal_sentence(self, other, correspondance_dict):
        return isinstance(other, Implies) and ((self.left.equal_sentence(other.left, correspondance_dict) and self.right.equal_sentence(other.right, correspondance_dict)) or (self.left.equal_sentence(other.right, correspondance_dict) and self.right.equal_sentence(other.left, correspondance_dict)))
    
    def get_xor_pairs(self):
        return self.left.get_xor_pairs() + self.right.get_xor_pairs()

    def transmute(self, correspondance_dict):
        return Implies(self.left.transmute(correspondance_dict), self.right.transmute(correspondance_dict))

class Equivalence(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} ↔ {self.right})"

    def is_universal(self):
        return (self.left.is_universal() or self.right.is_universal())
    
    def instantiate(self, first_allowed_letter, character_list):
        left, first_allowed_letter, charsl = self.left.instantiate(first_allowed_letter, character_list)
        right, first_allowed_letter, charsr = self.right.instantiate(first_allowed_letter, character_list)
        chars = charsl+charsr
        return Equivalence(left, right), first_allowed_letter, chars  
    
    def remove_quantif(self):
        return Equivalence(self.left.remove_quantif(), self.right.remove_quantif()) 
     
    def get_variables(self):
        return [self.left.get_variables(), self.right.get_variables()]

    def transmute(self, correspondance_dict):
        return Equivalence(self.left.transmute(correspondance_dict), self.right.transmute(correspondance_dict))

class Not(Formula):
    def __init__(self, inner):
        self.inner = inner

    def __hash__(self) -> int:
        return hash(self.inner)

    def __eq__(self, other):
        return isinstance(other, Not) and self.inner == other.inner

    def copy(self):
        inner = self.inner.copy()
        return Not(inner)

    def __repr__(self):
        return f"(¬{self.inner})"

    def negate(self):
        return self.inner
    
    def is_universal(self):
        return self.inner.is_universal()

    def instantiate(self, first_allowed_letter, character_list):
        inner, first_allowed_letter, chars = self.inner.instantiate(first_allowed_letter, character_list)
        return Not(inner), first_allowed_letter, chars

    def remove_quantif(self):
        return Not(self.inner.remove_quantif())
    
    def get_variables(self):
        return self.inner.get_variables()

    def get_variables_implications(self):
        if isinstance(self.inner, Implies):
            left = self.inner.left.get_variables_implications()
            right = self.inner.right.get_variables_implications()
            return list(set(left[0]+right[0])), list(set(left[1]+right[1]))

        else:
            left, right = self.inner.get_variables_implications()
            return left, right

    def clean(self, correspondance_dict):  
        return Not(self.inner.clean(correspondance_dict))

    def equal_sentence(self, other, correspondance_dict):
        return isinstance(other, Not) and self.inner.equal_sentence(other.inner, correspondance_dict)

    def get_xor_pairs(self):
        return self.inner.get_xor_pairs()

    def transmute(self, correspondance_dict):
        return Not(self.inner.transmute(correspondance_dict))

class InstancesOr(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def copy(self):
        left = self.left.copy()
        right = self.right.copy()
        return InstancesOr(left, right)

    def __repr__(self):
        return f"({self.left}∪{self.right})"
    
    def __hash__(self) -> int:
        return hash((self.left, self.right))
    
    def __eq__(self, other):
        return isinstance(other, InstancesOr) and self.left == other.left and self.right == other.right

    def get_variables(self):
        return self.left.get_variables() + self.right.get_variables()

    def get_variables_implications(self):
        left = self.left.get_variables_implications()
        right = self.right.get_variables_implications()
        return list(set(left[0]+right[0])), list(set(left[1]+right[1]))
    
    def clean(self, correspondance_dict):
        return InstancesOr(self.left.clean(correspondance_dict), self.right.clean(correspondance_dict))

    def get_xor_pairs(self):
        return self.left.get_xor_pairs() + self.right.get_xor_pairs()

    def transmute(self, correspondance_dict):
        return InstancesOr(self.left.transmute(correspondance_dict), self.right.transmute(correspondance_dict))

class XOr(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def copy(self):
        left = self.left.copy()
        right = self.right.copy()
        return XOr(left, right)

    def __repr__(self):
        return f"({self.left}⊻{self.right})"
    
    def __hash__(self) -> int:
        return hash((self.left, self.right))
    
    def __eq__(self, other):
        return isinstance(other, XOr) and self.left == other.left and self.right == other.right

    def is_universal(self):
        return (self.left.is_universal() or self.right.is_universal())

    def instantiate(self, first_allowed_letter, character_list):
        left, first_allowed_letter, charsl = self.left.instantiate(first_allowed_letter, character_list)
        right, first_allowed_letter, charsr = self.right.instantiate(first_allowed_letter, character_list)
        chars = charsl+charsr
        return XOr(left, right), first_allowed_letter, chars

    def remove_quantif(self):
        return XOr(self.left.remove_quantif(), self.right.remove_quantif())

    def get_variables(self):
        return self.left.get_variables() + self.right.get_variables()

    def get_variables_implications(self):
        left = self.left.get_variables_implications()
        right = self.right.get_variables_implications()
        return list(set(left[0]+right[0])), list(set(left[1]+right[1]))

    def negate(self):
        return Or(And(self.left.negate(), self.right), And(self.left, self.right.negate()))

    def clean(self, correspondance_dict):
        if isinstance(self.left, Implies) and isinstance(self.right, Implies):
            left_prem = self.left.left
            right_prem = self.right.left
            if left_prem.equal_sentence(right_prem, correspondance_dict):
                return Implies(left_prem, XOr(self.left.right, self.right.right)).clean(correspondance_dict)
                
        return XOr(self.left.clean(correspondance_dict), self.right.clean(correspondance_dict))

    def equal_sentence(self, other, correspondance_dict):
        return isinstance(other, XOr) and ((self.left.equal_sentence(other.left, correspondance_dict) and self.right.equal_sentence(other.right, correspondance_dict)) or (self.left.equal_sentence(other.right, correspondance_dict) and self.right.equal_sentence(other.left, correspondance_dict)))

    def get_xor_pairs(self):
        lefts, rights = self.left.get_variables(), self.right.get_variables()
        return [(l, r) for l in lefts for r in rights]

    def transmute(self, correspondance_dict):
        return XOr(self.left.transmute(correspondance_dict), self.right.transmute(correspondance_dict))