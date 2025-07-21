import collections

from reasoner.logic import Literal, And, Or, Implies, Not, InstancesOr, XOr, UnaryPredicate
from reasoner.solver import Proof
from reasoner.utils import flatten_formula
import itertools

def parse_formula(string, mode="entailments"):
    # Base case: if the string is empty, return None
    if not string:
        return None

    # Base case: if the string is a single literal, return a Literal object
    if len(string) == 1:
        return Literal(string)

    if string == "-1":
        return -1
    
    if string[0] == "(":
        # Find the matching closing parenthesis
        parentheses_count = 1
        for i, char in enumerate(string[1:]):
            if char == "(":
                parentheses_count += 1
            elif char == ")":
                parentheses_count -= 1
            if parentheses_count == 0:
                break
        # If the matching closing parenthesis is at the end of the string, the string is already in parentheses
        if i == len(string) - 2:
            return parse_formula(string[1:-1], mode=mode)
        
    # Find the main connective in the string
    connectives = ["∧", "∨", "→", "¬", "∪", "⊻"]
    index = None
    parentheses_count = 0
    for i, char in enumerate(string):
        if char == "(":
            parentheses_count += 1
        elif char == ")":
            parentheses_count -= 1
        elif parentheses_count == 0 and char in connectives:
            index = i
            break

    # Base case: if the string does not contain any connective but is not len(1), then it must be len(4) - it is a unary formula (1 universal quantifier)
    if index is None and len(string) == 4:
        return UnaryPredicate(string[0], string[2])

    # Recursive case: split the string into left and right subformulas
    left_subformula = parse_formula(string[:index], mode=mode)
    right_subformula = parse_formula(string[index + 1:], mode=mode)

    # Create the appropriate formula object based on the main connective
    if string[index] == "∧":
        return And(left_subformula, right_subformula)
    elif string[index] == "∨":
        return Or(left_subformula, right_subformula)
    elif string[index] == "⊻":
        if mode == "reasoning":
            return And(Or(left_subformula, right_subformula), Or(Not(left_subformula), Not(right_subformula)))
        else:
            return XOr(left_subformula, right_subformula)
    elif string[index] == "→":
        return Implies(left_subformula, right_subformula)
    elif string[index] == "¬":
        return Not(right_subformula)
    elif string[index] == "∪":
        if mode == "reasoning":
            return And(left_subformula, right_subformula)
            #return [left_subformula, right_subformula]
        elif mode == "conclusion-reasoning":
            return Or(left_subformula, right_subformula)
        else:
            return InstancesOr(left_subformula, right_subformula)

def parse_entailments(entailments_dict: dict):
    entailments = []
    for key, value in entailments_dict.items():
        for v in value:
            if v[0] == "¬":
                entails = Implies(Literal(key), Not(Literal(v[1])))
                entailments.append(entails)
            elif v[0] != key:
                entails = Implies(Literal(key), Literal(v[0]))
                entailments.append(entails)
            
    return entailments

def parse_entailments_reasoning(entailments_dict: dict, premises, conclusion):
    variables = flatten_formula([formula.get_variables() for formula in premises+[conclusion]])
    entailments = []
    #print(entailments_dict)
    for key, value in entailments_dict.items():
        if type(key) != str: #The key is a a formula
            entailments = [Implies(k, v) for k, vals in entailments_dict.items() for v in vals]
            return entailments

        key = key.replace("(",'').replace(")",'')
        value = [v.replace("(",'').replace(")",'') for v in value]
        if key[0] == "¬":
            key, element = key[1], Not(Literal(key[1]))
        else:
            key, element = key, Literal(key)
        for v in value:
            if v[0] == "¬":
                #print(key)
                if key in variables and v[1] in variables:
                    entails = Implies(element, Not(Literal(v[1])))
                    entailments.append(entails)
            elif v[0] != key:
                if key in variables and v[0] in variables:
                    entails = Implies(element, Literal(v[0]))
                    entailments.append(entails)
    return entailments

def reason(premises, conclusion, entailments_dict):
    entailments = parse_entailments_reasoning(entailments_dict, premises, conclusion)
    #print(premises, conclusion)
    premises_set = set(premises) | set(entailments)
    proof = Proof(premises_set, conclusion)
    return proof

def build_proofs(premises, conclusion, entailments_dict):
    proofs = []
    #print(premises)
    if type(premises[0]) == str:
        premises = [flatten_formula(parse_formula(premise, mode="reasoning")) for premise in premises]
        ps = itertools.product(*premises)
        #print(list(ps))
        conclusions = flatten_formula(parse_formula(conclusion, mode="conclusion-reasoning"))
    else:
        premises = [flatten_formula(parse_formula(str(premise), mode="reasoning")) for premise in premises]
        ps = itertools.product(*premises)
        conclusions = flatten_formula(parse_formula(str(conclusion), mode="conclusion-reasoning"))
    for p in ps:
        for c in conclusions:
            proof = reason(list(p), c, entailments_dict)
            proofs.append(proof)
    return proofs