import re
from nltk.tag import pos_tag
import os, fnmatch
import torch
import spacy
import benepar
import lemminflect
from fastcoref import spacy_component
from negate import Negator
from coref_utils import resolve_coref_custom
from constituent_treelib import ConstituentTree, BracketedTree, Language, Structure
from datetime import date
from reasoner.logic import Literal, UnaryPredicate, And, Or, Implies, Not, InstancesOr, XOr
from tregex_parsing.main import Node
import string

class SpacyModel():
    def __init__(self):
        #self.nlp = spacy.load("en_core_web_trf")
        self.nlp = spacy.load("en_core_web_lg")
        config = {"attrs": {"tensor": None}}
        self.nlp.add_pipe("doc_cleaner", config=config)
        self.nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
        self.nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda'})
        #self.nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
    
    def process(self, text):
        with torch.no_grad():
            return self.nlp(text)

    def process_coref(self, text):
        with torch.no_grad():
            return resolve_coref_custom(self.nlp(text))

    def get_tree(self, sentence):
        return str(ConstituentTree(sentence, nlp=self.nlp))

def flatten_list(l):
    result = []
    for sl in l:
        if type(sl) == list:
            result.extend(flatten_list(sl))
        else:
            result.append(sl)
            
    return result


def find_instances(text, spacy_model):
    print("FIND INSTANCES EN COURS")
    result = set()
    with torch.no_grad():
        doc = spacy_model.process(text)
    sent = list(doc.sents)[0]
    print(sent._.parse_string)
    for i,word in enumerate(doc):
        if word.tag_ in {"NNP","NNPS", "NN", "NNS"}:
            np_ancestor = find_biggest_NP_parent(word)
            if type(np_ancestor) == type(word): #NP only 1 word long, must be an entity
                if word.tag_ in {"NNP", "NNPS"}:
                    result.add(np_ancestor.text)
            elif np_ancestor is not None:
                """if any([w.tag_.startswith("V") for w in np_ancestor]):
                    continue"""
                the_marker = False
                for np_word in np_ancestor: #we check if the NP is definite. If that is the case, we add it to the result
                    if np_word.lower_ in {"a", "an"}:
                        break
                    elif np_word.lower_ == "the":
                        the_marker = True
                    elif np_word == word: #end of the check of the sequence
                        print(word, word.tag_, the_marker, np_ancestor.text)
                        if word.tag_ in {"NNS", "NNPS", "NN","NNS"} and not the_marker: #if the word is plural but there is no
                            break
                        result.add(np_ancestor.text)
                        break
    instances = set()
    for instance in result: #Just in case an instance appears as part of an other, we only take the biggest available
        if not any([instance in other for other in result if other != instance]):
            instances.add(instance)
    return instances

def build_instances_disjunction(formulas_list):
    if len(formulas_list) == 1:
        return formulas_list[0]
    else:
        return InstancesOr(formulas_list[0], build_instances_disjunction(formulas_list[1:]))

def in_implies_left(formula, variable, left_formula = None):
    #recursively checks if a certain variable is on the left side of an implication within formula
    if isinstance(formula, Literal):
        if left_formula is not None and formula.char == variable:
            return left_formula
        return False
    elif isinstance(formula, UnaryPredicate):
        if left_formula is not None and formula.name == variable:
            return left_formula
        return False
    elif isinstance(formula, Implies):
        if in_implies_left(formula.left, variable, formula.left) and left_formula is None:
            return formula
        else:
            return in_implies_left(formula.left, variable, formula.left)
    elif isinstance(formula, Not):
        return in_implies_left(formula.inner, variable, left_formula)
    else:
        return in_implies_left(formula.left, variable, left_formula) or in_implies_left(formula.right, variable, left_formula)

def replace_implies_left(formula: Implies, variable):
    #removes variable from the left side of formula, and possibly removes completely the implication if the variable is alone on the left side
    #print(formula, variable)
    if isinstance(formula, Implies):
        if isinstance(formula.left, Literal) and formula.left.char == variable:
            return formula.right
        elif isinstance(formula.left, UnaryPredicate) and formula.left.name == variable:
            return formula.right
        else:
            return Implies(replace_implies_left(formula.left, variable), formula.right)
    elif isinstance(formula, Not):
        return Not(replace_implies_left(formula.inner, variable))
    elif isinstance(formula, Literal):
        return formula
    elif isinstance(formula, UnaryPredicate):
        return formula
    else:
        if isinstance(formula.left, Literal) and formula.left.char == variable:
            return formula.right
        elif isinstance(formula.right, Literal) and formula.right.char == variable:
            return formula.left
        elif isinstance(formula.left, UnaryPredicate) and formula.left.name == variable:
            return formula.right
        elif isinstance(formula.right, UnaryPredicate) and formula.right.name == variable:
            return formula.left
        else:
            if isinstance(formula, And):
                return And(replace_implies_left(formula.left, variable), replace_implies_left(formula.right, variable))
            elif isinstance(formula, Or):
                return Or(replace_implies_left(formula.left, variable), replace_implies_left(formula.right, variable))
            elif isinstance(formula, XOr):
                return XOr(replace_implies_left(formula.left, variable), replace_implies_left(formula.right, variable))

def replace_implies(formula, new_implication, old_implication):
    if isinstance(formula, Implies):
        if formula == old_implication:
            return new_implication
        else:
            return Implies(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))
    elif isinstance(formula, Not):
        return Not(replace_implies(formula.inner, new_implication, old_implication))
    elif isinstance(formula, Literal):
        return formula
    elif isinstance(formula, UnaryPredicate):
        return formula
    elif isinstance(formula, And):
        return And(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))
    elif isinstance(formula, Or):
        return Or(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))
    elif isinstance(formula, XOr):
        return XOr(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))

def transform_truth(premise, key_truth_quantif):
    valid_candidate = in_implies_left(premise, key_truth_quantif)
    #print("valid candidate", valid_candidate)
    if not valid_candidate:
        return premise
    new_implication = replace_implies_left(valid_candidate, key_truth_quantif)
    #print("new_imp", new_implication)
    return replace_implies(premise, new_implication, valid_candidate)

def convert_to_logic(tree, dic, first_allowed_letter, spacy_model, character_list, invert=False, neg=False):
    if tree.label() == "ROOT":
        text = " ".join(tree.get_words())
        if neg:
            negator = Negator(spacy_model, fail_on_unsupported=True)
            text = negator.negate_sentence(text, prefer_contractions=False)
        dic[first_allowed_letter] = text
        #new_first_allowed_letter = chr(ord(first_allowed_letter)+1)
        new_first_allowed_letter = character_list[character_list.index(first_allowed_letter)+1]
        if "X " in text or " X" in text:
            return UnaryPredicate(first_allowed_letter, "X"), dic, new_first_allowed_letter
        else:
            return Literal(first_allowed_letter), dic, new_first_allowed_letter
    elif tree.label() == "Attribution":
        #We consider its left child is only one Root.
        prefix = " ".join(tree[0].get_words()[:-1]) + " that" 
        tree[1] = tree[1].prefix_left(prefix)
        return convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
    elif tree.label() == "Punctuation":
        prefix = " ".join(tree[0].get_words()[:-1])
        tree[1] = tree[1].prefix_left(prefix)
        return convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
    else:
        rel = tree.label()
        if rel == "Universal": 
            return convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
        elif rel.lower() in {"implies", "if", "whenever", "when", "once"}:
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            return Implies(left, right), dic, first_allowed_letter
        elif rel.lower() == "or":
            if len(tree) > 2:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(Node("or", *tree[1:]), dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            else:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            if invert:
                return And(left, right), dic, first_allowed_letter
            return Or(left, right), dic, first_allowed_letter
        elif rel == "XOR":
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            return XOr(left, right), dic, first_allowed_letter
        elif rel == "Not":
            inner, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            return inner, dic, first_allowed_letter
        elif rel == "NeitherNor":
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            return And(left, right), dic, first_allowed_letter
        elif rel == "Nor":
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            return And(left, right), dic, first_allowed_letter
        else: #rel = "and"
            if len(tree) > 2:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(Node("and", *tree[1:]), dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            else:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            if invert:
                return Or(left, right), dic, first_allowed_letter
            return And(left, right), dic, first_allowed_letter
    return

def coref_step(step, spacy_model):
    full_text = " \ ".join(step.premises+[step.conclusion])
    resolved_text = spacy_model.process_coref(full_text)
    premises = [prem.strip() for prem in resolved_text.split(" \ ")]
    step.premises = premises[:-1]
    step.conclusion = premises[-1]
    return step

def get_date():
    today = date.today()
    return today.strftime("%m_%d")

def find_most_recent_file(folder, name):
    files = os.listdir(folder)
    matching_files = []
    if name.startswith("None_"):
        name = name[5:]
    if "EntailmentBank" in name:
        name = name.replace("EntailmentBank", "EB")
        name= name.replace("neg_*EB", "EB_neg")
        name= name.replace("hallu_*EB", "EB_hallu")
    for f in files:
            matching_files.append(f)
    matching_files.sort(reverse=True, key = lambda x: x.replace(".jsonl", ""))
    return matching_files[0]

def clear_cache():
    torch.cuda.empty_cache()