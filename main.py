from entailment_model import initialize_entailment_model
import itertools
from utils import rel_simplify, clean_temp, SpacyModel, find_instances, flatten_list, clear_cache, tree_transform, get_date, find_most_recent_file, build_instances_disjunction, transform_truth, coref_step, convert_to_logic
import json, os, sys
from collections import defaultdict
from reasoner.reasoner import build_proofs, parse_formula
from reasoner.logic import Literal, InstancesOr, Not
import re
from func_timeout import FunctionTimedOut, func_timeout
from tregex_parsing.patterns import process_tree

class ReasoningStep():
    def __init__(self, premises, conclusion, correspondance_dict = None, entailments_dict = None):
        self.premises = premises.copy()
        self.conclusion = conclusion
        self.first_allowed_letter = "A"
        
        if correspondance_dict is None:
            correspondance_dict = {}
        self.correspondance_dict = correspondance_dict

        if entailments_dict is None:
            entailments_dict = defaultdict(list)
        self.entailments_dict = entailments_dict

    def logic_transform(self, spacy_model, parsing_cache={}):
        print("LOGIC TRANSFORM")
        print(self.premises)
        print(self.conclusion)
        rst_clean_trees, original_premises = split_premises(self.premises+[self.conclusion], spacy_model, parsing_cache)
        for i, premise in enumerate(self.premises):
            rst_clean_tree = rst_clean_trees[i]
            if type(rst_clean_tree) == list:
                formula, correspondance = rst_clean_tree[0], rst_clean_tree[1]
                print(formula)
                formula = parse_formula(formula)
                print(formula)
                transmute_dict = {}
                for key, value in correspondance.items():
                    self.correspondance_dict[self.first_allowed_letter] = value
                    transmute_dict[key] = self.first_allowed_letter
                    self.first_allowed_letter = chr(ord(self.first_allowed_letter)+1)
                self.premises[i] = formula.transmute(transmute_dict)
            else:
                formula, dict_update, self.first_allowed_letter = tree_transform(rst_clean_tree, {}, self.first_allowed_letter)
                self.premises[i] = formula
                self.correspondance_dict.update(dict_update)

        conclusion_rst_clean_tree = rst_clean_trees[-1]
        if type(conclusion_rst_clean_tree) == list:
            formula, correspondance = conclusion_rst_clean_tree[0], conclusion_rst_clean_tree[1]
            formula = parse_formula(formula)
            transmute_dict = {}
            for key, value in correspondance.items():
                self.correspondance_dict[self.first_allowed_letter] = value
                transmute_dict[key] = self.first_allowed_letter
                self.first_allowed_letter = chr(ord(self.first_allowed_letter)+1)
            self.conclusion = formula.transmute(transmute_dict)
        else:
            if self.conclusion != "":
                formula, dict_update, self.first_allowed_letter = tree_transform(conclusion_rst_clean_tree, {}, self.first_allowed_letter)
                self.conclusion = formula
                self.correspondance_dict.update(dict_update)
            else:
                self.conclusion = Literal("0")
        self.update_parsing_cache(parsing_cache, original_premises)
        print("before instantiate", self.premises, self.conclusion)
        self.instantiate_formulas(spacy_model)
        print("before clean", self.premises, self.conclusion)
        self.clean_formulas()
        print("after clean", self.premises, self.conclusion)
        return

    def logic_transform_tregex(self, spacy_model, parsing_cache={}):
        print("LOGIC TRANSFORM")
        self.premises = [prem.split("(")[0]+"." if "(" in prem else prem for prem in self.premises]
        self.conclusion = self.conclusion.split("(")[0]+"." if "(" in self.conclusion else self.conclusion
        
        self = coref_step(self, spacy_model)
        for i, sentence in enumerate(self.premises):
            if sentence in parsing_cache:
                formula, correspondance = parsing_cache[sentence]
                formula = parse_formula(formula)
                transmute_dict = {}
                for key, value in correspondance.items():
                    self.correspondance_dict[self.first_allowed_letter] = value
                    transmute_dict[key] = self.first_allowed_letter
                    self.first_allowed_letter = chr(ord(self.first_allowed_letter)+1)
                self.premises[i] = formula.transmute(transmute_dict)
            else:
                a = process_tree(sentence, spacy_model, universal=True)
                formula, dict_update, self.first_allowed_letter = convert_to_logic(a, {}, self.first_allowed_letter, spacy_model)

                parsing_cache[sentence] = [str(formula), {key: value for (key, value) in dict_update.items()}]
                self.premises[i] = formula
                self.correspondance_dict.update(dict_update)
        if self.conclusion != "":
            if self.conclusion in parsing_cache:
                formula, correspondance = parsing_cache[self.conclusion]
                formula = parse_formula(formula)
                transmute_dict = {}
                for key, value in correspondance.items():
                    self.correspondance_dict[self.first_allowed_letter] = value
                    transmute_dict[key] = self.first_allowed_letter
                    self.first_allowed_letter = chr(ord(self.first_allowed_letter)+1)
                self.conclusion = formula.transmute(transmute_dict)
            else:
                a = process_tree(self.conclusion, spacy_model, universal=True)
                formula, dict_update, self.first_allowed_letter = convert_to_logic(a, {}, self.first_allowed_letter, spacy_model)

                parsing_cache[self.conclusion] = [str(formula), {key: value for (key, value) in dict_update.items()}]
                self.conclusion = formula
                self.correspondance_dict.update(dict_update)
        else:
            self.conclusion = Literal("0")

        self.instantiate_formulas(spacy_model)
        self.clean_formulas()
        print(self.premises, self.conclusion, self.correspondance_dict)
        return parsing_cache

    def get_entailments(self, model, entailment_cache={}):
        print("ENTAILMENT TEMPS")
        print(self.premises)
        print(self.conclusion)
        print(self.correspondance_dict)
        groups = [premise.get_variables() for premise in self.premises]
        fuses = []
        for list1, list2 in itertools.combinations(groups, 2):
            for item1 in list1:
                for item2 in list2:
                    fuses.append((item1, item2))
        new_pairs = flatten_list([list(itertools.permutations(l)) for l in fuses])
        new_pairs = [pair for pair in new_pairs if len(pair) == 2]

        left_right_premises = [premise.get_variables_implications() for premise in self.premises]
        left_premises = flatten_list([premise[0] for premise in left_right_premises])
        right_premises = flatten_list([premise[1] for premise in left_right_premises])
        full_premises = flatten_list([premise.get_variables() for premise in self.premises])
        
        RL_fuses = list(itertools.product(right_premises, left_premises))
        RL_premises_pairs = list(set(new_pairs) & set(RL_fuses))
        LL_fuses = list(itertools.product(left_premises, left_premises))
        LL_premises_pairs = list(set(new_pairs) & set(LL_fuses))
        RR_fuses = list(itertools.product(right_premises, right_premises))
        RR_premises_pairs = list(set(new_pairs) & set(RR_fuses))

        left_concs, right_concs = self.conclusion.get_variables_implications()
        left_concs_pairs = list(itertools.product(left_concs, left_premises))
        right_concs_pairs = list(itertools.product(right_premises, right_concs))
        left_negs_concs_pairs = list(itertools.product(left_concs, right_premises))


        xor_pairs = flatten_list([premise.get_xor_pairs() for premise in self.premises])
        batch_size = 8
        print("RLP", RL_premises_pairs)
        print("LC", left_concs_pairs)
        print("RC", right_concs_pairs)
        print("---")
        print("RRP", RR_premises_pairs)
        print("LNC", left_negs_concs_pairs)
        print("XP", xor_pairs)
        print("RLP+LC+RC", len(RL_premises_pairs + left_concs_pairs + right_concs_pairs))
        entailment, entailment_cache = model.entail(RL_premises_pairs + left_concs_pairs + right_concs_pairs, self.correspondance_dict, batch_size, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "E":
                self.entailments_dict[Literal(pair[0])].append(Literal(pair[1]))
        print("RRP+LNC+XP", len(RR_premises_pairs + left_negs_concs_pairs + xor_pairs))
        entailment, entailment_cache = model.entail(RR_premises_pairs + left_negs_concs_pairs + xor_pairs, self.correspondance_dict, batch_size, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "C":
                self.entailments_dict[Literal(pair[0])].append(Not(Literal(pair[1])))
        contradict_pairs = list(itertools.product(right_concs, left_premises))
        print("---")
        print("CP", contradict_pairs)
        print("LLP", LL_premises_pairs)
        print("CP+LLP", len(contradict_pairs + LL_premises_pairs))
        #input("TOUS LES ENTAILMENTS SONT FAITS")
        entailment, entailment_cache = model.entail(contradict_pairs + LL_premises_pairs, self.correspondance_dict, batch_size, contrad_premise=True, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "E":
                self.entailments_dict[Not(Literal(pair[0]))].append(Literal(pair[1]))
        print(self.entailments_dict)
        return entailment_cache
        
    def check_conclusion(self):
        if self.conclusion == -1:
            raise ZeroDivisionError("No conclusion found")
        print("CHECK CONCLUSION")
        print(self.premises)
        print(self.conclusion)
        print(self.entailments_dict)
        proofs = build_proofs(self.premises, self.conclusion, self.entailments_dict)
        results = []
        for proof in proofs:
            try:
                result = proof.verify()
            except ValueError:
                result = "Contradiction in the premises"
            except FunctionTimedOut:
                result = False
            results.append(result)
            if result == True:
                return True
        if all([result == "Contradiction in the premises" for result in results]):
            raise ValueError("Contradiction in the premises")
        if any([result == True for result in results]):
            return True
        return False
    
    def instantiate_formulas(self, spacy_model):
        print("INSTANTIATE FORMULAS")
        universal_present = [premise.is_universal() for premise in self.premises] + [self.conclusion.is_universal()]
        if all(universal_present):
            self.remove_quantif()
            return
        if not any(universal_present):  
            return
        else:
            instances_list = []
            for prop in self.premises+[self.conclusion]:
                if not prop.is_universal():
                    premise_keys = flatten_list(prop.get_variables())
                    for key in premise_keys:
                        text = self.correspondance_dict[key]
                        instances = find_instances(text, spacy_model)
                        for instance in instances:
                            if instance.lower().split()[0] in {"some","someone", "something", "somebody", "somewhere", "a", "an"}:
                                pass
                            else:
                                instances_list.append(re.sub("`|'", "", instance).strip())                    #parcours de la formule
                    #pour chaque phrase, extraire des GN et les mettre dans une liste
            if instances_list == []:
                self.remove_quantif()
                return
            instances_list = list(set(instances_list))
            print("INSTALCES LIST AVANT CHECK", instances_list)

            formulas_to_add = []
            formulas_to_remove = []
            not_viable_instances = set()
            instances_truth_quantif = {}
            #we verify that instances are correct: if an instance is already in a universally quantified sentence, then it is not
            #Exception: "X is something" and the candidate instance is contained in "something", while being on the left side of an implication. In this case, the proposition will be True, and we need to remove the sentence from the implication (the condition is fulfilled and won't be solved by entailment)
            univ_premises_nb = 0
            for premise in self.premises + [self.conclusion]:
                if premise.is_universal():
                    univ_premises_nb += 1
                    premise_keys = flatten_list(premise.get_variables())
                    for key in premise_keys:
                        for instance in instances_list:
                            if " is " in self.correspondance_dict[key] and self.correspondance_dict[key].split(" is")[1].replace(" a ", " ").replace(" an ", " ").replace(" .", "").replace(" the ", " ").strip() in instance:
                                print("PTDR CA ARRIVE POUR DE VRAI")
                                print(instance, self.correspondance_dict[key])
                                instances_truth_quantif[premise] = (instance, key)
                                break
                            elif re.search(r"(?i)"+" "+instance+" ", " "+self.correspondance_dict[key]):
                                print("INSTANCE", instance, "IS IN", self.correspondance_dict[key])
                                not_viable_instances.add(instance)
                                break
            if univ_premises_nb == 1 and len(set(instances_list) - not_viable_instances) == 0:
                not_viable_instances = set()
            instances_list = list(set(instances_list) - not_viable_instances)
            print("INSTALCES LIST apres CHECK", instances_list)
            if instances_list == []:
                self.remove_quantif()
                #il faut peut-etre rajouter que c'est un cas d'erreur ici
                return
            for premise in self.premises:
                if premise.is_universal():
                    formulas_to_remove.append(premise)
                    instance_truth_quantif = ""
                    if premise in instances_truth_quantif:
                        instance_truth_quantif, key_truth_quantif = instances_truth_quantif[premise]
                    new_premises = []
                    for instance in instances_list:
                        new_formula = premise.copy()
                        if instance == instance_truth_quantif:
                            new_formula = transform_truth(new_formula, key_truth_quantif)
                        new_formula, _, chars = new_formula.instantiate(self.first_allowed_letter)
                        new_premises.append(new_formula)
                        for i, char in enumerate(chars):
                            current_letter = chr(ord(self.first_allowed_letter) + i)
                            self.correspondance_dict[current_letter] = self.correspondance_dict[char].replace("X", instance)
                        self.first_allowed_letter = chr(ord(current_letter)+1)
                    if new_premises != []:
                        formulas_to_add.append(build_instances_disjunction(new_premises))

            self.premises = [prem for prem in self.premises if prem not in formulas_to_remove]
            self.premises.extend(formulas_to_add)

            if self.conclusion.is_universal():
                instance_truth_quantif = ""
                if self.conclusion in instances_truth_quantif:
                    instance_truth_quantif, key_truth_quantif = instances_truth_quantif[self.conclusion]
                formulas_to_add = []
                for instance in instances_list:
                    new_formula = self.conclusion.copy()
                    if instance == instance_truth_quantif:
                        new_formula = transform_truth(new_formula, key_truth_quantif)
                    new_formula, _, chars = new_formula.instantiate(self.first_allowed_letter)
                    formulas_to_add.append(new_formula)
                    for i, char in enumerate(chars):
                        current_letter = chr(ord(self.first_allowed_letter) + i)
                        self.correspondance_dict[current_letter] = self.correspondance_dict[char].replace("X", instance)
                    self.first_allowed_letter = chr(ord(current_letter)+1)

                self.conclusion = build_instances_disjunction(formulas_to_add)
                     
        return

    def remove_quantif(self):
        print(self.premises)
        for i, premise in enumerate(self.premises):
            self.premises[i] = premise.remove_quantif()
        self.conclusion = self.conclusion.remove_quantif()
        print(self.premises)

    def clean_formulas(self):
        self.premises = [premise.clean(self.correspondance_dict) for premise in self.premises]
        self.conclusion = self.conclusion.clean(self.correspondance_dict)

    def update_parsing_cache(self, parsing_cache, original_sentences):
        #print("UPDATE PARSING CACHE - ", len(parsing_cache))
        for premise, i in original_sentences:
            if i == len(self.premises):
                parsing_cache[premise] = [str(self.conclusion), {key: value for (key, value) in self.correspondance_dict.items() if key in self.conclusion.get_variables()}]
            else:
                print("PREMISE", self.premises[i])
                print("Corr", self.premises[i].get_variables())
                parsing_cache[premise] = [str(self.premises[i]), {key: value for (key, value) in self.correspondance_dict.items() if key in self.premises[i].get_variables()}]
        """for i, premise in enumerate(self.premises):
            parsing_cache[original_sentences[i]] = [str(premise), {key: value for (key, value) in self.correspondance_dict.items() if key in premise.get_variables()}]
        if conclusion:
            parsing_cache[original_sentences[-1]] = [str(self.conclusion), {key: value for (key, value) in self.correspondance_dict.items() if key in self.conclusion.get_variables()}]"""
        #print("UPDATED PARSING CACHE - ", len(parsing_cache))
        #input()
        

def read_reasoning(lines):
    reasoning_steps = []
    premises = []
    for line in lines:
        if ":" in line:
            type = line.split(":")[0].split()[0]
            if type == "Conclusion":
                conclusion = ":".join(line.split(":")[1:])#.split(".")[0]+"." #In the case there are several :, only the first one is of interest
                premises = premises
                step = ReasoningStep(premises, conclusion)
                reasoning_steps.append(step)
                premises = []
            elif type == "Premise":
                premises.append(":".join(line.split(":")[1:]))#.split(".")[0]+".")
            else:
                continue
    return reasoning_steps

def check_consistency_symbolic(steps, original_text):
    problem_sentences = []
    for i, step in enumerate(steps):
        step_problem_sentences = []
        for j, premise in enumerate(step.premises):
            sentence = premise.strip()
            if original_text.startswith(premise.strip()) or ". "+premise.strip() in original_text.strip():
                continue
            else:
                step_problem_sentences.append(premise.strip())
        problem_sentences.append(step_problem_sentences)
        original_text = original_text + " " + step.conclusion.strip()
    return problem_sentences

def check_consistency_entailments(steps, original_text, model):
    original_sentences = [sentence.strip()+"." for sentence in original_text.split(".")[:-1]]
    original_sentences_dict = {i: sentence for i, sentence in enumerate(original_sentences)}
    problem_sentences = []
    batch_size = 8

    entailment_cache = json.load(open(entailment_cache_file, 'r'))
    #input("LOADED")
    for step in steps:
        step_problem_sentences = []
        n = len(original_sentences_dict)
        for premise in step.premises:
            non_hallucinated = False
            if original_text.startswith(premise.strip()) or ". "+premise.strip() in original_text.strip():
                continue
            pairs = [(original_sentence_id, n) for original_sentence_id in original_sentences_dict.keys() if original_sentence_id != n]
            original_sentences_dict[n] = premise.strip()
            entailment, entailment_cache = model.entail(pairs, original_sentences_dict, batch_size, entailment_cache=entailment_cache)
            #json.dump(entailment_cache, open(entailment_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
            for pair, result in entailment:
                if result == "E":
                    non_hallucinated = True
                    break
            if not non_hallucinated:
                step_problem_sentences.append(premise.strip())
        problem_sentences.append(step_problem_sentences)
        original_sentences_dict[n] = step.conclusion.strip()
        original_text = original_text + " " + step.conclusion.strip()
    return problem_sentences

def check_consistency_direct_entailments(steps, original_text, entailment_model):
    batch_size = 1
    entailment_cache = json.load(open(entailment_cache_file, 'r', encoding='utf-8'))

    problem_sentences = []
    for step in steps:
        step_problem_sentences = []
        for premise in step.premises:
            non_hallucinated = False
            if original_text.startswith(premise.strip()) or ". "+premise.strip() in original_text.strip():
                continue
            pair = ("A", "B")
            corresp_dict = {"A": original_text, "B": premise.strip()}
            entailment, entailment_cache = entailment_model.entail([pair], corresp_dict, batch_size, entailment_cache=entailment_cache)
            json.dump(entailment_cache, open(entailment_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
            if entailment != [] and entailment[0][1] == "E":
                non_hallucinated = True
            if not non_hallucinated:
                step_problem_sentences.append(premise.strip())
        problem_sentences.append(step_problem_sentences)
        original_text = original_text + " " + step.conclusion.strip()

    return problem_sentences                

def check_consistency_vanessa(steps, original_text, spacy_model, entailment_model):
    original_sentences = [sentence.strip()+"." for sentence in original_text.split(".")[:-1]]
    original_sentences_dict = {i: sentence for i, sentence in enumerate(original_sentences)}
    problem_sentences = []
    batch_size = 8
    
    entailment_cache = json.load(open(entailment_cache_file, 'r', encoding='utf-8'))
    parsing_cache = json.load(open(parsing_cache_file, 'r', encoding='utf-8'))

    for step in steps:
        step_problem_sentences = []
        n = len(original_sentences_dict)
        for premise in step.premises:
            non_hallucinated = False
            if original_text.startswith(premise.strip()) or ". "+premise.strip() in original_text.strip():
                continue
            for sentence in original_sentences_dict.values():
                #first run parsing
                new_step = ReasoningStep([sentence], premise.strip(), {})
                try:
                    parsing_cache = new_step.logic_transform_tregex(spacy_model, parsing_cache)
                    #new_step.update_parsing_cache(parsing_cache, [sentence, premise.strip()], conclusion=True)
                except Exception as e:
                    pass
                #then run entailment
                try:
                    entailment_cache = new_step.get_entailments(entailment_model, entailment_cache)
                except Exception as e:
                    new_step.entailments_dict = {}
                #then run reasoning
                try:
                    result = new_step.check_conclusion()
                except Exception as e:
                    result = False
                if result == True:
                    non_hallucinated = True
                    break

            if not non_hallucinated:
                step_problem_sentences.append(premise.strip())
        problem_sentences.append(step_problem_sentences)
        original_sentences_dict[n] = step.conclusion.strip()
        original_text = original_text + " " + step.conclusion.strip()
    json.dump(parsing_cache, open(parsing_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(entailment_cache, open(entailment_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
    #input("FINI")
    return problem_sentences

def parse_reasoning(example, spacy_model):
    lines = example["reasoning"]
    steps = read_reasoning(lines.split("\n"))

    errors = []
    steps = [step for step in steps if step.premises != []]

    if steps == []:
        return [], [], [], ["No steps found in reasoning."]
    original_sentences = [[sentence.strip() for sentence in step.premises + [step.conclusion]] for step in steps]
    premises = []
    conclusions = []
    correspondance_dicts = []

    parsing_cache = json.load(open(parsing_cache_file, 'r', encoding='utf-8'))

    for i, step in enumerate(steps):
        if i < 0:
            continue
        #step.logic_transform(spacy_model, parsing_cache)
        #step.logic_transform_tregex(spacy_model)
        try:
            a = None
            parsing_cache = step.logic_transform_tregex(spacy_model, parsing_cache)
            #step.logic_transform(spacy_model, parsing_cache)
        except Exception as e:
            step.premises = []
            step.conclusion = "-1"
            errors.append("Logic Transform Error -  Step n°: " + str(i) + " " + str(e))
        clear_cache()
        premises.append(step.premises)
        conclusions.append(step.conclusion)
        correspondance_dicts.append(step.correspondance_dict)
        """print(step.premises)
        print("-------")
        print(step.conclusion)
        print("-------")
        print(step.correspondance_dict)
        print("-------")
        input()"""
    #print(parsing_cache)
    json.dump(parsing_cache, open(parsing_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
    #input("FINI")
    return premises, conclusions, correspondance_dicts, errors

def get_inconsistencies(example, entailment_model=None, spacy_model=None, direct=False):
    lines = example["reasoning"]
    steps = read_reasoning(lines.split("\n"))

    errors = []
    steps = [step for step in steps if step.premises != []]

    if steps == []:
        return [], ["No steps found in reasoning."]

    try:
        if entailment_model is None:
            print(example["id"])
            consistency_problems = check_consistency_symbolic(steps, example["text"])
        elif spacy_model is None:
            if direct:
                consistency_problems = check_consistency_direct_entailments(steps, example["text"], entailment_model)
            else:
                consistency_problems = check_consistency_entailments(steps, example["text"], entailment_model)
        else:
            consistency_problems = check_consistency_vanessa(steps, example["text"], spacy_model, entailment_model)
    except Exception as e:
        consistency_problems = []
        errors.append("Consistency Checking Error: "+str(e))

    return consistency_problems, errors

def get_entailments(premises, conclusions, correspondance_dicts, entailment_model):
    entailment_cache = json.load(open(entailment_cache_file, 'r', encoding='utf-8'))

    entailments_dicts = []
    errors = []
    print(len(premises))
    for i in range(len(premises)):
        step = ReasoningStep(premises[i], conclusions[i], correspondance_dicts[i])
        #entailment_cache = step.get_entailments(entailment_model)
        try:
            a = None
            entailment_cache = step.get_entailments(entailment_model, entailment_cache)
        except Exception as e:
            step.entailments_dict = {}
            errors.append("Entailments Error -  Step n°: " + str(i) + " " + str(e))
        #input("C'EST FINI LA STOP")
        entailments_dicts.append(step.entailments_dict)
    json.dump(entailment_cache, open(entailment_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
    return entailments_dicts, errors

def get_direct_entailments(example, entailment_model):
    lines = example["reasoning"]
    steps = read_reasoning(lines.split("\n"))
    steps = [step for step in steps if step.premises != []]

    if steps == []:
        return ["No steps found in reasoning."]
    direct_entailments = []
    for step in steps:
        pair = ("A", "B")
        corresp_dict = {"A": "*".join(prem.strip() for prem in step.premises).replace(".*", " and "), "B": step.conclusion}
        entailment, ent_cache = entailment_model.entail([pair], corresp_dict, 1)
        if entailment == []:
            direct_entailments.append(False)
        else:
            if entailment[0][1] == "E":
                direct_entailments.append(True)
            else:
                direct_entailments.append(False)
    return direct_entailments

def get_reasoning(premises, conclusions, entailments_dicts):
    predicted_reasonings = []
    errors = []
    for i in range(len(premises)):
        step = ReasoningStep(premises[i], conclusions[i], {}, entailments_dicts[i])
        #predicted_reasoning = step.check_conclusion()
        try:
            predicted_reasoning = step.check_conclusion()
            a = None
        except ZeroDivisionError as e:
            predicted_reasoning = "-1"
            errors.append("Reasoning Error -  Step n°: " + str(i) + " " + str(e))
        except ValueError as e:
            predicted_reasoning = "Contradiction in the premises"
        except Exception as e:
            predicted_reasoning = "-1"
            errors.append("Reasoning Error -  Step n°: " + str(i) + " " + str(e))
        predicted_reasonings.append(predicted_reasoning)
    return predicted_reasonings, errors


def main(dataset, mode, entailment_model_name=None, dataset_version = None):
    assert dataset in {"FOLIO", "ProntoQA", "ProofWriter", "LogicBench", "EntailmentBank"}
    date_string = "-" + get_date() + ".jsonl"
    global parsing_cache_file
    parsing_cache_file = "cache/parsing_cache_tregex2.json"
    global entailment_cache_file

    if dataset == "FOLIO":
        assert dataset_version in {"LLaMa", "Mixtral", "LLaMa3"}
        if dataset_version == "LLaMa":
            dataset_file = "results/generation/LLaMa_FOLIO.jsonl"
        elif dataset_version == "LLaMa3":
            dataset_file = "results/generation/LLaMa3_FOLIO.jsonl"
        else:
            dataset_file = "results/generation/Mixtral_FOLIO.jsonl"
    elif dataset == "ProofWriter":
        assert dataset_version in {"vanilla", "alt", "alt_2", "alt_3", "remove", "hallu", "neg"}
        if dataset_version == "vanilla":
            dataset_file = "results/generation/ProofWriter.jsonl"
        elif dataset_version == "alt":
            dataset_file = "results/generation/ProofWriter_alt.jsonl"
        elif dataset_version == "alt_3":
            dataset_file = "results/generation/ProofWriter_alt3_test.jsonl"
        elif dataset_version == "remove":
            dataset_file = "results/generation/ProofWriter_remove.jsonl"
        elif dataset_version == "hallu":
            dataset_file = "results/generation/ProofWriter_hallu.jsonl"
        elif dataset_version == "neg":
            dataset_file = "results/generation/ProofWriter_neg.jsonl"
        else:
            dataset_file = "results/generation/ProofWriter_alt2.jsonl"
    elif dataset == "EntailmentBank":
        if dataset_version == "neg":
            dataset_file = "results/generation/EB_neg_NEW.jsonl"
        elif dataset_version == "hallu":
            dataset_file = "results/generation/EB_hallu_NEW.jsonl"
    elif dataset == "LogicBench":
        if dataset_version == "PBD":
            dataset_file = "LogicBench/Vanessa_data/cleaned/PBD.jsonl"
        elif dataset_version == "CT":
            dataset_file = "LogicBench/Vanessa_data/cleaned/CT.jsonl"
        elif dataset_version == "CD":
            dataset_file = "LogicBench/Vanessa_data/cleaned/CD.jsonl"
        elif dataset_version == "DD":
            dataset_file = "LogicBench/Vanessa_data/cleaned/DD.jsonl"
        elif dataset_version == "DS":
            dataset_file = "LogicBench/Vanessa_data/cleaned/DS.jsonl"
        elif dataset_version == "PMT":
            dataset_file = "LogicBench/Vanessa_data/cleaned/PMT.jsonl"
        elif dataset_version == "PHS":
            dataset_file = "LogicBench/Vanessa_data/cleaned/PHS.jsonl"
        elif dataset_version == "PMI":
            dataset_file = "LogicBench/Vanessa_data/cleaned/PMI.jsonl"
    else:
        if dataset_version == "LLaMa":
            dataset_file = "results/generation/LLaMa_ProntoQA.jsonl"
        elif dataset_version == "Mixtral":
            dataset_file = "results/generation/Mixtral_ProntoQA.jsonl"
        elif dataset_version == "LLaMa3":
            dataset_file = "results/generation/LLaMa3_ProntoQA.jsonl"
        else:
            dataset_file = "results/generation/ProntoQA.jsonl"

    assert mode in {"full", "parsing", "entailments", "reasoning", "consistency", "consistency_VANESSA", "direct-entailments", "consistency_direct"}

    if mode == "parsing":
        print("start parsing")

        spacy_model = SpacyModel()
        output_file = dataset_file.replace("/generation/", "/parsing/")
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", date_string)
        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        for i, example in enumerate(dataset):
            if i < 0:
                continue
            print("example", i)
            premises, conclusions, correspondance_dicts, errors = parse_reasoning(example, spacy_model)
            example["logic_premises"] = [[str(premise) for premise in premises_list] for premises_list in premises]
            example["logic_conclusion"] = [str(conclusion) for conclusion in conclusions] 
            example["correspondance"] = correspondance_dicts
            example["errors"] = errors
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")
            #input("fini exemple")

    elif mode == "full":
        assert entailment_model_name in {"T5", "LLaMa", "Symbolic"}

        spacy_model = SpacyModel()
        output_file = dataset_file.replace("/generation/", "/reasoning/").replace(".jsonl", "-" + entailment_model_name + date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]


        for example in dataset:
            premises, conclusions, correspondance_dicts, errors = parse_reasoning(example, spacy_model)
            example["logic_premises"] = str(premises)
            example["logic_conclusion"] = str(conclusions)
            example["correspondance"] = correspondance_dicts
            example["errors"] = errors
            entailment_model = initialize_entailment_model(entailment_model_name)
            entailments_dicts = get_entailments(premises, conclusions, correspondance_dicts, entailment_model)
            example["entailments_dict"] =  [{str(key): str(value) for (key, value) in entailments_dict.items()} for entailments_dict in entailments_dicts]
            example["predicted_steps"] = get_reasoning(premises, conclusions, entailments_dicts)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")

    elif mode == "entailments":
        assert entailment_model_name in {"T5", "LLaMa", "Symbolic", "LLaMaSoft", "Mistral", "LLaMa3", "GPT", "Deberta"}
        entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"

        if dataset == "LogicBench":
            if dataset_version == "PBD":
                dataset_file = "LogicBench/Vanessa_data/PBD-09_25.jsonl"
                dataset_file = "LogicBench/Vanessa_data/PBD-11_13.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PBD-11_23.jsonl"
            elif dataset_version == "CT":
                dataset_file = "LogicBench/Vanessa_data/CT-09_25.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/CT-11_23.jsonl"
            elif dataset_version == "CD":
                dataset_file = "LogicBench/Vanessa_data/CD-09_24.jsonl"
                dataset_file = "LogicBench/Vanessa_data/CD-11_13.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/CD-11_23.jsonl"
            elif dataset_version == "DD":
                dataset_file = "LogicBench/Vanessa_data/DD-09_24.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/DD-11_23.jsonl"
            elif dataset_version == "DS":
                dataset_file = "LogicBench/Vanessa_data/DS-09_24.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/DS-11_23.jsonl"
            elif dataset_version == "PMT":
                dataset_file = "LogicBench/Vanessa_data/PMT-09_25.jsonl"
                dataset_file = "LogicBench/Vanessa_data/PMT-11_13.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PMT-11_23.jsonl"
            elif dataset_version == "PHS":
                dataset_file = "LogicBench/Vanessa_data/PHS-09_27.jsonl"
                dataset_file = "LogicBench/Vanessa_data/PHS-11_05.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PHS-11_23.jsonl"
            elif dataset_version == "PMI":
                dataset_file = "LogicBench/Vanessa_data/PMI-09_09.jsonl"
        else:
            dataset_file = "results/parsing/" + find_most_recent_file("results/parsing/", dataset_file.replace("results/generation/", "").replace(".jsonl", ""))
        output_file = dataset_file.replace("/parsing/", "/entailments/").replace(".jsonl", "-" + entailment_model_name + ".jsonl").replace(".jsonl", date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")
        print(dataset_file)
        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        entailment_model = initialize_entailment_model(entailment_model_name)
        for i, example in enumerate(dataset):
            if i < 0:
                continue
            print("example", i)
            premises = [[parse_formula(premise) for premise in premises_list] for premises_list in example["logic_premises"]]
            
            conclusions = [parse_formula(conclusion) for conclusion in example["logic_conclusion"]]
            correspondance_dicts = example["correspondance"]

            entailments_dicts, errors = get_entailments(premises, conclusions, correspondance_dicts, entailment_model)
            
            example["entailments_dict"] =  [{str(key): [str(val) for val in value] for (key, value) in entailments_dict.items()} for entailments_dict in entailments_dicts]

            example["errors"] = example["errors"] + errors
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")
            #input("Exemple fini !")

    elif mode == "reasoning":
        assert entailment_model_name in {"T5", "LLaMa", "Symbolic", "LLaMaSoft", "Mistral", "LLaMa3", "GPT", "Deberta"}

        if dataset_version is None:
            dataset_version = ""
        if dataset == "ProofWriter":
            if dataset_version == "vanilla":
                dataset_file = "results/entailments/" + find_most_recent_file("results/entailments/", dataset + "*" + entailment_model_name)
            elif dataset_version == "alt_2":
                dataset_file = "results/entailments/" + find_most_recent_file("results/entailments/", dataset + "*" + "alt2" + "*" + entailment_model_name)
            elif dataset_version == "alt_3":
                dataset_file = "results/entailments/" + find_most_recent_file("results/entailments/", dataset + "*" + "alt3" + "*" + entailment_model_name)
            else:
                dataset_file = "results/entailments/" + find_most_recent_file("results/entailments/", dataset + "*" + dataset_version + "*" + entailment_model_name)
        elif dataset == "LogicBench":
            if dataset_version == "PBD":
                dataset_file = "LogicBench/Vanessa_data/PBD-09_25-LLaMa3-09_25.jsonl"
                dataset_file = "LogicBench/Vanessa_data/PBD-09_25-Mistral-09_26.jsonl"
                dataset_file = "LogicBench/Vanessa_data/PBD-11_13-LLaMa3-11_13.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PBD-11_23-LLaMa3-11_23.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PBD-11_23-Deberta-11_23.jsonl"
            elif dataset_version == "CT":
                dataset_file = "LogicBench/Vanessa_data/CT-09_25-LLaMa3-09_25.jsonl"
                dataset_file = "LogicBench/Vanessa_data/CT-09_25-Mistral-09_26.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/CT-11_23-LLaMa3-11_23.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/CT-11_23-Deberta-11_23.jsonl"
            elif dataset_version == "CD":
                dataset_file = "LogicBench/Vanessa_data/CD-09_24-LLaMa3-09_24.jsonl"
                dataset_file = "LogicBench/Vanessa_data/CD-09_24-Mistral-09_26.jsonl"
                dataset_file = "LogicBench/Vanessa_data/CD-11_13-LLaMa3-11_13.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/CD-11_23-LLaMa3-11_23.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/CD-11_23-Deberta-11_23.jsonl"
            elif dataset_version == "DD":
                dataset_file = "LogicBench/Vanessa_data/DD-09_24-LLaMa3-09_25.jsonl"
                dataset_file = "LogicBench/Vanessa_data/DD-09_24-Mistral-09_26.jsonl"
                dataset_file = "LogicBench/Vanessa_data/DD-11_13-LLaMa3-11_13.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/DD-11_23-LLaMa3-11_23.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/DD-11_23-Deberta-11_23.jsonl"
            elif dataset_version == "DS":
                dataset_file = "LogicBench/Vanessa_data/DS-09_24-LLaMa3-09_24.jsonl"
                dataset_file = "LogicBench/Vanessa_data/DS-09_24-Mistral-09_26.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/DS-11_23-LLaMa3-11_23.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/DS-11_23-Deberta-11_23.jsonl"
            elif dataset_version == "PMT":
                dataset_file = "LogicBench/Vanessa_data/PMT-09_25-LLaMa3-09_25.jsonl"
                dataset_file = "LogicBench/Vanessa_data/PMT-09_25-Mistral-09_26.jsonl"
                dataset_file = "LogicBench/Vanessa_data/PMT-11_13-LLaMa3-11_13.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PMT-11_23-LLaMa3-11_23.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PMT-11_23-Deberta-11_23.jsonl"
            elif dataset_version == "PHS":
                dataset_file = "LogicBench/Vanessa_data/PHS-09_27-LLaMa3-09_27.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/PHS-09_27-Mistral-09_27.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/PHS-11_05-LLaMa3-11_06.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PHS-11_23-LLaMa3-11_23.jsonl"
                dataset_file = "LogicBench/Vanessa_data/cleaned/PHS-11_23-Deberta-True.jsonl"
            elif dataset_version == "PMI":
                dataset_file = "LogicBench/Vanessa_data/PMI-09_09-LLaMa3-09_10.jsonl"
        else:
            dataset_file = "results/entailments/" + find_most_recent_file("results/entailments/", dataset_version + "_" + "*" + dataset + "*" + entailment_model_name + "-")

        print(dataset_file)

        output_file = dataset_file.replace("/entailments/", "/reasoning/").replace(".jsonl", date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        for example in dataset:
            if int(example["id"]) < 0:
                continue
            print("example", example["id"])
            prems = [[premise for premise in premises_list] for premises_list in example["logic_premises"]]
            premises = example["logic_premises"]
            conclusions = example["logic_conclusion"]
            entailments_dicts = example["entailments_dict"]
            predicted_steps, errors = get_reasoning(premises, conclusions, entailments_dicts)
            example["predicted_steps"] = predicted_steps
            print(errors, predicted_steps)
            example["errors"] = example["errors"] + errors
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")
    
    elif mode == "consistency" or mode == "consistency_VANESSA" or mode == "consistency_direct":
        assert entailment_model_name in {"T5", "LLaMa", "Symbolic", "LLaMaSoft", "Mistral", "LLaMa3", "GPT", "None", "Deberta"}

        entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"
        output_file = dataset_file.replace("/generation/", "/consistency/")
        output_file = output_file.replace(".jsonl", "-consistency" + date_string)
        if entailment_model_name == "None":
            entailment_model = None
            output_file = output_file.replace("-consistency", "-consistency_Symbolic")
        else:
            output_file = output_file.replace("-consistency", "-consistency_" + entailment_model_name)
            entailment_model = initialize_entailment_model(entailment_model_name)

        print(output_file)
        spacy_model = None
        direct = False
        if mode == "consistency_VANESSA":
            spacy_model = SpacyModel()
            output_file = output_file.replace(".jsonl", "_VANESSA.jsonl")
        elif mode == "consistency_direct":
            output_file = output_file.replace(".jsonl", "_direct.jsonl")
            direct = True

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]
        for i, example in enumerate(dataset):
            if i < 0:
                continue
            inconsistencies, errors = get_inconsistencies(example, entailment_model, spacy_model, direct)
            example["inconsistencies"] = inconsistencies
            example["errors"] = errors
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")
            #input("Exemple fini !")

    elif mode == "direct-entailments":
        assert entailment_model_name in {"T5", "LLaMa", "Symbolic", "LLaMaSoft", "Mistral", "LLaMa3", "GPT", "Deberta"}
        entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"
        entailment_cache_file = "cache/empty.json"

        if dataset == "LogicBench":
            if dataset_version == "CD":
                dataset_file = "LogicBench/Vanessa_data/cleaned/CD.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/CD.jsonl"
            elif dataset_version == "CT":
                dataset_file = "LogicBench/Vanessa_data/cleaned/CT.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/CT.jsonl"
            elif dataset_version == "DD":
                dataset_file = "LogicBench/Vanessa_data/cleaned/DD.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/DD.jsonl"
            elif dataset_version == "PHS":
                dataset_file = "LogicBench/Vanessa_data/cleaned/PHS.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/PHS.jsonl"
            elif dataset_version == "DS":
                dataset_file = "LogicBench/Vanessa_data/cleaned/DS.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/DS.jsonl"
            elif dataset_version == "PBD":
                dataset_file = "LogicBench/Vanessa_data/cleaned/PBD.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/PBD.jsonl"
            elif dataset_version == "PMT":
                dataset_file = "LogicBench/Vanessa_data/cleaned/PMT.jsonl"
                #dataset_file = "LogicBench/Vanessa_data/PMT.jsonl"
        output_file = dataset_file.replace("/generation/", "/reasoning/").replace(".jsonl", "-" + entailment_model_name + "-direct.jsonl").replace(".jsonl", date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")

        print(output_file)

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        entailment_model = initialize_entailment_model(entailment_model_name)

        for i, example in enumerate(dataset):
            if i <0:
                continue
            print("example", i)
            entailments = get_direct_entailments(example, entailment_model)
            print(entailments)
            example["predicted_steps"] = entailments
            example["entailments_dict"] = [{} for _ in entailments]
            example["logic_premises"] = [["A"] for _ in entailments]
            example["logic_conclusion"] = ["B" for _ in entailments]
            example["correspondance"] = [{} for _ in entailments]
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        entailment_model_name = sys.argv[3]
        dataset_version = sys.argv[4]

    elif len(sys.argv) == 3:
        entailment_model_name, dataset_version = None, None
    
    dataset = sys.argv[1]
    mode = sys.argv[2]
    main(dataset, mode, entailment_model_name, dataset_version)
        
    
    