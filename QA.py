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
import string
import time

class ReasoningStep():
    def __init__(self, premises, conclusion, correspondance_dict = None, entailments_dict = None):
        self.premises = premises.copy()
        self.conclusion = conclusion
        self.character_list = string.printable[:68] + string.printable[71:-6] + "".join([chr(i) for i in range(200, 400)])
        self.first_allowed_letter = self.character_list[0]
        
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

    def logic_transform_tregex(self, spacy_model, parsing_cache={}, instances_cache={}):
        print("LOGIC TRANSFORM")
        print(self.premises)
        self.premises = [prem.split("(")[0]+"." if "(" in prem else prem for prem in self.premises]
        self.conclusion = self.conclusion.split("(")[0]+"." if "(" in self.conclusion else self.conclusion
        instances_list, instances_cache = self.detect_instances(spacy_model, instances_cache)
        self = coref_step(self, spacy_model)
        print("APRES COREF", self.premises)
        for i, sentence in enumerate(self.premises):
            if sentence in parsing_cache:
                formula, correspondance = parsing_cache[sentence]
                formula = parse_formula(formula)
                transmute_dict = {}
                for key, value in correspondance.items():
                    self.correspondance_dict[self.first_allowed_letter] = value
                    transmute_dict[key] = self.first_allowed_letter
                    self.first_allowed_letter = self.character_list[self.character_list.index(self.first_allowed_letter)+1]
                    #self.first_allowed_letter = chr(ord(self.first_allowed_letter)+1)
                self.premises[i] = formula.transmute(transmute_dict)
            else:
                a = process_tree(sentence, spacy_model, universal=True)
                formula, dict_update, self.first_allowed_letter = convert_to_logic(a, {}, self.first_allowed_letter, spacy_model, self.character_list)

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
                    self.first_allowed_letter = self.character_list[self.character_list.index(self.first_allowed_letter)+1]
                    #self.first_allowed_letter = chr(ord(self.first_allowed_letter)+1)
                self.conclusion = formula.transmute(transmute_dict)
            else:
                a = process_tree(self.conclusion, spacy_model, universal=True)
                formula, dict_update, self.first_allowed_letter = convert_to_logic(a, {}, self.first_allowed_letter, spacy_model, self.character_list)

                parsing_cache[self.conclusion] = [str(formula), {key: value for (key, value) in dict_update.items()}]
                self.conclusion = formula
                self.correspondance_dict.update(dict_update)
        else:
            self.conclusion = Literal("0")
        self.instantiate_formulas(spacy_model, instances_list)
        self.clean_formulas()
        print(self.premises, self.conclusion, self.correspondance_dict)
        return parsing_cache, instances_cache

    def get_entailments(self, model, entailment_cache={}):
        print("ENTAILMENT TEMPS")
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
        print("CP+LLP", len(contradict_pairs + LL_premises_pairs))
        #input("TOUS LES ENTAILMENTS SONT FAITS")
        entailment, entailment_cache = model.entail(contradict_pairs + LL_premises_pairs, self.correspondance_dict, batch_size, contrad_premise=True, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "E":
                self.entailments_dict[Not(Literal(pair[0]))].append(Literal(pair[1]))
        
        print("-------Negation---------")
        #contradict_pairs: RC -> LP
        entailment, entailment_cache = model.entail(contradict_pairs, self.correspondance_dict, batch_size, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "E":
                self.entailments_dict[Literal(pair[0])].append(Literal(pair[1]))

        #pairs for conc negation: -LC -> LP
        entailment, entailment_cache = model.entail(left_concs_pairs, self.correspondance_dict, batch_size, contrad_premise=True, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "E":
                self.entailments_dict[Not(Literal(pair[0]))].append(Literal(pair[1]))

        #pairs for conc negation - contradiction: -LC -/-> RP & RP -/-> RC
        left_concs_right_pairs = list(itertools.product(left_concs, right_premises)) # -LC -> RP
        entailment, entailment_cache = model.entail(left_concs_right_pairs, self.correspondance_dict, batch_size, contrad_premise=True, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "C":
                self.entailments_dict[Not(Literal(pair[0]))].append(Not(Literal(pair[1])))
        
        #RP -> -RC 
        entailment, entailment_cache = model.entail(right_concs_pairs, self.correspondance_dict, batch_size, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "C":
                self.entailments_dict[Literal(pair[0])].append(Not(Literal(pair[1])))

        return entailment_cache
        
    def check_conclusion(self):
        if self.conclusion == -1:
            raise ZeroDivisionError("No conclusion found")
        """print("CHECK CONCLUSION")
        print("Premises:", self.premises)
        print("Conclusion:", self.conclusion)"""

        proofs = build_proofs(self.premises, self.conclusion, self.entailments_dict)
        proof = proofs[0]
        try:
            result, proof_lines = proof.verify()
        except ValueError:
            result = "Contradiction in the premises"
            proof_lines = []
        except FunctionTimedOut:
            result = "Uncertain"
            proof_lines = []

        return result, proof_lines
        
        if all([result == "Contradiction in the premises" for result in results]):
            raise ValueError("Contradiction in the premises")
        if any([result == True for result in results]):
            return True
        if any([result == False for result in results]):
            return False
        return "Uncertain"
  
    def detect_instances(self, spacy_model, instances_cache):
        instances_list = []
        for text in self.premises+[self.conclusion]:
            if text in instances_cache:
                if instances_cache[text] == []:
                    continue
                instances_list.extend(instances_cache[text])
            else:
                instances = find_instances(text, spacy_model)
                for instance in instances:
                    if instance.lower().split()[0] in {"some","someone", "something", "somebody", "somewhere", "a", "an"}:
                        pass
                    else:
                        instances_list.append(re.sub("`|'", "", instance).strip())
                        if text in instances_cache:
                            instances_cache[text].append(re.sub("`|'", "", instance).strip())
                        else:
                            instances_cache[text] = [re.sub("`|'", "", instance).strip()]
                if len(instances) == 0:
                    instances_cache[text] = []
        return instances_list, instances_cache

    def instantiate_formulas(self, spacy_model, instances_list):
        print("INSTANTIATE FORMULAS")
        universal_present = [premise.is_universal() for premise in self.premises] + [self.conclusion.is_universal()]
        if all(universal_present):
            self.remove_quantif()
            return
        if not any(universal_present):  
            return
        else:
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
            univ_premises_nb = 0
            for premise in self.premises + [self.conclusion]:
                break
                if premise.is_universal():
                    univ_premises_nb += 1
                    premise_keys = flatten_list(premise.get_variables())
                    for key in premise_keys:
                        for instance in instances_list:
                            if re.search(r"(?i)"+" "+instance+" ", " "+self.correspondance_dict[key]):
                                not_viable_instances.add(instance)
                                break
            if univ_premises_nb == 1 and len(set(instances_list) - not_viable_instances) == 0:
                not_viable_instances = set()
            instances_list = list(set(instances_list) - not_viable_instances)
            print("INSTALCES LIST apres CHECK", instances_list)
            if instances_list == []:
                self.remove_quantif()
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
                        new_formula, _, chars = new_formula.instantiate(self.first_allowed_letter, self.character_list)
                        new_premises.append(new_formula)
                        for i, char in enumerate(chars):
                            current_letter = self.character_list[self.character_list.index(self.first_allowed_letter)+i]
                            #current_letter = chr(ord(self.first_allowed_letter) + i)
                            self.correspondance_dict[current_letter] = self.correspondance_dict[char].replace("X", instance)
                        self.first_allowed_letter = self.character_list[self.character_list.index(current_letter)+1]
                        #self.first_allowed_letter = chr(ord(current_letter)+1)
                    if new_premises != []:
                        formulas_to_add.append(build_instances_disjunction(new_premises))

            for i, prem_remove in enumerate(formulas_to_remove):
                index = self.premises.index(prem_remove)
                self.premises[index] = formulas_to_add[i]
            """self.premises = [prem for prem in self.premises if prem not in formulas_to_remove]
            self.premises.extend(formulas_to_add)"""

            if self.conclusion.is_universal():
                instance_truth_quantif = ""
                if self.conclusion in instances_truth_quantif:
                    instance_truth_quantif, key_truth_quantif = instances_truth_quantif[self.conclusion]
                formulas_to_add = []
                for instance in instances_list:
                    new_formula = self.conclusion.copy()
                    if instance == instance_truth_quantif:
                        new_formula = transform_truth(new_formula, key_truth_quantif)
                    new_formula, _, chars = new_formula.instantiate(self.first_allowed_letter, self.character_list)
                    formulas_to_add.append(new_formula)
                    for i, char in enumerate(chars):
                        current_letter = self.character_list[self.character_list.index(self.first_allowed_letter)+i]
                        #current_letter = chr(ord(self.first_allowed_letter) + i)
                        self.correspondance_dict[current_letter] = self.correspondance_dict[char].replace("X", instance)
                    #self.first_allowed_letter = chr(ord(current_letter)+1)
                    self.first_allowed_letter = self.character_list[self.character_list.index(current_letter)+1]

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

def process_instance(example, spacy_model, entailment_model, entailment_cache_file, parsing_cache_file, instances_cache_file):
    context = example["text"]
    context = context.replace("No.", "No")
    sentences = [s.strip()+"." for s in context.split(".") if s.strip() != ""]
    conclusion = example["question"]
    if conclusion.startswith("Is it true that "):
        conclusion = conclusion[16:]
    if conclusion.endswith("?"):
        conclusion = conclusion[:-1]
        if conclusion[-1] != '.':
            conclusion = conclusion+"."
    example_step = ReasoningStep(sentences, conclusion)
    parsing_cache = json.load(open(parsing_cache_file, 'r', encoding='utf-8'))
    entailment_cache = json.load(open(entailment_cache_file, 'r', encoding='utf-8'))
    instances_cache = json.load(open(instances_cache_file, 'r', encoding='utf-8'))
    #parsing_cache, instances_cache = example_step.logic_transform_tregex(spacy_model, parsing_cache, instances_cache)
    try:
        parsing_cache, instances_cache = example_step.logic_transform_tregex(spacy_model, parsing_cache, instances_cache)
        json.dump(parsing_cache, open(parsing_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(instances_cache, open(instances_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
    except Exception as e:
        example_step.premises = []
        return "Uncertain", [], "-1", {}, {}, [], ["Logic Transform Error: "+str(e)]
    #entailment_cache = example_step.get_entailments(entailment_model, entailment_cache)
    try:
        entailment_cache = example_step.get_entailments(entailment_model, entailment_cache)
        json.dump(entailment_cache, open(entailment_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
    except Exception as e:
        example_step.entailments_dict = {}
        return "Uncertain", example_step.premises, example_step.conclusion, example_step.correspondance_dict, {}, [], ["Entailments Error: "+str(e)]
    print(example_step.premises)
    print(example_step.conclusion)
    print(example_step.correspondance_dict)
    print(example_step.entailments_dict)
    #result = example_step.check_conclusion()
    try:
        result, proof_lines = example_step.check_conclusion()
    except ZeroDivisionError as e:
        return "Uncertain", example_step.premises, example_step.conclusion, example_step.correspondance_dict, example_step.entailments_dict, [], ["Reasoning TimeOut"]
    except ValueError as e:
        return "Uncertain", example_step.premises, example_step.conclusion, example_step.correspondance_dict, example_step.entailments_dict, [], ["Contradiction in the premises"]
    except Exception as e:
        return "Uncertain", example_step.premises, example_step.conclusion, example_step.correspondance_dict, example_step.entailments_dict, [], ["Reasoning Error: "+str(e)]
    return result, example_step.premises, example_step.conclusion, example_step.correspondance_dict, example_step.entailments_dict, proof_lines, []
    
    
def main(dataset, entailment_model_name=None, dataset_version = None):
    assert dataset in {"FOLIO", "ProntoQA", "ProofWriter", "LogicBench", "EntailmentBank", "Web"}
    date_string = "-" + get_date() + ".jsonl"
    global parsing_cache_file
    parsing_cache_file = "cache/parsing_cache_final.json"
    global entailment_cache_file
    global instances_cache_file
    instances_cache_file = "cache/instances_cache.json"


    if dataset == "FOLIO":
        assert dataset_version in {"LLaMa", "Mixtral", "LLaMa3"}
        if dataset_version == "LLaMa":
            dataset_file = "results/generation/LLaMa_FOLIO.jsonl"
        elif dataset_version == "LLaMa3":
            dataset_file = "results/generation/LLaMa3_FOLIO.jsonl"
        else:
            dataset_file = "results/generation/test.jsonl"
    elif dataset == "ProofWriter":
        assert dataset_version in {"remove", "hallu", "neg", "QA"}
        if dataset_version == "remove":
            dataset_file = "results/generation/ProofWriter_remove.jsonl"
        elif dataset_version == "hallu":
            dataset_file = "results/generation/ProofWriter_hallu.jsonl"
        elif dataset_version == "neg":
            dataset_file = "results/generation/ProofWriter_neg.jsonl"
        elif dataset_version == "QA":
            dataset_file = "results/generation/ProofWriter_QA.jsonl"
            #dataset_file = "results/generation/test.jsonl"
        else:
            dataset_file = "results/generation/ProofWriter_alt2.jsonl"
    elif dataset == "EntailmentBank":
        if dataset_version == "neg":
            dataset_file = "results/generation/EB_neg_NEW.jsonl"
        elif dataset_version == "hallu":
            dataset_file = "results/generation/EB_hallu_NEW.jsonl"
    elif dataset == "LogicBench":
        if dataset_version == "PBD":
            dataset_file = "LogicBench/Vanessa_data/PBD.jsonl"
        elif dataset_version == "PCT":
            dataset_file = "LogicBench/Vanessa_data/CT.jsonl"
        elif dataset_version == "PCD":
            dataset_file = "LogicBench/Vanessa_data/CD.jsonl"
        elif dataset_version == "PDD":
            dataset_file = "LogicBench/Vanessa_data/DD.jsonl"
        elif dataset_version == "PDS":
            dataset_file = "LogicBench/Vanessa_data/DS.jsonl"
        elif dataset_version == "PMT":
            dataset_file = "LogicBench/Vanessa_data/PMT.jsonl"
        elif dataset_version == "PHS":
            dataset_file = "LogicBench/Vanessa_data/PHS.jsonl"
        elif dataset_version == "PMI":
            dataset_file = "LogicBench/Vanessa_data/PMI.jsonl"
        elif dataset_version == "FMT":
            dataset_file = "LogicBench/Vanessa_data/FMT.jsonl"
        elif dataset_version == "FBD":
            dataset_file = "LogicBench/Vanessa_data/FBD.jsonl"
        elif dataset_version == "FDS":
            dataset_file = "LogicBench/Vanessa_data/FDS.jsonl"
        elif dataset_version == "FHS":
            dataset_file = "LogicBench/Vanessa_data/FHS.jsonl"
        elif dataset_version == "FDD":
            dataset_file = "LogicBench/Vanessa_data/FDD.jsonl"
        elif dataset_version == "FCD":
            dataset_file = "LogicBench/Vanessa_data/FCD.jsonl"
        elif dataset_version == "FMP":
            dataset_file = "LogicBench/Vanessa_data/FMP.jsonl"
    elif dataset == "Web":
        dataset_file = "webapp/input.json"
    else:
        if dataset_version == "LLaMa":
            dataset_file = "results/generation/LLaMa_ProntoQA.jsonl"
        elif dataset_version == "Mixtral":
            dataset_file = "results/generation/Mixtral_ProntoQA.jsonl"
        elif dataset_version == "LLaMa3":
            dataset_file = "results/generation/LLaMa3_ProntoQA.jsonl"
        elif dataset_version == "QA":
            dataset_file = "results/generation/QA_ProntoQA.jsonl"
        else:
            dataset_file = "results/generation/ProntoQA.jsonl"

    start_time = time.time()
    assert entailment_model_name in {"LLaMa3", "Symbolic", "Deberta", "oLLaMa3", "Gemma2", "T5", "Ministral"}
    entailment_model = initialize_entailment_model(entailment_model_name)
    entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"

    spacy_model = SpacyModel()
    output_file = dataset_file.replace("/generation/", "/direct_QA/reasoning/").replace("LogicBench/Vanessa_data/cleaned/", "results/direct_QA/reasoning/").replace("LogicBench/Vanessa_data/", "results/direct_QA/reasoning/").replace(".jsonl", "-VANESSA-" + entailment_model_name + date_string)
    if os.path.exists(output_file):
        output_file = output_file.replace(".jsonl", "_2.jsonl")

    with open(dataset_file, 'r') as f:
        data = f.readlines()
    dataset = [json.loads(d) for d in data]

    loaded_models = time.time()
    for i, example in enumerate(dataset):
        if i < 0:
            continue
        result, premises, conclusion, correspondance_dict, entailments_dict, _, errors = process_instance(example, spacy_model, entailment_model, entailment_cache_file, parsing_cache_file, instances_cache_file)
        print("RESULT", result)
        print("ERRORS", errors)
        #example["logic_premises"] = str(premises)
        #example["logic_conclusion"] = str(conclusion)
        #example["correspondance"] = correspondance_dict
        example["errors"] = errors
        #example["entailments_dict"] =  entailments_dict
        example["predicted_instance"] = result
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(example, ensure_ascii=False)+"\n")

        finish_time = time.time()
        print("Time to load models", loaded_models-start_time)
        print("Time to process", finish_time-loaded_models)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        entailment_model_name = sys.argv[3]
        dataset_version = sys.argv[4]

    elif len(sys.argv) == 2:
        entailment_model_name, dataset_version = None, None
    
    dataset = sys.argv[1]
    mode = sys.argv[2]
    main(dataset, entailment_model_name, dataset_version)