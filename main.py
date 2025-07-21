from entailment_model import initialize_entailment_model
import itertools
from utils import SpacyModel, find_instances, flatten_list, clear_cache, get_date, find_most_recent_file, build_instances_disjunction, transform_truth, coref_step, convert_to_logic
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

    def logic_transform_tregex(self, spacy_model, parsing_cache={}, instances_cache={}):
        print("LOGIC TRANSFORM")
        self.premises = [prem.split("(")[0]+"." if "(" in prem else prem for prem in self.premises]
        self.conclusion = self.conclusion.split("(")[0]+"." if "(" in self.conclusion else self.conclusion
        instances_list, instances_cache = self.detect_instances(spacy_model, instances_cache)
        self = coref_step(self, spacy_model)
        for i, sentence in enumerate(self.premises):
            if sentence in parsing_cache:
                formula, correspondance = parsing_cache[sentence]
                formula = parse_formula(formula)
                transmute_dict = {}
                for key, value in correspondance.items():
                    self.correspondance_dict[self.first_allowed_letter] = value
                    transmute_dict[key] = self.first_allowed_letter
                    self.first_allowed_letter = self.character_list[self.character_list.index(self.first_allowed_letter)+1]
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
        return parsing_cache, instances_cache

    def get_entailments(self, model, entailment_cache={}):
        print("ENTAILMENT TIME")
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
        entailment, entailment_cache = model.entail(contradict_pairs + LL_premises_pairs, self.correspondance_dict, batch_size, contrad_premise=True, entailment_cache=entailment_cache)
        for pair, result in entailment:
            if result == "E":
                self.entailments_dict[Not(Literal(pair[0]))].append(Literal(pair[1]))

        return entailment_cache
        
    def check_conclusion(self):
        if self.conclusion == -1:
            raise ZeroDivisionError("No conclusion found")

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
                            self.correspondance_dict[current_letter] = self.correspondance_dict[char].replace("X", instance)
                        self.first_allowed_letter = self.character_list[self.character_list.index(current_letter)+1]
                    if new_premises != []:
                        formulas_to_add.append(build_instances_disjunction(new_premises))

            for i, prem_remove in enumerate(formulas_to_remove):
                index = self.premises.index(prem_remove)
                self.premises[index] = formulas_to_add[i]

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
                        self.correspondance_dict[current_letter] = self.correspondance_dict[char].replace("X", instance)
                    self.first_allowed_letter = self.character_list[self.character_list.index(current_letter)+1]

                self.conclusion = build_instances_disjunction(formulas_to_add)

        return

    def remove_quantif(self):
        for i, premise in enumerate(self.premises):
            self.premises[i] = premise.remove_quantif()
        self.conclusion = self.conclusion.remove_quantif()

    def clean_formulas(self):
        self.premises = [premise.clean(self.correspondance_dict) for premise in self.premises]
        self.conclusion = self.conclusion.clean(self.correspondance_dict)

def read_reasoning(lines):
    reasoning_steps = []
    premises = []
    for line in lines:
        if ":" in line:
            type = line.split(":")[0].split()[0]
            if type == "Conclusion":
                conclusion = (":".join(line.split(":")[1:])).strip()
                premises = premises
                step = ReasoningStep(premises, conclusion)
                reasoning_steps.append(step)
                premises = []
            elif type == "Premise":
                premises.append((":".join(line.split(":")[1:]).strip()))
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

    entailment_cache = json.load(open(entailment_cache_file, 'r', encoding='utf-8'))
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
            json.dump(entailment_cache, open(entailment_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
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

def check_consistency_FC_entailments(steps, original_text, entailment_model):
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
    instances_cache = json.load(open(instances_cache_file, 'r', encoding='utf-8'))

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
                    parsing_cache, instances_cache = new_step.logic_transform_tregex(spacy_model, parsing_cache, instances_cache)
                    json.dump(parsing_cache, open(parsing_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
                    json.dump(instances_cache, open(instances_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
                except Exception as e:
                    pass
                #then run entailment
                try:
                    entailment_cache = new_step.get_entailments(entailment_model, entailment_cache)
                    json.dump(entailment_cache, open(entailment_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
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
    instances_cache = json.load(open(instances_cache_file, 'r', encoding='utf-8'))

    for i, step in enumerate(steps):
        if i < 0:
            continue
        try:
            parsing_cache, instances_cache = step.logic_transform_tregex(spacy_model, parsing_cache, instances_cache)
            json.dump(parsing_cache, open(parsing_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
            json.dump(instances_cache, open(instances_cache_file, 'w', encoding='utf-8'), ensure_ascii=False)
        except Exception as e:
            step.premises = []
            step.conclusion = "-1"
            errors.append("Logic Transform Error -  Step n°: " + str(i) + " " + str(e))
        clear_cache()
        premises.append(step.premises)
        conclusions.append(step.conclusion)
        correspondance_dicts.append(step.correspondance_dict)

    return premises, conclusions, correspondance_dicts, errors

def get_inconsistencies(example, entailment_model=None, spacy_model=None, FC=False):
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
            if FC:
                consistency_problems = check_consistency_FC_entailments(steps, example["text"], entailment_model)
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
    for i in range(len(premises)):
        step = ReasoningStep(premises[i], conclusions[i], correspondance_dicts[i])
        try:
            entailment_cache = step.get_entailments(entailment_model, entailment_cache)
        except Exception as e:
            step.entailments_dict = {}
            errors.append("Entailments Error -  Step n°: " + str(i) + " " + str(e))
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
    parsing_cache_file = "cache/parsing_cache.json"
    global entailment_cache_file
    global instances_cache_file
    instances_cache_file = "cache/instances_cache.json"

    if dataset == "FOLIO":
        assert dataset_version in {"LLaMa2", "Mixtral", "LLaMa3"}
        if dataset_version == "LLaMa2":
            dataset_file = "results/generation/LLaMa2_FOLIO.jsonl"
        elif dataset_version == "LLaMa3":
            dataset_file = "results/generation/LLaMa3_FOLIO.jsonl"
        else:
            dataset_file = "results/generation/Mixtral_FOLIO.jsonl"
    elif dataset == "ProofWriter":
        assert dataset_version in {"remove", "hallu", "neg"}
        if dataset_version == "remove":
            dataset_file = "results/generation/ProofWriter_remove.jsonl"
        elif dataset_version == "hallu":
            dataset_file = "results/generation/ProofWriter_hallu.jsonl"
        elif dataset_version == "neg":
            dataset_file = "results/generation/ProofWriter_neg.jsonl"
    elif dataset == "EntailmentBank":
        if dataset_version == "neg":
            dataset_file = "results/generation/EB_neg.jsonl"
        elif dataset_version == "hallu":
            dataset_file = "results/generation/EB_hallu.jsonl"
    elif dataset == "ProntoQA":
        if dataset_version == "LLaMa2":
            dataset_file = "results/generation/LLaMa2_ProntoQA.jsonl"
        elif dataset_version == "Mixtral":
            dataset_file = "results/generation/Mixtral_ProntoQA.jsonl"
        elif dataset_version == "LLaMa3":
            dataset_file = "results/generation/LLaMa3_ProntoQA.jsonl"

    assert mode in {"full_validity", "parsing", "entailments", "reasoning", "consistency", "consistency_VANESSA", "direct_LLM", "consistency_FC"}

    if mode == "parsing":
        spacy_model = SpacyModel()
        output_file = dataset_file.replace("/generation/", "/parsing/")
        output_file = output_file.replace(".jsonl", date_string)
        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        for i, example in enumerate(dataset):
            if i < 0:
                continue
            premises, conclusions, correspondance_dicts, errors = parse_reasoning(example, spacy_model)
            example["logic_premises"] = [[str(premise) for premise in premises_list] for premises_list in premises]
            example["logic_conclusion"] = [str(conclusion) for conclusion in conclusions] 
            example["correspondance"] = correspondance_dicts
            example["errors"] = errors
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")

    elif mode == "full_validity":
        assert entailment_model_name in {"LLaMa3", "Symbolic", "Deberta", "Mistral"}
        entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"
        spacy_model = SpacyModel()
        output_file = dataset_file.replace("/generation/", "/reasoning/").replace(".jsonl", "-" + entailment_model_name + date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        entailment_model = initialize_entailment_model(entailment_model_name)
        for i, example in enumerate(dataset):
            if i <0:
                continue
            premises, conclusions, correspondance_dicts, errors = parse_reasoning(example, spacy_model)
            example["logic_premises"] = [[str(premise) for premise in premises_list] for premises_list in premises]
            example["logic_conclusion"] = [str(conclusion) for conclusion in conclusions] 
            example["correspondance"] = correspondance_dicts
            example["errors"] = errors
            entailments_dicts, errors = get_entailments(premises, conclusions, correspondance_dicts, entailment_model)
            example["entailments_dict"] =  [{str(key): [str(val) for val in value] for (key, value) in entailments_dict.items()} for entailments_dict in entailments_dicts]
            example["errors"].append(errors)
            predictions, errors = get_reasoning(premises, conclusions, entailments_dicts)
            example["predicted_steps"] = [True if pred[0] == True else "Contradiction in the premises" if pred[0] == "Contradiction in the premises" else False for pred in predictions]
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")

    elif mode == "entailments":
        assert entailment_model_name in {"LLaMa3", "Symbolic", "Deberta", "Mistral"}
        entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"

        dataset_file = "results/parsing/" + find_most_recent_file("results/parsing/", dataset_file.replace("results/generation/", "").replace(".jsonl", ""))
        output_file = dataset_file.replace("/parsing/", "/entailments/").replace(".jsonl", "-" + entailment_model_name + ".jsonl").replace(".jsonl", date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")
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

    elif mode == "reasoning":
        assert entailment_model_name in {"LLaMa3", "Symbolic", "Deberta"}

        dataset_file = "results/entailments/" + find_most_recent_file("results/entailments/", dataset_version + "_" + "*" + dataset + "*" + entailment_model_name + "-")
        output_file = dataset_file.replace("/entailments/", "/reasoning/").replace(".jsonl", date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        for example in dataset:
            if int(example["id"]) < 0:
                continue
            prems = [[premise for premise in premises_list] for premises_list in example["logic_premises"]]
            premises = example["logic_premises"]
            conclusions = example["logic_conclusion"]
            entailments_dicts = example["entailments_dict"]
            predicted_steps, errors = get_reasoning(premises, conclusions, entailments_dicts)
            example["predicted_steps"] = [True if pred[0] == True else "Contradiction in the premises" if pred[0] == "Contradiction in the premises" else False for pred in predicted_steps]
            example["errors"] = example["errors"] + errors
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")
    
    elif mode == "consistency" or mode == "consistency_VANESSA" or mode == "consistency_FC":
        assert entailment_model_name in {"LLaMa3", "Symbolic", "Deberta", "Mistral", "None"}

        entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"
        output_file = dataset_file.replace("/generation/", "/consistency/")
        output_file = output_file.replace(".jsonl", "-consistency" + date_string)
        if entailment_model_name == "None":
            entailment_model = None
            output_file = output_file.replace("-consistency", "-consistency_Symbolic")
        else:
            output_file = output_file.replace("-consistency", "-consistency_" + entailment_model_name)
            entailment_model = initialize_entailment_model(entailment_model_name)

        spacy_model = None
        FC = False
        if mode == "consistency_VANESSA":
            spacy_model = SpacyModel()
            output_file = output_file.replace(".jsonl", "_VANESSA.jsonl")
        elif mode == "consistency_FC":
            output_file = output_file.replace(".jsonl", "_FC.jsonl")
            FC = True

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]
        for i, example in enumerate(dataset):
            if i < 0:
                continue
            inconsistencies, errors = get_inconsistencies(example, entailment_model, spacy_model, FC)
            example["inconsistencies"] = inconsistencies
            example["errors"] = errors
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(example, ensure_ascii=False)+"\n")

    elif mode == "direct_LLM":
        assert entailment_model_name in {"LLaMa3", "Symbolic", "Deberta"}
        entailment_cache_file = "cache/" + entailment_model_name + "_cache.json"

        output_file = dataset_file.replace("/generation/", "/reasoning/").replace(".jsonl", "-" + entailment_model_name + "-direct_LLM.jsonl").replace(".jsonl", date_string)
        if os.path.exists(output_file):
            output_file = output_file.replace(".jsonl", "_2.jsonl")

        with open(dataset_file, 'r') as f:
            data = f.readlines()
        dataset = [json.loads(d) for d in data]

        entailment_model = initialize_entailment_model(entailment_model_name)

        for i, example in enumerate(dataset):
            if i <0:
                continue
            entailments = get_direct_entailments(example, entailment_model)
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
        
    
    