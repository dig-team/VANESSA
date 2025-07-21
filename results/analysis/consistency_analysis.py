import json
import numpy as np


def load_data(file, ground_truth_file):
    with open(file, 'r') as f:
            data = f.readlines()
    dataset = [json.loads(d) for d in data]
    new_dataset = []
    print("file", file, ground_truth_file)
    if ground_truth_file is not None:
        print("loading ground truth")
        with open(ground_truth_file, 'r') as f:
            annotations_data = f.readlines()
        annotations_dataset = [json.loads(d) for d in annotations_data]
    else:
        annotations_dataset = dataset.copy()

    for elem, annotations in zip(dataset, annotations_dataset):
        try:
            text = elem["text"]
            id = elem["id"]
            ground_truth_hallucinations = []
            reasoning_steps = []
            premises = []
            instance_premises = []
            for line in elem["reasoning"].split("\n"):
                if ":" in line:
                    type = line.split(":")[0].split()[0]
                    if type == "Conclusion":
                        conclusion = ":".join(line.split(":")[1:]).strip()
                        if "(" in conclusion:
                            conclusion = conclusion.split("(")[0]+"."
                        if premises != []:
                            reasoning_steps.append((premises, conclusion))
                        premises = []
                    elif type == "Premise":
                        premises.append(":".join(line.split(":")[1:]).strip())
                    else:
                        continue
            for i, step in enumerate(reasoning_steps):
                step_ground_truth = []
                instance_premises.append(step[0])
                for premise in step[0]:
                    if premise not in text and premise.replace(" and ", ", ") not in text:
                        step_ground_truth.append("y")
                    else:
                        step_ground_truth.append("n")
                text = text + " " + step[1]
                ground_truth_hallucinations.append(step_ground_truth)
            
            if "hallucinated_premises" in annotations:
                ground_truth_hallucinations = annotations["hallucinated_premises"]
            
            steps_hallucinations = []
            for i, step_inconsistency in enumerate(elem["inconsistencies"]):
                hallus = [False for _ in range(len(ground_truth_hallucinations[i]))]
                for inconsistency in step_inconsistency:
                    if "cold and " in inconsistency or "kind and " in inconsistency:
                        continue
                    index = instance_premises[i].index(inconsistency)
                    hallus[index] = True
                steps_hallucinations.append(hallus)

            ground_truth_reasoning = elem["ground_truth"]
            text_reasoning = elem["reasoning"]

            inst = {"id": id, "ground_truth_hallucinations": ground_truth_hallucinations, "ground_truth_reasoning": ground_truth_reasoning, "predicted_hallucinations": steps_hallucinations, "text_reasoning": text_reasoning}
            new_dataset.append(inst)
        except ZeroDivisionError as e:
            continue
    return new_dataset

def check_steps(dataset):
    steps_results = []
    steps_counts = {True: 0, False: 0, "Error": 0}
    steps_ground_truth_counts = {True: 0, False: 0}
    for i, instance in enumerate(dataset):
        steps_result = []
        for j, pred in enumerate(instance["predicted_hallucinations"]):
            try:
                truth = instance["ground_truth_hallucinations"][j]
            except IndexError:
                continue
            if True in pred:
                steps_counts[False] += 1
                if "y" in truth:
                    steps_ground_truth_counts[False] += 1
                    steps_result.append("TN")
                else:
                    steps_ground_truth_counts[True] += 1
                    steps_result.append("FN")
            else:
                steps_counts[True] += 1
                if "y" not in truth:
                    steps_ground_truth_counts[True] += 1
                    steps_result.append("TP")
                else:
                    steps_ground_truth_counts[False] += 1
                    steps_result.append("FP")
        steps_results.append(steps_result)
    
    return steps_results, steps_counts, steps_ground_truth_counts

def get_f1_scores(steps_results):
    steps = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for steps_result in steps_results:
        for step_result in steps_result:
            if step_result == "TP":
                steps["TP"] += 1
            elif step_result == "FP":
                steps["FP"] += 1
            elif step_result == "FN":
                steps["FN"] += 1
            elif step_result == "TN":
                steps["TN"] += 1
    print(steps)
    try:
        precision = wilson_conf_interval(steps["TP"]/(steps["TP"]+steps["FP"]), steps["TP"]+steps["FP"])
        recall = wilson_conf_interval(steps["TP"]/(steps["TP"]+steps["FN"]), steps["TP"]+steps["FN"])
        accuracy = wilson_conf_interval((steps["TP"]+steps["TN"])/(steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"]), steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"])
        f05 = wilson_conf_interval((1+0.5**2)*precision[0]*recall[0]/((0.5**2)*precision[0]+recall[0]), (1+0.5**2)*steps["TP"]+0.5**2*steps["FN"]+steps["FP"])
        f1 = wilson_conf_interval(2*precision[0]*recall[0]/(precision[0]+recall[0]), 2*steps["TP"]+steps["FN"]+steps["FP"])
        somers_d = (steps["TP"]*steps["TN"]-steps["FP"]*steps["FN"])/((steps["TP"]+steps["FP"])*(steps["TN"]+steps["FN"])*(steps["TN"]+steps["FP"])*(steps["TP"]+steps["FN"]))**0.5
    except ZeroDivisionError:
        precision, recall, accuracy, f05, f1, somers_d = (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), 0
    return precision, recall, accuracy, f05, f1, somers_d

def wilson_conf_interval(p, n, z=1.96):
	denominator = 1 + z*z/n
	center_adjusted_probability = (p + z*z / (2*n)) / denominator
	adjusted_standard_deviation = (np.sqrt((p*(1 - p) + z*z / (4*n)) / n)) / denominator

	return center_adjusted_probability, adjusted_standard_deviation

def get_entailments_nb(dataset):
    count = 0
    for example in dataset:
        entailments = example["entailments"]
        for entailment_dict in entailments:
            for key in entailment_dict:
                count += len(entailment_dict[key])
    return count

def get_f1(file, ground_truth_file=None):
    dataset = load_data(file, ground_truth_file)
    steps_results, counts, ground_truth_counts = check_steps(dataset)
    print("counts", counts)
    print("ground_truth_counts", ground_truth_counts)
    precision, recall, accuracy, f05, f1, somers_d = get_f1_scores(steps_results)
    print("Category", "Accuracy", "Precision", "Recall", "F05", "F1")
    try:
        print("Steps:", accuracy, precision, recall, f05, f1) 
    except ZeroDivisionError:
        print("Steps:", accuracy, precision, recall, 0, 0, 0)   
    print("Somer's D", somers_d)
    print("String:", f"{file} & {round(precision[0], 2)}$_{{\pm \\text{{{round(precision[1], 2)}}}}}$ & {round(recall[0], 2)}$_{{\pm \\text{{{round(recall[1], 2)}}}}}$ & {round(f05[0], 2)}$_{{\pm \\text{{{round(f05[1], 2)}}}}}$ & {round(somers_d, 2)}")

def get_all_f1s():
    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_Symbolic-12_04.jsonl"
    print("Symbolic")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_Symbolic-12_10_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_LLaMa3-11_27_VANESSA.jsonl"
    print("LLaMa3 VANESSA")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa3 Direct")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_LLaMa3-12_02.jsonl"
    print("LLaMa3 Single")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_Deberta-12_02_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_FOLIO-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file, "LLaMa3_FOLIO_annotations_full.jsonl")


    print("######################")
    print("######################")
    print("######################")
    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_Symbolic-12_02.jsonl"
    print("Symbolic")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_Symbolic-12_10_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")


    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_LLaMa3-11_28_VANESSA.jsonl"
    print("LLaMa VANESSA")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa Direct")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_LLaMa3-12_02.jsonl"
    print("LLaMa Single")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")
    
    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_Deberta-12_03_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_FOLIO-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file, "LLaMa2_FOLIO_annotations_full.jsonl")

    print("######################")
    print("######################")
    print("######################")
    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_Symbolic-12_02.jsonl"
    print("Symbolic")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_Symbolic-12_11_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")


    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_LLaMa3-11_28_VANESSA.jsonl"
    print("LLaMa VANESSA")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa Direct")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_LLaMa3-12_02.jsonl"
    print("LLaMa Single")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_Deberta-12_03_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_FOLIO-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file, "Mixtral_FOLIO_annotations_full.jsonl")

    """
    print("######################")
    print("######################")
    print("######################")
    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_Symbolic-12_02.jsonl"
    print("Symbolic")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_Symbolic-12_11_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_LLaMa3-11_29_VANESSA.jsonl"
    print("LLaMa VANESSA")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa Direct")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_LLaMa3-12_02.jsonl"
    print("LLaMa Single")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_Deberta-12_03_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa2_ProntoQA-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file, "LLaMa2_ProntoQA_annotations.jsonl")

    print("######################")
    print("######################")
    print("######################")
    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_Symbolic-12_02.jsonl"
    print("Symbolic")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_Symbolic-12_11_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_LLaMa3-11_29_VANESSA.jsonl"
    print("LLaMa VANESSA")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")
    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa Direct")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_LLaMa3-12_02.jsonl"
    print("LLaMa Single")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_Deberta-12_05_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/Mixtral_ProntoQA-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file, "Mixtral_ProntoQA_annotations.jsonl")


    print("######################")
    print("######################")
    print("######################")
    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_Symbolic-12_02.jsonl"
    print("Symbolic")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_Symbolic-12_11_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_LLaMa3-11_27_VANESSA.jsonl"
    print("LLaMa VANESSA")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa Direct")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_LLaMa3-12_02.jsonl"
    print("LLaMa Single")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_Deberta-12_03_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    print("-------------------")
    file = "../consistency/LLaMa3_ProntoQA-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file, "LLaMa3_ProntoQA_annotations.jsonl")

    
    print("######################")
    print("######################")
    print("######################")
    print("-------------------")

    file = "../consistency/ProofWriter_alt-consistency_Symbolic-12_02.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt-consistency_Symbolic-12_11_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt-consistency_LLaMa3-11_29_VANESSA.jsonl"
    print("LLaMa3 VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa3 Direct")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt-consistency_LLaMa3-12_08.jsonl"
    print("LLaMa3 Single")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt-consistency_Deberta-12_03_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file)

    
    print("######################")
    print("######################")
    print("######################")
    print("-------------------")

    file = "../consistency/ProofWriter_alt2-consistency_Symbolic-12_02.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt2-consistency_Symbolic-12_11_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt2-consistency_LLaMa3-12_01_VANESSA.jsonl"
    print("LLaMa3 VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt2-consistency_LLaMa3-12_02_direct.jsonl"
    print("LLaMa3 Direct")
    get_f1(file)


    print("-------------------")
    file = "../consistency/ProofWriter_alt2-consistency_LLaMa3-12_02.jsonl"
    print("LLaMa3 Single")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt2-consistency_Deberta-12_04_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt2-consistency_Deberta-12_05_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_alt2-consistency_Deberta-12_05.jsonl"
    print("DeBERTA Single")
    get_f1(file)

    print("######################")
    print("######################")
    print("######################")
    print("-------------------")

    file = "../consistency/EB_hallu_NEW-consistency_Symbolic-12_12.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_hallu_NEW-consistency_Symbolic-12_13_VANESSA.jsonl"
    print("Symbolic VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_hallu_NEW-consistency_LLaMa3-12_13_VANESSA.jsonl"
    print("LLaMa3 VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_hallu_NEW-consistency_LLaMa3-12_12_direct.jsonl"
    print("LLaMa3 Direct")
    get_f1(file)


    print("-------------------")
    file = "../consistency/EB_hallu_NEW-consistency_LLaMa3-12_12.jsonl"
    print("LLaMa3 Single")
    get_f1(file)


    print("-------------------")
    file = "../consistency/EB_hallu_NEW-consistency_Deberta-12_13_VANESSA.jsonl"
    print("DeBERTa VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_hallu_NEW-consistency_Deberta-12_12_direct.jsonl"
    print("DeBERTa Direct")
    get_f1(file)


    print("-------------------")
    file = "../consistency/EB_hallu_NEW-consistency_Deberta-12_12.jsonl"
    print("DeBERTa Single")
    get_f1(file)

    print("######################")
    print("######################")
    print("######################")
    print("-------------------")

    file = "../consistency/EB_neg_NEW-consistency_Symbolic-12_12.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_neg_NEW-consistency_Symbolic-12_13_VANESSA.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_neg_NEW-consistency_LLaMa3-12_13_VANESSA.jsonl"
    print("LLaMa3 VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_neg_NEW-consistency_LLaMa3-12_12_direct.jsonl"
    print("LLaMa3 Direct")
    get_f1(file)


    print("-------------------")
    file = "../consistency/EB_neg_NEW-consistency_LLaMa3-12_12.jsonl"
    print("LLaMa3 Single")
    get_f1(file)


    print("-------------------")
    file = "../consistency/EB_neg_NEW-consistency_Deberta-12_13_VANESSA.jsonl"
    print("DeBERTa VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/EB_neg_NEW-consistency_Deberta-12_12_direct.jsonl"
    print("DeBERTa Direct")
    get_f1(file)


    print("-------------------")
    file = "../consistency/EB_neg_NEW-consistency_Deberta-12_12.jsonl"
    print("DeBERTa Single")
    get_f1(file, "../generation/EB_neg_NEWW.jsonl")


    print("######################")
    print("######################")
    print("######################")
    print("-------------------")

    file = "../consistency/ProofWriter_hallu-consistency_Symbolic-12_11.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_hallu-consistency_Symbolic-12_12_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_hallu-consistency_LLaMa3-12_12_VANESSA.jsonl"
    print("LLaMa3 VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_hallu-consistency_LLaMa3-12_11_direct.jsonl"
    print("LLaMa3 Direct")
    get_f1(file)


    print("-------------------")
    file = "../consistency/ProofWriter_hallu-consistency_LLaMa3-12_11.jsonl"
    print("LLaMa3 Single")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_hallu-consistency_Deberta-12_12_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_hallu-consistency_Deberta-12_11_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_hallu-consistency_Deberta-12_11.jsonl"
    print("DeBERTA Single")
    get_f1(file)


    print("######################")
    print("######################")
    print("######################")
    print("-------------------")

    file = "../consistency/ProofWriter_neg-consistency_Symbolic-12_13.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_neg-consistency_Symbolic-12_14_VANESSA.jsonl"
    print("VANESSA Symbolic")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_neg-consistency_LLaMa3-12_13_VANESSA.jsonl"
    print("LLaMa3 VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_neg-consistency_LLaMa3-12_13_direct.jsonl"
    print("LLaMa3 Direct")
    get_f1(file)


    print("-------------------")
    file = "../consistency/ProofWriter_neg-consistency_LLaMa3-12_13.jsonl"
    print("LLaMa3 Single")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_neg-consistency_Deberta-12_14_VANESSA.jsonl"
    print("DeBERTA VANESSA")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_neg-consistency_Deberta-12_13_direct.jsonl"
    print("DeBERTA Direct")
    get_f1(file)

    print("-------------------")
    file = "../consistency/ProofWriter_neg-consistency_Deberta-12_13.jsonl"
    print("DeBERTA Single")
    get_f1(file)
    """


get_all_f1s()
