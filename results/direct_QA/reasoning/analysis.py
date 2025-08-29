import json
import numpy as np
from collections import defaultdict

def load_data(file):
    with open(file, 'r') as f:
        data = f.readlines()
    dataset = [json.loads(d) for d in data]
    new_dataset = []
    invert=False
    invert_false_uncertains_files = {"CD", "DD", "BD"}
    if any([x in file for x in invert_false_uncertains_files]):
        invert = True
    for elem in dataset:
        try:
            if "ground truth" in elem:
                if elem["ground truth"] == "Yes":
                    instance_truth = True
                elif elem["ground truth"] == "No":
                    instance_truth = False
                else:
                    instance_truth = "Uncertain"
            else:
                if elem["ground_truth"] in {"True", "true", True}:
                    instance_truth = True
                elif elem["ground_truth"] in {"False", "false", False}:
                    instance_truth = False
                else:
                    instance_truth = "Uncertain"
            id = elem["id"]
            if invert and instance_truth in {False, "Uncertain"}:
                instance_truth = "Uncertain"
                if int(id)%4 in(1,3):
                    continue
            instance_prediction = elem["predicted_instance"]
            inst = {"id": id, "ground_truth": instance_truth, "predicted_instance": instance_prediction, "errors": elem["errors"]}
            new_dataset.append(inst)
        except:
            continue

    return new_dataset

def check_reasonings(dataset):
    #reasoning_results = {"TP": 0, "FP": 0, "FU": 0, "TU":0, "FN": 0, "TN": 0, "Error": 0}
    reasoning_results = defaultdict(int)
    ground_truth_counts = {True: 0, False: 0, "Uncertain": 0}
    for instance in dataset:
        if instance["errors"] != [] and "Reasoning Error: maximum recursion depth exceeded" not in instance["errors"][0]:
            ground_truth_counts[instance["ground_truth"]] += 1
            #reasoning_results["Error"] += 1
            if instance["ground_truth"] == True:
                reasoning_results["PE"] += 1
            elif instance["ground_truth"] == False:
                reasoning_results["NE"] += 1
            else:
                reasoning_results["UE"] += 1
            continue
        if instance["ground_truth"] == True:
            ground_truth_counts[True] += 1
            if instance["predicted_instance"] == True:
                reasoning_results["TP"] += 1
            elif instance["predicted_instance"] == False:
                reasoning_results["PFN"] += 1
            else:
                reasoning_results["PFU"] += 1
        elif instance["ground_truth"] == False:
            ground_truth_counts[False] += 1
            if instance["predicted_instance"] == True:
                reasoning_results["NFP"] += 1
            elif instance["predicted_instance"] == False:
                reasoning_results["TN"] += 1
            else:
                reasoning_results["NFU"] += 1
        else:
            ground_truth_counts["Uncertain"] += 1
            if instance["predicted_instance"] == True:
                reasoning_results["UFP"] += 1
            elif instance["predicted_instance"] == False:
                reasoning_results["UFN"] += 1
            else:
                reasoning_results["TU"] += 1
    return reasoning_results, ground_truth_counts
    
def get_f1_scores(reasoning_results):
    try:
        """accuracy = wilson_conf_interval((reasoning_results["TP"]+reasoning_results["TN"]+reasoning_results["TU"])/(reasoning_results["TP"]+reasoning_results["FP"]+reasoning_results["FN"]+reasoning_results["TN"]+reasoning_results["TU"]+reasoning_results["FU"]+reasoning_results["Error"]), reasoning_results["TP"]+reasoning_results["FP"]+reasoning_results["FN"]+reasoning_results["TN"]+reasoning_results["TN"]+reasoning_results["TU"]+reasoning_results["FU"]+reasoning_results["Error"])
        precision = wilson_conf_interval(reasoning_results["TP"]/(reasoning_results["TP"]+reasoning_results["FP"]), reasoning_results["TP"]+reasoning_results["FP"])
        recall = wilson_conf_interval(reasoning_results["TP"]/(reasoning_results["TP"]+reasoning_results["FN"]), reasoning_results["TP"]+reasoning_results["FN"])
        f05 = wilson_conf_interval((1+0.5**2)*precision[0]*recall[0]/((0.5**2)*precision[0]+recall[0]), (1+0.5**2)*reasoning_results["TP"]+0.5**2*reasoning_results["FN"]+reasoning_results["FP"])
        f1 = wilson_conf_interval(2*precision[0]*recall[0]/(precision[0]+recall[0]), 2*reasoning_results["TP"]+reasoning_results["FN"]+reasoning_results["FP"])
        somers_d = (2*reasoning_results["TP"]*reasoning_results["TN"]-2*reasoning_results["FP"]*reasoning_results["FN"])/((2*reasoning_results["TP"]+2*reasoning_results["FP"])*(reasoning_results["TN"]+reasoning_results["FN"])*(reasoning_results["TN"]+2*reasoning_results["FP"])*(2*reasoning_results["TP"]+reasoning_results["FN"]))**0.5"""
        print("SUM", sum(reasoning_results.values()))
        accuracy = wilson_conf_interval((reasoning_results["TP"]+reasoning_results["TN"]+reasoning_results["TU"])/sum(reasoning_results.values()), sum(reasoning_results.values()))

        if reasoning_results["TP"] == 0:
            precision_positives, recall_positives, f1_positives = 0, 0, 0
        else:
            precision_positives = round(reasoning_results["TP"]/(reasoning_results["TP"]+reasoning_results["NFP"]+reasoning_results["UFP"]), 4)
            recall_positives = round(reasoning_results["TP"]/(reasoning_results["TP"]+reasoning_results["PFN"]+reasoning_results["PFU"]+reasoning_results["PE"]), 4)
            f1_positives = round(2*precision_positives*recall_positives/(precision_positives+recall_positives), 4)

        if reasoning_results["TN"] == 0:
            precision_negatives, recall_negatives, f1_negatives = 0, 0, 0
        else:
            precision_negatives = round(reasoning_results["TN"]/(reasoning_results["TN"]+reasoning_results["UFN"]+reasoning_results["PFN"]), 4)
            recall_negatives = round(reasoning_results["TN"]/(reasoning_results["TN"]+reasoning_results["NFP"]+reasoning_results["NFU"]+reasoning_results["NE"]), 4)   
            f1_negatives = round(2*precision_negatives*recall_negatives/(precision_negatives+recall_negatives)   , 4)

        if reasoning_results["TU"] == 0:
            precision_uncertain, recall_uncertain, f1_uncertain = 0, 0, 0
        else:
            precision_uncertain = round(reasoning_results["TU"]/(reasoning_results["TU"]+reasoning_results["PFU"]+reasoning_results["NFU"]), 4)
            recall_uncertain = round(reasoning_results["TU"]/(reasoning_results["TU"]+reasoning_results["UFP"]+reasoning_results["UFN"]+reasoning_results["UE"]), 4)
            f1_uncertain = round(2*precision_uncertain*recall_uncertain/(precision_uncertain+recall_uncertain), 4)

        precision_affirmatives = round(100*(reasoning_results["TP"]+reasoning_results["TN"])/(reasoning_results["TP"]+reasoning_results["NFP"]+reasoning_results["UFP"]+reasoning_results["TN"]+reasoning_results["PFN"]+reasoning_results["UFN"]), 2)
        recall_affirmatives = round(100*(reasoning_results["TP"]+reasoning_results["TN"])/(reasoning_results["TP"]+reasoning_results["PFN"]+reasoning_results["PFU"]+reasoning_results["PE"]+reasoning_results["TN"]+reasoning_results["NFP"]+reasoning_results["NFU"]+reasoning_results["NE"]), 2)
        f1_affirmatives = round(2*precision_affirmatives*recall_affirmatives/(precision_affirmatives+recall_affirmatives), 2)

        print("Positives", precision_positives, recall_positives, f1_positives)
        print("Negatives", precision_negatives, recall_negatives, f1_negatives)
        print("Uncertain", precision_uncertain, recall_uncertain, f1_uncertain)
        print("Affirmatives", precision_affirmatives, "&", recall_affirmatives, "&", f1_affirmatives)

        print("Macro-Averages", round((precision_positives+precision_negatives+precision_uncertain)/3, 4), round((recall_positives+recall_negatives+recall_uncertain)/3, 4), round((f1_positives+f1_negatives+f1_uncertain)/3, 4))
        return (0,0), (0,0), accuracy, (0,0), (0,0), 0
    except ZeroDivisionError:
        print("ZeroDivisionError")
        if 'accuracy' not in locals():
            accuracy = (0,0)
        return (0,0), (0,0), accuracy, (0,0), (0,0), 0
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

def get_f1(file):
    dataset = load_data(file)
    reasoning_results, ground_truth = check_reasonings(dataset)
    print("Ground truth counts", ground_truth)
    print("Predictions", {True: reasoning_results["TP"]+reasoning_results["NFP"]+reasoning_results["UFP"],False: reasoning_results["TN"]+reasoning_results["PFN"]+reasoning_results["UFN"], "Uncertain": reasoning_results["TU"]+reasoning_results["PFU"]+reasoning_results["NFU"], "Errors": reasoning_results["NE"]+reasoning_results["PE"]+reasoning_results["UE"]})
    print("Reasoning results", reasoning_results)
    precision, recall, accuracy, f05, f1, somers_d = get_f1_scores(reasoning_results)
    """print("Category", "Accuracy", "Precision", "Recall", "F05", "F1")
    try:
        print("Steps:", accuracy, precision, recall, f05, f1) 
    except ZeroDivisionError:
        print("Steps:", accuracy, precision, recall, 0, 0, 0)  """
    print("Accuracy", accuracy)
    print("Somer's D", somers_d)
    #print("String:", f"{file} & {int(round(100*counts['Error']/(counts[True] + counts[False] + counts['Error']), 0))}\% & {round(precision[0], 2)}$_{{\pm \\text{{{round(precision[1], 2)}}}}}$ & {round(recall[0], 2)}$_{{\pm \\text{{{round(recall[1], 2)}}}}}$ & {round(f05[0], 2)}$_{{\pm \\text{{{round(f05[1], 2)}}}}}$ & {round(somers_d, 2)} & {round(f1[0], 2)}$_{{\pm \\text{{{round(f1[1], 2)}}}}}$\\\\")
    return

def get_all_f1s():


    print("-----")
    file = "LLaMa3_FOLIO-VANESSA-Symbolic-03_27.jsonl"
    print("FOLIO - VANESSA Symbolic")
    get_f1(file)

    print("-----")
    file = "LLaMa3_FOLIO-VANESSA-LLaMa3-03_26.jsonl"
    print("FOLIO - VANESSA LLaMa3")
    get_f1(file)



    """print("-----")
    file = "LLaMa3_FOLIO-VANESSA-Gemma1.1-03_19.jsonl"
    print("FOLIO - VANESSA Gemma1.1")
    get_f1(file)

    print("-----")
    file = "LLaMa3_FOLIO-VANESSA-Gemma2-03_22.jsonl"
    print("FOLIO - VANESSA Gemma2")
    get_f1(file)

    file = "LLaMa3_FOLIO-VANESSA-T5-03_20.jsonl"
    print("FOLIO - VANESSA T5")
    get_f1(file)"""

    print("-----")
    file = "LLaMa3_FOLIO-VANESSA-Mistral-03_26.jsonl"
    print("FOLIO - VANESSA Ministral")
    get_f1(file)

    print("-----")
    file = "QA_ProntoQA-VANESSA-Symbolic-03_27.jsonl"
    print("ProntoQA FULL NEW - VANESSA (full new) Symbolic")
    get_f1(file)

    print("-----")
    file = "QA_ProntoQA-VANESSA-LLaMa3-03_26.jsonl"
    print("ProntoQA - VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "QA_ProntoQA-VANESSA-Mistral-03_27.jsonl"
    print("ProntoQA - VANESSA Mistral")
    get_f1(file)

    print("-----")
    file = "ProofWriter_QA-VANESSA-LLaMa3-03_27.jsonl"
    print("ProofWriter - VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "ProofWriter_QA-VANESSA-Mistral-03_27.jsonl"
    print("ProofWriter - VANESSA Mistral")
    get_f1(file)
    
    print("-----")
    file = "ProofWriter_QA-VANESSA-Symbolic-03_28.jsonl"
    print("ProofWriter QA - VANESSA newgen 2 Symbolic")
    get_f1(file)

    """print("-----")
    file = "PMT-VANESSA-Symbolic-03_25.jsonl"
    print("PMT (vanilla)- VANESSA Symbolic")
    get_f1(file)

    print("-----")
    file = "PMT-VANESSA-LLaMa3-03_14.jsonl"
    print("PMT (vanilla)- VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "PMT-VANESSA-Mistral-03_24.jsonl"
    print("PMT (vanilla)- VANESSA Mistral")
    get_f1(file)
    """
    
    print("-----")
    file = "PHS-VANESSA-Symbolic-03_27.jsonl"
    print("PHS (vanilla)- VANESSA Symbolic")
    get_f1(file)

    print("-----")
    file = "PHS-VANESSA-LLaMa3-03_27.jsonl"
    print("PHS (vanilla)- VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "PHS-VANESSA-Mistral-03_27.jsonl"
    print("PHS (vanilla)- VANESSA Mistral")
    get_f1(file)


    print("-----")
    file = "DS-VANESSA-Symbolic-03_28.jsonl"
    print("PDS (vanilla)- VANESSA Symbolic")
    get_f1(file)

    print("-----")
    file = "DS-VANESSA-LLaMa3-03_27.jsonl"
    print("PDS (vanilla)- VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "DS-VANESSA-Mistral-03_28.jsonl"
    print("PDS (vanilla)- VANESSA Mistral")
    get_f1(file)

    print("-----")
    file = "CD-VANESSA-Symbolic-03_27.jsonl"
    print("PCD (vanilla)- VANESSA Symbolic")
    get_f1(file)

    print("-----")
    file = "CD-VANESSA-LLaMa3-03_27.jsonl"
    print("PCD (vanilla)- VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "CD-VANESSA-Mistral-03_27.jsonl"
    print("PCD (vanilla)- VANESSA Mistral")
    get_f1(file)


    print("-----")
    file = "DD-VANESSA-Symbolic-03_27.jsonl"
    print("PDD (vanilla)- VANESSA Symbolic")
    get_f1(file)

    print("-----")
    file = "DD-VANESSA-LLaMa3-03_27.jsonl"
    print("PDD (vanilla)- VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "DD-VANESSA-Mistral-03_28.jsonl"
    print("PDD (vanilla)- VANESSA Mistral")
    get_f1(file)

    print("-----")
    file = "PBD-VANESSA-Symbolic-03_27.jsonl"
    print("PBD (vanilla)- VANESSA Symbolic")
    get_f1(file)

    print("-----")
    file = "PBD-VANESSA-LLaMa3-03_27.jsonl"
    print("PBD (vanilla)- VANESSA LLaMa3")
    get_f1(file)

    print("-----")
    file = "PBD-VANESSA-Mistral-03_28.jsonl"
    print("PBD (vanilla)- VANESSA Mistral")
    get_f1(file)

get_all_f1s()

