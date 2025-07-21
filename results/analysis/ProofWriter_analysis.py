import json
import numpy as np

def load_data(file):
    with open(file, 'r') as f:
        data = f.readlines()
    dataset = [json.loads(d) for d in data]
    new_dataset = []

    for elem in dataset:
        steps_prediction = elem["predicted_steps"]
        id = elem["id"]
        if "ground_truth" in elem:
            ground_truth_steps = elem["ground_truth"]
            ground_truth_reasoning = all(ground_truth_steps)
        else:
            ground_truth_steps = [True for _ in steps_prediction]
            ground_truth_reasoning = True
        predicted_reasoning = all(steps_prediction)
        inst = {"id": id, "ground_truth_steps": ground_truth_steps, "ground_truth_reasoning": ground_truth_reasoning, "predicted_steps": steps_prediction, "predicted_reasoning": predicted_reasoning, "entailments": elem["entailments_dict"], "premises": elem["logic_premises"], "text_reasoning": elem["reasoning"], "question": elem["question"]}
        new_dataset.append(inst)
    return new_dataset

def check_steps(dataset):
    steps_results = []
    counts = {True: 0, False: 0, "Error": 0, "Contradiction": 0}
    ground_truths = {True: 0, False: 0}
    premises_lengths = []
    for i, instance in enumerate(dataset):
        steps_result = []
        for j, p in enumerate(instance["predicted_steps"]):
            premises_lengths.append(len(instance["premises"][j]))
            truth = instance["ground_truth_steps"][j]
            ground_truths[truth] += 1
            if p == True:
                counts[True] += 1
                if truth:
                    steps_result.append("TP")
                else:
                    steps_result.append("FP")
            elif p == False:
                counts[False] += 1
                if truth:
                    steps_result.append("FN")
                else:
                    steps_result.append("TN")
            else:
                counts["Error"] += 1
                if "Contradiction" in p:
                    counts["Contradiction"] += 1
                if truth:
                    steps_result.append("FN")
                else:
                    steps_result.append("TN")
                #steps_result.append("Error")
        steps_results.append(steps_result)
    return steps_results, counts, ground_truths, premises_lengths

def check_reasonings(dataset):
    reasoning_results = {True: 0, False: 0}
    for instance in dataset:
        last_conclusion = instance["text_reasoning"].split(": ")[-1].replace(".\n\n", "")
        if last_conclusion in instance["question"]:
            reasoning_results[True] += 1
        else:
            reasoning_results[False] += 1
    return reasoning_results
    

def get_f1_scores(steps_results):
    reasoning_precision, reasoning_recall, reasoning_accuracy = 0,0,0
    steps = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    micro_precisions, micro_recalls, micro_accuracies = [], [], []
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
    print("Steps", steps)
    try:
        precision = wilson_conf_interval(steps["TP"]/(steps["TP"]+steps["FP"]), steps["TP"]+steps["FP"])
        recall = wilson_conf_interval(steps["TP"]/(steps["TP"]+steps["FN"]), steps["TP"]+steps["FN"])
        accuracy = wilson_conf_interval((steps["TP"]+steps["TN"])/(steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"]), steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"])
        f05 = wilson_conf_interval((1+0.5**2)*precision[0]*recall[0]/((0.5**2)*precision[0]+recall[0]), (1+0.5**2)*steps["TP"]+0.5**2*steps["FN"]+steps["FP"])
        f1 = wilson_conf_interval(2*precision[0]*recall[0]/(precision[0]+recall[0]), 2*steps["TP"]+steps["FN"]+steps["FP"])
        somers_d = (steps["TP"]*steps["TN"]-steps["FP"]*steps["FN"])/((steps["TP"]+steps["FP"])*(steps["TN"]+steps["FN"])*(steps["TN"]+steps["FP"])*(steps["TP"]+steps["FN"]))**0.5
    except ZeroDivisionError:
        precision, recall, accuracy = 0, steps["TP"]/(steps["TP"]+steps["FN"]), (steps["TP"]+steps["TN"])/(steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"])
        somers_d = 0
    return precision, recall, accuracy, f05, f1, reasoning_precision, reasoning_recall, reasoning_accuracy, somers_d

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
    print("instances", len(dataset))
    steps_results, counts, ground_truths, premises = check_steps(dataset)
    reasonings = check_reasonings(dataset)
    print("Reasonings", reasonings)
    print("Ground truths", ground_truths)
    print("Premises", sum(premises)/len(premises))
    print("4+ Premises", sum([1 for p in premises if p >= 4])/len(premises))
    print("counts", counts)
    precision, recall, accuracy, f05, f1, reasoning_precision, reasoning_recall, reasoning_accuracy, somers_d = get_f1_scores(steps_results)
    print("Somers D:", somers_d)
    #print("Reasoning:", reasoning_accuracy, reasoning_precision, reasoning_recall, 2*reasoning_precision*reasoning_recall/(reasoning_precision+reasoning_recall))
    print("Category", "Accuracy", "Precision", "Recall", "F1", "F05")
    #print("Reasoning:", reasoning_accuracy, reasoning_precision, reasoning_recall, 2*reasoning_precision*reasoning_recall/(reasoning_precision+reasoning_recall))
    print("Steps:", accuracy, precision, recall, f1, f05)  
    print("String:", f"{file} & {int(round(100*counts['Error']/(counts[True] + counts[False] + counts['Error']), 0))}\% & {round(precision[0], 2)}$_{{\pm \\text{{{round(precision[1], 2)}}}}}$ & {round(recall[0], 2)}$_{{\pm \\text{{{round(recall[1], 2)}}}}}$ & {round(f05[0], 2)}$_{{\pm \\text{{{round(f05[1], 2)}}}}}$ & {round(somers_d, 2)} & {round(f1[0], 2)}$_{{\pm \\text{{{round(f1[1], 2)}}}}}$\\\\")

    return

def get_all_f1s():
    print("#######\n#######\n#######")

    print("-----")
    file = "../reasoning/ProofWriter_remove-Symbolic-05_25.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_remove-LLaMa3-05_23.jsonl"
    print("LLaMa3")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_remove-Deberta-05_23.jsonl"
    print("Deberta")
    get_f1(file)

    print("------")
    print("remove, LLaMa3-direct")
    file = "../reasoning/ProofWriter_remove-LLaMa3-direct-12_11.jsonl"
    get_f1(file)

    print("------")
    print("remove, DeBERTa-direct")
    file = "../reasoning/ProofWriter_remove-Deberta-direct-12_11.jsonl"
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_remove-GPT-direct-12_14.jsonl"
    print("remove - GPT direct")
    get_f1(file)

    print("#######\n#######\n#######")

    print("-----")
    file = "../reasoning/ProofWriter_hallu-Symbolic-05_25.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_hallu-LLaMa3-05_22.jsonl"
    print("LLaMa3")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_hallu-Deberta-05_24.jsonl"
    print("Deberta")
    get_f1(file)

    print("------")
    print("hallu, LLaMa3-direct")
    file = "../reasoning/ProofWriter_hallu-LLaMa3-direct-12_11.jsonl"
    get_f1(file)

    print("------")
    print("hallu, DeBERTa-direct")
    file = "../reasoning/ProofWriter_hallu-Deberta-direct-12_11.jsonl"
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_hallu-GPT-direct-12_14.jsonl"
    print("hallu - GPT direct")
    get_f1(file)


    print("#######\n#######\n#######")

    print("-----")
    file = "../reasoning/ProofWriter_neg-Symbolic-05_25.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_neg-LLaMa3-05_22.jsonl"
    print("LLaMa3")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_neg-Deberta-05_23.jsonl"
    print("Deberta")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_neg-LLaMa3-direct-12_15.jsonl"
    print("neg - LLaMa3 direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_neg-Deberta-direct-12_15.jsonl"
    print("neg - DeBERTa direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/ProofWriter_neg-GPT-direct-12_14.jsonl"
    print("neg - GPT direct")
    get_f1(file)

get_all_f1s()