import json
import numpy as np

def load_data(file, ReCEval):
    with open(file, 'r') as f:
        data = f.readlines()
    dataset = [json.loads(d) for d in data]
    new_dataset = []

    for elem in dataset:
        try:
            steps_prediction = elem["predicted_steps"]
            id = elem["id"]
            ground_truth_steps = elem["ground_truth_steps"]
            ground_truth_reasoning = elem["ground_truth"]
            text_reasoning = elem["reasoning"]
            if steps_prediction == []:
                predicted_reasoning = False
            else:
                predicted_reasoning = all([pred if pred != '-1' else 0 for pred in steps_prediction])
            inst = {"id": id, "ground_truth_steps": ground_truth_steps, "ground_truth_reasoning": ground_truth_reasoning, "predicted_steps": steps_prediction, "predicted_reasoning": predicted_reasoning, "text_reasoning": text_reasoning, "entailments": elem["entailments_dict"], "log_premises": elem["logic_premises"], "log_conclusion": elem["logic_conclusion"], "correspondance": elem["correspondance"], "question": elem["question"]}
            new_dataset.append(inst)
        except:
            continue
    
    if ReCEval is not None:
        with open(ReCEval, 'r') as f:
            ReCEval_data = f.readlines()
        ReCEval_data = [json.loads(d) for d in ReCEval_data]
        for i, instance in enumerate(new_dataset):
            instance["predicted_steps"] = [pred > 0.5 for pred in ReCEval_data[i]["preds"]]
    print(len(new_dataset))
    return new_dataset

def check_steps(dataset):
    steps_results = []
    counts = {True: 0, False: 0, "Error": 0, "Contradiction": 0}
    ground_truth_counts = {True: 0, False: 0}
    premises_lengths = []
    total_count = 0
    for i, instance in enumerate(dataset):
        steps_result = []
        for j, p in enumerate(instance["predicted_steps"]):
            premises_lengths.append(len(instance["log_premises"][j]))
            try:
                truth = instance["ground_truth_steps"][j]
            except IndexError:
                continue
            if p == True:
                counts[True] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("TP")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("FP")
                    dico = {"id": instance["id"], "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
            elif p == False:
                counts[False] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("FN")
                    dico = {"id": instance["id"], "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("TN")
            else:
                counts["Error"] += 1
                if "Contradiction" in p:
                    counts["Contradiction"] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("FN")
                    dico = {"id": instance["id"], "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("TN")
                #steps_result.append("Error")
            total_count += 1
        steps_results.append(steps_result)
    return steps_results, counts, ground_truth_counts, premises_lengths

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
    print(steps)
    try:
        precision = wilson_conf_interval(steps["TP"]/(steps["TP"]+steps["FP"]), steps["TP"]+steps["FP"])
        recall = wilson_conf_interval(steps["TP"]/(steps["TP"]+steps["FN"]), steps["TP"]+steps["FN"])
        accuracy = wilson_conf_interval((steps["TP"]+steps["TN"])/(steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"]), steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"])
        f05 = wilson_conf_interval((1+0.5**2)*precision[0]*recall[0]/((0.5**2)*precision[0]+recall[0]), (1+0.5**2)*steps["TP"]+0.5**2*steps["FN"]+steps["FP"])
        f1 = wilson_conf_interval(2*precision[0]*recall[0]/(precision[0]+recall[0]), 2*steps["TP"]+steps["FN"]+steps["FP"])
        somers_d = (2*steps["TP"]*steps["TN"]-2*steps["FP"]*steps["FN"])/((2*steps["TP"]+2*steps["FP"])*(steps["TN"]+steps["FN"])*(steps["TN"]+2*steps["FP"])*(2*steps["TP"]+steps["FN"]))**0.5
    except ZeroDivisionError:
        print("ZeroDivisionError")
        return (0,0), (0,0), 0, (0,0), (0,0), 0, 0, 0, (0,0)
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

def get_f1(file, ReCEval=None):
    dataset = load_data(file, ReCEval)
    steps_results, counts, ground_truth_counts, premises = check_steps(dataset)
    reasoning_results = check_reasonings(dataset)
    print("Reasoning results", reasoning_results)
    print("counts", counts)
    print("ground_truth_counts", ground_truth_counts)
    print("Premises", sum(premises)/len(premises))
    print("4+ Premises", sum([1 for p in premises if p >= 4])/len(premises))
    precision, recall, accuracy, f05, f1, somers_d = get_f1_scores(steps_results)
    print("Category", "Accuracy", "Precision", "Recall", "F05", "F1")
    try:
        print("Steps:", accuracy, precision, recall, f05, f1) 
    except ZeroDivisionError:
        print("Steps:", accuracy, precision, recall, 0, 0, 0)   
    entailments_nb = get_entailments_nb(dataset)
    print("Entailments:", entailments_nb)
    print("Somer's D", somers_d)
    print("String:", f"{file} & {int(round(100*counts['Error']/(counts[True] + counts[False] + counts['Error']), 0))}\% & {round(precision[0], 2)}$_{{\pm \\text{{{round(precision[1], 2)}}}}}$ & {round(recall[0], 2)}$_{{\pm \\text{{{round(recall[1], 2)}}}}}$ & {round(f05[0], 2)}$_{{\pm \\text{{{round(f05[1], 2)}}}}}$ & {round(somers_d, 2)} & {round(f1[0], 2)}$_{{\pm \\text{{{round(f1[1], 2)}}}}}$\\\\")
    return

def get_all_f1s():
    file = "../reasoning/EB_hallu_NEW-Symbolic-05_26.jsonl"
    print("Hallu - Symbolic")
    get_f1(file)

    file = "../reasoning/EB_hallu_NEW-LLaMa3-05_26.jsonl"
    print("Hallu - LLaMa3")
    get_f1(file)

    file = "../reasoning/EB_hallu_NEW-Deberta-05_26.jsonl"
    print("Hallu - Deberta")
    get_f1(file)

    print("-----")
    file = "../reasoning/EB_hallu_NEW-Deberta-direct-12_13.jsonl"
    print("Hallu - Deberta direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/EB_hallu_NEW-LLaMa3-direct-12_13.jsonl"
    print("Hallu - LLaMa3 direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/EB_hallu_NEW-GPT-direct-12_14.jsonl"
    print("Hallu - GPT direct")
    get_f1(file)

    print("#####\n#####\n#####")

    file = "../reasoning/EB_neg_NEW-Symbolic-05_26.jsonl"
    print("neg - Symbolic")
    get_f1(file)

    file = "../reasoning/EB_neg_NEW-LLaMa3-05_26.jsonl"
    print("neg - LLaMa3")
    get_f1(file)

    file = "../reasoning/EB_neg_NEW-Deberta-05_26.jsonl"
    print("neg - Deberta")
    get_f1(file)

    print("-----")
    file = "../reasoning/EB_neg_NEW-LLaMa3-direct-12_13.jsonl"
    print("neg - LLaMa3 direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/EB_neg_NEW-Deberta-direct-12_13.jsonl"
    print("neg - Deberta direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/EB_neg_NEW-GPT-direct-12_14.jsonl"
    print("Neg - GPT direct")
    get_f1(file)

get_all_f1s()

