import json
import numpy as np

def load_data(file, ground_truth_file):
    with open(file, 'r') as f:
            data = f.readlines()
    dataset = [json.loads(d) for d in data]
    new_dataset = []

    with open(ground_truth_file, 'r') as f:
        annotations_data = f.readlines()
    annotations_dataset = [json.loads(d) for d in annotations_data]

    for elem, annotations in zip(dataset, annotations_dataset):
        try:
            steps_prediction = elem["predicted_steps"]
            id = elem["id"]
            ground_truth_steps = annotations["ground_truth_steps"]
            ground_truth_reasoning = elem["ground_truth"]
            text_reasoning = elem["reasoning"]
            if steps_prediction == []:
                predicted_reasoning = False
            else:
                predicted_reasoning = all([pred if pred != '-1' else 0 for pred in steps_prediction])
            inst = {"id": id, "ground_truth_steps": ground_truth_steps, "ground_truth_reasoning": ground_truth_reasoning, "predicted_steps": steps_prediction, "predicted_reasoning": predicted_reasoning, "text_reasoning": text_reasoning, "entailments": elem["entailments_dict"], "log_premises": elem["logic_premises"], "log_conclusion": elem["logic_conclusion"], "correspondance": elem["correspondance"]}
            new_dataset.append(inst)
        except:
            continue
    print(len(new_dataset))
    return new_dataset

def check_steps(dataset):
    steps_results = []
    counts = {True: 0, False: 0, "Error": 0, "Contradiction": 0}
    ground_truth_counts = {True: 0, False: 0}
    premises_lengths = []
    for i, instance in enumerate(dataset):
        steps_result = []
        for j, p in enumerate(instance["predicted_steps"]):
            premises_lengths.append(len(instance["log_premises"][j]))
            truth = instance["ground_truth_steps"][j]
            if p == True:
                counts[True] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("TP")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("FP")
                    dico = {"id": instance["id"], "step_id": j, "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
                    with open("EA/FP_LLaMa.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")

            elif p == False:
                counts[False] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("FN")
                    dico = {"id": instance["id"], "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
                    with open("FN.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
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
                    with open("FN.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("TN")
                #steps_result.append("Error")
        steps_results.append(steps_result)
    return steps_results, counts, ground_truth_counts, premises_lengths

def check_steps_vote(LLaMa_dataset, T5_dataset, Mistral_dataset, LLaMa3_dataset):
    steps_results = []
    counts = {True: 0, False: 0, "Error": 0}
    ground_truth_counts = {True: 0, False: 0}
    premises_lengths = []
    for i, instance in enumerate(LLaMa_dataset):
        steps_result = []

        for j, p in enumerate(instance["predicted_steps"]):
            premises_lengths.append(len(instance["log_premises"][j]))
            truth = instance["ground_truth_steps"][j]
            p = vote(i, j, 1, LLaMa_dataset, T5_dataset, Mistral_dataset, LLaMa3_dataset)
            if p == True:
                counts[True] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("TP")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("FP")
                    dico = {"id": instance["id"], "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
                    with open("FP.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
            elif p == False:
                counts[False] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("FN")
                    dico = {"id": instance["id"], "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
                    with open("FN.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("TN")
            else:
                counts["Error"] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("FN")
                    dico = {"id": instance["id"], "reasoning":instance["text_reasoning"], "predicted_steps":instance["predicted_steps"], "ground_truth_steps":instance["ground_truth_steps"], "entailments":instance["entailments"][j], "log_premises":instance["log_premises"][j], "log_conclusion":instance["log_conclusion"][j], "correspondance":instance["correspondance"][j]}
                    with open("FN.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("TN")
                #steps_result.append("Error")
        steps_results.append(steps_result)
    return steps_results, counts, ground_truth_counts, premises_lengths



def check_steps_merged(dataset1, dataset2):
    steps_results = []
    counts = {True: 0, False: 0, "Error": 0}
    ground_truth_counts = {True: 0, False: 0}
    for i, (instance1, instance2) in enumerate(zip(dataset1, dataset2)):
        steps_result = []

        for j, p in enumerate(instance1["predicted_steps"]):
            truth = instance1["ground_truth_steps"][j]
            if p == True and instance2["predicted_steps"][j] == True:
                counts[True] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("TP")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("FP")
                    dico = {"id": instance1["id"], "reasoning":instance1["text_reasoning"], "predicted_steps":instance1["predicted_steps"], "ground_truth_steps":instance1["ground_truth_steps"], "entailments":instance1["entailments"][j], "log_premises":instance1["log_premises"][j], "log_conclusion":instance1["log_conclusion"][j], "correspondance":instance1["correspondance"][j]}
                    with open("FP.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
            elif p == False or instance2["predicted_steps"][j] == False:
                counts[False] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("FN")
                    dico = {"id": instance1["id"], "reasoning":instance1["text_reasoning"], "predicted_steps":instance1["predicted_steps"], "ground_truth_steps":instance1["ground_truth_steps"], "entailments":instance1["entailments"][j], "log_premises":instance1["log_premises"][j], "log_conclusion":instance1["log_conclusion"][j], "correspondance":instance1["correspondance"][j]}
                    with open("FN.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("TN")
            else:
                counts["Error"] += 1
                if truth in {"y","i"}:
                    ground_truth_counts[True] += 1
                    steps_result.append("FN")
                    dico = {"id": instance1["id"], "reasoning":instance1["text_reasoning"], "predicted_steps":instance1["predicted_steps"], "ground_truth_steps":instance1["ground_truth_steps"], "entailments":instance1["entailments"][j], "log_premises":instance1["log_premises"][j], "log_conclusion":instance1["log_conclusion"][j], "correspondance":instance1["correspondance"][j]}
                    with open("FN.jsonl", "a") as f:
                        pass
                        #f.write(json.dumps(dico, ensure_ascii=False)+"\n")
                else:
                    ground_truth_counts[False] += 1
                    steps_result.append("TN")
                #steps_result.append("Error")
        steps_results.append(steps_result)
    return steps_results, counts, ground_truth_counts


def check_reasonings(dataset):
    steps_based_reasonings = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    text_based_reasonings = {"True": 0, "False": 0}
    perfects_reasonings = []
    perfects_count = 0
    for i, instance in enumerate(dataset):
        if instance["predicted_reasoning"] == True and instance["ground_truth_reasoning"] == "True":
            steps_based_reasonings["TP"] += 1
        elif instance["predicted_reasoning"] == True and instance["ground_truth_reasoning"] in {"False", "Uncertain"}:
            steps_based_reasonings["FP"] += 1
        elif instance["predicted_reasoning"] == False and instance["ground_truth_reasoning"] == "True":
            steps_based_reasonings["FN"] += 1
        else:
            steps_based_reasonings["TN"] += 1
        #print(instance["text_reasoning"])
        #input()
        if instance["predicted_reasoning"]:
            perfects_count += 1
        if ("A. Yes" in instance["text_reasoning"] or "(A) Yes" in instance["text_reasoning"]) and instance["ground_truth_reasoning"] == "True":
            text_based_reasonings["True"] += 1
            if instance["predicted_reasoning"]:
                perfects_reasonings.append(i)
        elif "B. No" in instance["text_reasoning"] and instance["ground_truth_reasoning"] == "False":
            text_based_reasonings["True"] += 1
            if instance["predicted_reasoning"]:
                perfects_reasonings.append(i)
        elif "C. Uncertain" in instance["text_reasoning"] and instance["ground_truth_reasoning"] == "Uncertain":
            text_based_reasonings["True"] += 1
            if instance["predicted_reasoning"]:
                perfects_reasonings.append(i)
        else:
            text_based_reasonings["False"] += 1
            #print("oh", i)

        if not ("A. Yes" in instance["text_reasoning"] or "(A) Yes" in instance["text_reasoning"] or "B. No" in instance["text_reasoning"] or "C. Uncertain" in instance["text_reasoning"]):
            print("probleme reasoning", i)
    print("Perfects", perfects_count, len(perfects_reasonings), perfects_reasonings)
    return steps_based_reasonings, text_based_reasonings, len(perfects_reasonings)/perfects_count

def get_f1_scores(steps_results, reasonings):
    #print(reasonings)
    reasoning_precision, reasoning_recall, reasoning_accuracy = reasonings["TP"]/(reasonings["TP"]+reasonings["FP"]), reasonings["TP"]/(reasonings["TP"]+reasonings["FN"]), (reasonings["TP"]+reasonings["TN"])/(reasonings["TP"]+reasonings["FP"]+reasonings["FN"]+reasonings["TN"])
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
        f1 = wilson_conf_interval(2*precision[0]*recall[0]/(precision[0]+recall[0]), 2*steps["TP"]+steps["FN"]+steps["FP"])
        f05 = wilson_conf_interval((1+0.5**2)*precision[0]*recall[0]/((0.5**2)*precision[0]+recall[0]), (1+0.5**2)*steps["TP"]+0.5**2*steps["FN"]+steps["FP"])
        somers_d = (steps["TP"]*steps["TN"]-steps["FP"]*steps["FN"])/((steps["TP"]+steps["FP"])*(steps["TN"]+steps["FN"])*(steps["TN"]+steps["FP"])*(steps["TP"]+steps["FN"]))**0.5
    except ZeroDivisionError:
        precision, recall, accuracy = 0, steps["TP"]/(steps["TP"]+steps["FN"]), (steps["TP"]+steps["TN"])/(steps["TP"]+steps["FP"]+steps["FN"]+steps["TN"])
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
    dataset = load_data(file, "LLaMa2_FOLIO_annotations_full.jsonl")
    steps_results, counts, ground_truth_counts, premises = check_steps(dataset)
    steps_reasonings, text_based_reasonings, correct_perfects = check_reasonings(dataset)
    print("reasonings", text_based_reasonings, round(text_based_reasonings["True"]/(text_based_reasonings["True"]+text_based_reasonings["False"]), 2))
    print("perfects", round(correct_perfects, 2), text_based_reasonings["True"]/(text_based_reasonings["True"]+text_based_reasonings["False"]) < correct_perfects)    
    print("counts", counts)
    print("ground_truth_counts", ground_truth_counts)
    print("Premises", sum(premises)/len(premises))
    print("4+ Premises", sum([1 for p in premises if p >= 4])/len(premises))
    precision, recall, accuracy, f05, f1, reasoning_precision, reasoning_recall, reasoning_accuracy, somers_d = get_f1_scores(steps_results, steps_reasonings)
    print("Category", "Accuracy", "Precision", "Recall", "F1")
    #print("Reasoning:", reasoning_accuracy, reasoning_precision, reasoning_recall, 2*reasoning_precision*reasoning_recall/(reasoning_precision+reasoning_recall))
    print("Steps:", accuracy, precision, recall, f05)  
    entailments_nb = get_entailments_nb(dataset)
    print("Entailments:", entailments_nb)
    print("String:", f"{file} & {int(round(100*counts['Error']/(counts[True] + counts[False] + counts['Error']), 0))}\% & {round(precision[0], 2)}$_{{\pm \\text{{{round(precision[1], 2)}}}}}$ & {round(recall[0], 2)}$_{{\pm \\text{{{round(recall[1], 2)}}}}}$ & {round(f05[0], 2)}$_{{\pm \\text{{{round(f05[1], 2)}}}}}$ & {round(somers_d, 2)} & {round(f1[0], 2)}$_{{\pm \\text{{{round(f1[1], 2)}}}}}$\\\\")
    return

def get_all_f1s():
    print("-----")
    file = "../reasoning/LLaMa2_FOLIO-Symbolic-05_25.jsonl"
    print("Symbolic")
    get_f1(file)

    print("-----")
    file = "../reasoning/LLaMa2_FOLIO-LLaMa3-05_23.jsonl"
    print("LLaMa3")
    get_f1(file)

    print("-----")
    file = "../reasoning/LLaMa2_FOLIO-Deberta-05_22.jsonl"
    print("Deberta")
    get_f1(file)

    print("-----")
    file = "../reasoning/LLaMa2_FOLIO-LLaMa3-direct-10_11.jsonl"
    print("LLaMa3-direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/LLaMa2_FOLIO-Deberta-direct-11_18.jsonl"
    print("Deberta-direct")
    get_f1(file)

    print("-----")
    file = "../reasoning/LLaMa2_FOLIO-GPT-direct-12_14.jsonl"
    print("GPT direct")
    get_f1(file)


get_all_f1s()