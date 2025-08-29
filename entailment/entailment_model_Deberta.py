import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Levenshtein import ratio

class DeBERTaEntailmentModel():
    def __init__(self, model_name):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        #self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print(self.device)

    def entail(self, pairs, correspondance_dict, batch_size, contrad_premise=False, entailment_cache={}):
        text_pairs = [(correspondance_dict[pair[0]], correspondance_dict[pair[1]]) for pair in pairs]
        result = []
        model_text_pairs_indexes = []
        model_text_pairs = []

        for i, pair in enumerate(text_pairs):
            contradicted_premise = self.contradict_premise(pair[0], contrad_premise)
            if (contradicted_premise + "/SEP/" + pair[1]) in entailment_cache:
                if entailment_cache[contradicted_premise + "/SEP/" + pair[1]] in {"E", "C"}:
                    result.append((pairs[i], entailment_cache[contradicted_premise + "/SEP/" + pair[1]]))
                continue
            
            res_symbo = self.entail_symbolic(contradicted_premise, pair[1])
            if res_symbo in {"E", "C"}:
                result.append((pairs[i], res_symbo))
                entailment_cache[(contradicted_premise + "/SEP/" + pair[1])] = res_symbo
            else:
                model_text_pairs_indexes.append(i)
                model_text_pairs.append((contradicted_premise, pair[1]))
        
        if len(model_text_pairs) == 0:
            return result, entailment_cache

        iterations = (len(model_text_pairs) - 1) // batch_size +1
        print("iterations", iterations)

        for it in range(iterations):
            if (it+1)*batch_size >= len(model_text_pairs):
                model_text_pairs_batch = model_text_pairs[it*batch_size:]
                model_text_pairs_indexes_batch = model_text_pairs_indexes[it*batch_size:]
            else:
                model_text_pairs_batch = model_text_pairs[it*batch_size:(it+1)*batch_size]
                model_text_pairs_indexes_batch = model_text_pairs_indexes[it*batch_size:(it+1)*batch_size]

            batch_premises = [pair[0] for pair in model_text_pairs_batch]
            batch_conclusions = [pair[1] for pair in model_text_pairs_batch]
            generated = []
            for prem, conc in zip(batch_premises, batch_conclusions):
                model_input = self.tokenizer(prem, conc, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model(model_input["input_ids"])
                    prediction = torch.softmax(output["logits"][0], -1).tolist()
                    if max(prediction) == prediction[0]:
                        generated.append("E")
                    elif max(prediction) == prediction[2]:
                        generated.append("C")
                    else:
                        generated.append("N")
            for j, gen in enumerate(generated):
                index = model_text_pairs_indexes_batch[j]
                contradicted_premise = self.contradict_premise(text_pairs[index][0], contrad_premise)
                if gen == "E":
                    result.append((pairs[index], "E"))
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "E"
                elif gen == "C":
                    result.append((pairs[index], "C"))
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "C"
                else:
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "N"

        return result, entailment_cache

    def contradict_premise(self, premise, contrad_premise):
        if not contrad_premise:
            return premise
        if "not" in premise:
            return premise.replace("not", "")
        if "does n't" in premise or "doesn't" in premise:
            return premise.replace("doesn't", "").replace("does n't", "")
        if "n't" in premise:
            return premise.replace("n't", "")
        return "It is not true that " + premise

    def entail_symbolic(self, premise, hypothesis):
            premise = premise.lower()
            hypothesis = hypothesis.lower()
            if premise.count(" not ") == hypothesis.count(" not ") and premise.count("n't") == hypothesis.count("n't"):
                if premise == hypothesis:
                    return "E" 
                elif premise.replace("a ", "").replace("thing ", "").replace("person ", "") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", ""):
                    return "E"
                elif premise.replace(" are ", " is ") == hypothesis.replace(" are ", " is "):
                    return "E"
                elif premise.replace("ves ", "f ").replace("es ", " ").replace("s ", "") == hypothesis.replace("ves ", "f ").replace("es ", " ").replace("s ", ""):
                    return "E"
                elif premise.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is ") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is "):
                    return "E"
                elif premise.replace("a ", "").replace("thing ", "").replace("person ", "").replace("ves ", "f ").replace("es ", " ").replace("s ", "") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", "").replace("ves ", "f ").replace("es ", " ").replace("s ", ""):
                    return "E"
                elif premise.replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", "") == hypothesis.replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", ""):
                    return "E"
                elif premise.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", "") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", ""):
                    return "E"
                return "N"
            else:
                if self.entail_symbolic(premise.replace(" not ", " ").replace("n't",""), hypothesis.replace(" not ", " ").replace("n't","")) == "E":
                    return "C"
                return "N"