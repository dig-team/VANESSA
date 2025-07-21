import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from Levenshtein import ratio

class T5EntailmentModel():
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", load_in_8bit=True)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)

        self.prefix_instructions ="Does the premise entail the hypothesis? Answer Entailment or Contradiction only when you're absolutely confident that the relation is particularly evident and clear. The rest of the time, answer Neutral. Answer Neutral as much as possible.\nPremise: Alex loves fruits. Hypothesis: John hates pears.\n Answer: Neutral"


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
                model_text_pairs.append(f"{self.prefix_instructions}\nPremise: {contradicted_premise} Hypothesis: {pair[1]}\n Answer:")
        
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
            
            model_input = self.tokenizer(model_text_pairs_batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                generation_tokens = self.model.generate(**model_input, num_beams=10, num_return_sequences=1, no_repeat_ngram_size=1,remove_invalid_values=True, do_sample=False, max_length=10)
                generated = self.tokenizer.batch_decode(generation_tokens,skip_special_tokens=True)
        
            for j, gen in enumerate(generated):
                index = model_text_pairs_indexes_batch[j]
                contradicted_premise = self.contradict_premise(text_pairs[index][0], contrad_premise)
                if gen.replace("<pad>","").strip()[0][0] == "E":
                    result.append((pairs[index], "E"))
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "E"
                elif gen.replace("<pad>","").strip()[0][0] == "C":
                    result.append((pairs[index], "C"))
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "C"
                else:
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "N"

        return result, entailment_cache

    def contradict_premise(self, premise, contrad_premise):
        if not contrad_premise:
            return premise
        if "not " in premise:
            return premise.replace("not ", "")
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
                elif premise.replace("is a thing that is a ", "is a ") == hypothesis.replace("is a thing that is a ", "is a "):
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
                if self.entail_symbolic(remove_does_not(premise).replace(" not ", " ").replace("n't",""), remove_does_not(hypothesis).replace(" not ", " ").replace("n't","")) == "E":
                    return "C"
                return "N"

def remove_does_not(sentence):
    words = sentence.split(" ")
    if "does" in words and words[words.index("does")+1] == "not":
        index = words.index("does")+1
        words[index+1] = words[index+1]+'s'
    if "doesn't" in words:
        index = words.index("doesn't")
        words[index+1] = words[index+1]+'s'
    return " ".join(words).replace(" does not ", " ").replace(" doesn't ", " ").replace(" do not ", " ")    
