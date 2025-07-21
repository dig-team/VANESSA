import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Levenshtein import ratio

class MistralEntailmentModel():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        )
        self.model.eval()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.sos, self.eos = "<s>", "</s>"
        self.system_instructions = """You are an expert linguistic annotator who performs textual entailment: You will be presented with a premise and a hypothesis, and you shall answer whether the premise is equivalent to or entails the hypothesis (Entailment), contradicts it (Contradiction) or does not give enough information to conclude (Neutral). Answer "Entailment" or "Contradiction" only when you're absolutely confident, and "Neutral" the rest of the time (or when making assumptions). Answer only with "Entailment", "Contradiction" or "Neutral", nothing else."""
        self.example_1 = "Premise: Ducks quack . Hypothesis: New York is a city ."
        self.answer_1 = "Neutral"
        self.example_2 = "Premise: X love the Beatles . Hypothesis: X likes the Beatles ."
        self.answer_2 = "Entailment"
        self.example_3 = "Premise: It is not true that X loves the Beatles . Hypothesis: X likes the Beatles ."
        self.answer_3 = "Neutral"
        self.example_4 = "Premise: Alex loves fruits . Hypothesis: John hates pears ."
        self.answer_4 = "Neutral"
        self.example_5 = "Premise: Alice changes often . Hypothesis: Alice change often ."
        self.answer_5 = "Entailment"
        self.example_6 = "Premise: It is not true that Jack is happy . Hypothesis: Jack is unhappy ."
        self.answer_6 = "Entailment"
        self.example_7 = "Premise: X hates cats . Hypothesis: X is a cat person ."
        self.answer_7 = "Contradiction"
        self.example_8 = "Premise: David has blonde hair . Hypothesis: David likes football ."
        self.answer_8 = "Neutral"
        self.example_9 = "Premise: X are a child . Hypothesis: X is a kid ."
        self.answer_9 = "Entailment"
        self.example_10 = "Premise: It is not true that X lives in France . Hypothesis: X lives in France ."
        self.answer_10 = "Contradiction"

    def entail(self, pairs, correspondance_dict, batch_size, contrad_premise=False, entailment_cache={}):
        #print("normal loop")
        text_pairs = [(correspondance_dict[pair[0]], correspondance_dict[pair[1]]) for pair in pairs]
        result = []
        model_text_pairs_indexes = []
        model_text_pairs = []

        messages = [
            {"role": "user", "content": self.system_instructions + "\n" + self.example_1},
            {"role": "assistant", "content": self.answer_1},
            {"role": "user", "content": self.example_2},
            {"role": "assistant", "content": self.answer_2},
            {"role": "user", "content": self.example_3},
            {"role": "assistant", "content": self.answer_3},
            {"role": "user", "content": self.example_4},
            {"role": "assistant", "content": self.answer_4},
            {"role": "user", "content": self.example_5},
            {"role": "assistant", "content": self.answer_5},
            {"role": "user", "content": self.example_6},
            {"role": "assistant", "content": self.answer_6},
            {"role": "user", "content": self.example_7},
            {"role": "assistant", "content": self.answer_7},
            {"role": "user", "content": self.example_8},
            {"role": "assistant", "content": self.answer_8},
            {"role": "user", "content": self.example_9},
            {"role": "assistant", "content": self.answer_9},
            {"role": "user", "content": self.example_10},
            {"role": "assistant", "content": self.answer_10}
        ]

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
                model_text_pairs.append(self.tokenizer.apply_chat_template(messages + [{"role": "user", "content": "Premise: " + contradicted_premise + " Hypothesis: " + pair[1]}], tokenize=False))
        #print("model_text_pairs", model_text_pairs[0])
        #print("model_text_pairs", model_text_pairs[1])
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
                outputs = self.model.generate(**model_input, max_new_tokens=10, do_sample=False)
                generated = self.tokenizer.batch_decode(outputs)

            for j, gen in enumerate(generated):
                gen = gen.split("[/INST]")[-1].replace("</s>", "").strip()
                index = model_text_pairs_indexes_batch[j]
                contradicted_premise = self.contradict_premise(text_pairs[index][0], contrad_premise)

                if "assuming" in gen:
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "N"
                elif "Entailment" in gen:
                    result.append((pairs[index], "E"))
                    entailment_cache[(contradicted_premise + "/SEP/" + text_pairs[index][1])] = "E"
                elif "Contradiction" in gen:
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
