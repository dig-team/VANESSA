import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from Levenshtein import ratio

class LLaMaEntailmentModel():
    def __init__(self, model_name):
        with open("hf_key", "r") as f:
            access_token = f.read().strip()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=access_token, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=access_token, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto", offload_folder="temp/") 
        self.model.eval()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.DEFAULT_SYSTEM_PROMPT = """\
        You are an expert linguistic annotator who performs textual entailment: You will be presented with a premise and a hypothesis, and you shall answer whether the premise is equivalent to or entails the hypothesis (Entailment), contradicts it (Contradiction) or does not give enough information to conclude (Neutral). Answer "Entailment" or "Contradiction" only when you're absolutely confident, and "Neutral" the rest of the time. Answer only with "Entailment", "Contradiction" or "Neutral", nothing else.
Here are some examples:
Premise: Ducks quack . Hypothesis: New York is a city .
Answer: Neutral .
Premise: X love the Beatles . Hypothesis: X likes the Beatles .
Answer: Entailment .
Premise: It is not true that X loves the Beatles . Hypothesis: X likes the Beatles .
Answer: Neutral .
Premise: Alex loves fruits . Hypothesis: John hates pears .
Answer: Neutral .
Premise: Alice changes often . Hypothesis: Alice changes often .
Answer: Entailment .
Premise: It is not true that Jack is happy . Hypothesis: Jack is unhappy .
Answer: Entailment .
Premise: X hates cats . Hypothesis: X is a cat person .
Answer: Contradiction .
Premise: David has blonde hair . Hypothesis: David likes football .
Answer: Neutral .
Premise: X are a child . Hypothesis: X is a kid .
Answer: Entailment .
Premise: It is not true that X lives in France . Hypothesis: X lives in France .
Answer: Contradiction .

Now it's your turn.\n"""


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
                prompt = "Premise: " + contradicted_premise + " Hypothesis: " + pair[1]
                prompt_full = (self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS + prompt + "\n").strip()
                model_text_pairs.append(f"{self.B_INST} {prompt_full.strip()} {self.E_INST}")
        
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
                generation_tokens = self.model.generate(**model_input, max_new_tokens=20, temperature=0.00001, repetition_penalty=1.1)
                generated = self.tokenizer.batch_decode(generation_tokens,skip_special_tokens=True)

            for j, gen in enumerate(generated):
                index = model_text_pairs_indexes_batch[j]
                contradicted_premise = self.contradict_premise(text_pairs[index][0], contrad_premise)

                prompt = "Premise: " + contradicted_premise + " Hypothesis: " +  text_pairs[index][1]
                prompt_full = (self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS + prompt + "\n").strip()

                gen = gen.replace(f"{self.B_INST} {prompt_full.strip()} {self.E_INST}", "")

                if "Entailment" in gen:
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

"""entailment_model = EntailmentModel("meta-llama/Llama-2-13b-chat-hf")
entailment_model.entail("X appear in the company today .", "X is a person who has lunch in the company .")
entailment_model.entail("X schedules meetings with their customers .", "X is a employee who schedule a meeting with their customer .")
entailment_model.entail("X is a employee who have lunch in the company .", "X is a employee who schedule a meeting with their customer .")
entailment_model.entail("X is a employee who have lunch in the company .", "X is a person who has lunch in the company .")
entailment_model.entail("X appears in the company today .", "X appears in the company today .")
entailment_model.entail("X schedules meetings with their customers .", "X appear in the company today .")
entailment_model.entail("Sam is a sterpus.","Sam is a wumpus .")
entailment_model.entail("Sam is a sterpuse.","Sam is a sterpus .")
entailment_model.entail("Sam is a sterpus.","Sam is a sterpus .")"""


class LLaMaBatchEntailmentModel(LLaMaEntailmentModel):
    def __init__(self, model_name):
        super().__init__(model_name)
    
    def entail(self, pairs, correspondance_dict,batch_size, contrad_premise=False):
        #print(len(pairs), batch_size, len(pairs) > batch_size)
        if len(pairs) > batch_size:
            #print("left", len(pairs[:batch_size]), "right", len(pairs[batch_size:]))
            left = self.entail(pairs[:batch_size], correspondance_dict, batch_size, contrad_premise) 
            right = self.entail(pairs[batch_size:], correspondance_dict, batch_size, contrad_premise)
            #print("right done")
            return left + right
        #print("normal loop")
        if not contrad_premise:
            contrad_premise = ""
        else:
            contrad_premise = "It is not true that "
        text_pairs = [(correspondance_dict[pair[0]], correspondance_dict[pair[1]]) for pair in pairs]
        result = []
        model_text_pairs_indexes = []
        model_text_pairs = []
        for i, pair in enumerate(text_pairs):
            res_symbo =  self.entail_symbolic(contrad_premise + pair[0], pair[1])
            if res_symbo in {"E", "C"}:
                result.append((pairs[i], res_symbo))
            else:
                model_text_pairs_indexes.append(i)
                prompt = "Premise: " + contrad_premise + pair[0] + " Hypothesis: " + pair[1]
                prompt_full = (self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS + prompt + "\n").strip()
                model_text_pairs.append(f"{self.B_INST} {prompt_full.strip()} {self.E_INST}")
        
        if len(model_text_pairs) == 0:
            return result
            
        model_input = self.tokenizer(model_text_pairs, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generation_tokens = self.model.generate(**model_input, max_new_tokens=20, temperature=0.00001, repetition_penalty=1.1)
            generated = self.tokenizer.batch_decode(generation_tokens,skip_special_tokens=True)

        for j, gen in enumerate(generated):
            prompt = "Premise: " + contrad_premise + text_pairs[model_text_pairs_indexes[j]][0] + " Hypothesis: " +  text_pairs[model_text_pairs_indexes[j]][1]
            prompt_full = (self.B_SYS + self.DEFAULT_SYSTEM_PROMPT + self.E_SYS + prompt + "\n").strip()
            model_text_pairs.append(f"{self.B_INST} {prompt_full.strip()} {self.E_INST}")
            gen = gen.replace(f"{self.B_INST} {prompt_full.strip()} {self.E_INST}", "")
            print(pairs[model_text_pairs_indexes[j]], text_pairs[model_text_pairs_indexes[j]], gen)
            if "Entailment" in gen:
                result.append((pairs[model_text_pairs_indexes[j]], "E"))
            elif "Contradiction" in gen:
                result.append((pairs[model_text_pairs_indexes[j]], "C"))
        
        return result