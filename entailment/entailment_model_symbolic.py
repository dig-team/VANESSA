import torch

class SymbolicEntailmentModel():
    def __init__(self):
        pass
        
    def entail(self, pairs, correspondance_dict, batch_size, contrad_premise=False, entailment_cache={}):
        result = []
        if not contrad_premise:
            contrad_premise = ""
        else:
            contrad_premise = "It is not true that "
        for pair in pairs:
            res_symbo = self.entail_single(contrad_premise + correspondance_dict[pair[0]], correspondance_dict[pair[1]])
            if res_symbo in {"E", "C"}:
                result.append((pair, res_symbo))

        return result, entailment_cache

    def entail_single(self, premise, hypothesis):
        premise = premise.lower()
        hypothesis = hypothesis.lower()
        if premise.startswith("it is not true that ") and self.entail_single(premise[20:], hypothesis) == "E":
            return "C"
        elif premise.startswith("it is not true that ") and self.entail_single(premise[20:], hypothesis) == "C":
            return "E"

        if premise.count(" not ") == hypothesis.count(" not ") and premise.count("n't") == hypothesis.count("n't"):
            if premise == hypothesis:
                return "E" 
            elif premise.replace("a ", "").replace("thing ", "").replace("person ", "") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", ""):
                return "E"
            elif premise.replace(" are ", " is ") == hypothesis.replace(" are ", " is "):
                return "E"
            elif premise.replace("ves ", "f ").replace("es ", " ").replace("s ", " ") == hypothesis.replace("ves ", "f ").replace("es ", " ").replace("s ", " "):
                return "E"
            elif premise.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is ") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is "):
                return "E"
            elif premise.replace("a ", "").replace("thing ", "").replace("person ", "").replace("ves ", "f ").replace("es ", " ").replace("s ", " ") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", "").replace("ves ", "f ").replace("es ", " ").replace("s ", " "):
                return "E"
            elif premise.replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", " ") == hypothesis.replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", " "):
                return "E"
            elif premise.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", " ") == hypothesis.replace("a ", "").replace("thing ", "").replace("person ", "").replace(" are ", " is ").replace("ves ", "f ").replace("es ", " ").replace("s ", " "):
                return "E"
            return "N"
        else:
            if self.entail_single(remove_does_not(premise).replace(" not ", " ").replace("n't",""), remove_does_not(hypothesis).replace(" not ", " ").replace("n't","")) == "E":
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