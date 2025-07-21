from entailment.entailment_model_T5 import T5EntailmentModel
from entailment.entailment_model_LLaMa2 import LLaMa2EntailmentModel
from entailment.entailment_model_symbolic import SymbolicEntailmentModel
from entailment.entailment_model_Mistral import MistralEntailmentModel
from entailment.entailment_model_LLaMa3 import LLaMa3EntailmentModel
from entailment.entailment_model_GPT import GPTEntailmentModel
from entailment.entailment_model_Deberta import DeBERTaEntailmentModel

def initialize_entailment_model(model_name):
    if model_name == "T5":
        return T5EntailmentModel()
    elif model_name == "LLaMa2":
        return LLaMa2EntailmentModel()
    elif model_name == "Symbolic":
        return SymbolicEntailmentModel()
    elif model_name == "Mistral":
        return MistralEntailmentModel()
    elif model_name == "LLaMa3":
        return LLaMa3EntailmentModel()
    elif model_name == "GPT":
        return GPTEntailmentModel()
    elif model_name == "Deberta":
        return DeBERTaEntailmentModel("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    else:
        raise ValueError("Invalid model name")
