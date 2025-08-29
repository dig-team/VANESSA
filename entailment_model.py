from entailment.entailment_model_T5 import T5EntailmentModel
from entailment.entailment_model_LLaMa import LLaMaEntailmentModel, LLaMaBatchEntailmentModel
from entailment.entailment_model_symbolic import SymbolicEntailmentModel
from entailment.entailment_model_Mistral import MistralEntailmentModel
from entailment.entailment_model_LLaMa3 import LLaMa3EntailmentModel
from entailment.entailment_model_GPT import GPTEntailmentModel
from entailment.entailment_model_Deberta import DeBERTaEntailmentModel
from entailment.entailment_model_LLaMa3_ollama import OLLaMa3EntailmentModel

def initialize_entailment_model(model_name):
    if model_name == "T5":
        return T5EntailmentModel("google/flan-t5-xxl")
    elif model_name == "LLaMa":
        return LLaMaEntailmentModel("meta-llama/Llama-2-13b-chat-hf")
    elif model_name == "Symbolic":
        return SymbolicEntailmentModel()
    elif model_name == "LLaMaSoft":
        return LLaMaSoftEntailmentModel("meta-llama/Llama-2-13b-chat-hf")
    elif model_name == "Mistral":
        return MistralEntailmentModel()
    elif model_name == "LLaMa3":
        return LLaMa3EntailmentModel()
    elif model_name == "LLaMaBatch":
        return LLaMaBatchEntailmentModel("meta-llama/Llama-2-13b-chat-hf")
    elif model_name == "GPT":
        return GPTEntailmentModel()
    elif model_name == "Deberta":
        return DeBERTaEntailmentModel("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    elif model_name == "oLLaMa3":
        return OLLaMa3EntailmentModel()
    else:
        raise ValueError("Invalid model name")
