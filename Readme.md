# VANESSA - Neuro-Symbolic Reasoning with Textual Entailment

## Web Application
To run the Web Application locally:
- Clone this repo
- Install requirements.txt
- $python app.py
- Spacy Models loads
- You can access the Web App at 127.0.0.1:5000
- If you choose to run with LLaMa3 (8B) model, it will load once at the first request - ensure you have enough GPU memory (~20GB)
- Requests can take some time to run on new input (~3-5 mins for a completely new input)

## Reproducing Experiments from the paper
python QA.py DATASET_NAME NLI_MODEL DATASET_VERSION

Datasets: FOLIO, ProofWriter, ProntoQA, LogicBench
Versions (for LogicBench): PBD, PCD, PDD, PMT, PHS, PDS