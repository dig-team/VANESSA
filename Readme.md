# VANESSA - Verifying the Steps of Deductive Reasoning Chains

Resources for the paper "[**Verifying the Steps of Deductive Reasoning Chains **](https://sadzac.github.io/files/VANESSA_verification.pdf)", published at ACL 2025.

Authors: [Zacchary Sadeddine](https://sadzac.github.io/) and [Fabian Suchanek](https://www.suchanek.name/).

## Main use
python QA.py TASK DATASET_NAME NLI_MODEL DATASET_VERSION

Datasets / Versions: 
- FOLIO
    - LLaMa2
    - LLaMa3
    - Mixtral
- ProofWriter
    - neg
    - remove
    - hallu
- ProntoQA
    - LLaMa2
    - LLaMa3
    - Mixtral
- EntailmentBank
    - neg
    - hallu

Tasks:
- parsing
- entailments
- reasoning
- full_validity (performs parsing + entailments + reasoning)
- direct_LLM
- consistency (sentence-wise consistency)
- consistency_FC (full context consistency)
- consistency_VANESSA (consistency using VANESSA)

NLI_MODEL:
- None (for parsing and consistency tasks. In consistency, will perform string matching)
- Symbolic
- Deberta
- LLaMa3
- Mistral
- GPT 3.5 Turbo

## Results
Results are saved in results/<task>/<version>_<dataset>-<nli_model>-<date>.jsonl

You can find our reported results in results/reasoning (for validity) and results/consistency (for groundedness)

results/analysis provides scripts to get metrics