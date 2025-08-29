from flask import Flask, render_template, request, jsonify, session
import time
import os
from QA import process_instance
from webapp.graph import build_graph
from utils import clear_cache, SpacyModel
from webapp.postprocessing import prepare_premises, prepare_proof_lines, get_new_variables, prepare_entailments
import json
from entailment_model import initialize_entailment_model

app = Flask(__name__, template_folder="webapp/")
app.secret_key = '@;q#^f*G8W%ZB?o'

with app.app_context():
    global Symbolic_model
    global spacyModel
    global LLaMa3_model
    print("o")
    Symbolic_model = initialize_entailment_model("Symbolic")
    spacyModel = SpacyModel()
    print("lllooooo")
    LLaMa3_model = None
    print("loaded")

@app.route('/')
def index():
    presets = {
        'Example 1': {
            'premises': "If someone likes bread, then they like chocolate or cheese.\nAnyone who likes eating tomatoes hates even the idea of cheese.\nLisa is the biggest tomato lover I know, but she also is a fan of bread.",
            'conclusion': 'Lisa is fond of chocolate.'
        },
        'Example 2': {
            'premises': "Every grimpus is a lorpus. Everything that is a zumpus, a shumpus, or a sterpus is a gorpus. Jompuses are gorpuses. Everything that is a wumpus, an impus, or a sterpus is a gorpus. Tumpuses are yumpuses. Everything that is a wumpus or a numpus or a dumpus is a lempus. Everything that is a wumpus or a numpus or a dumpus is a zumpus, a tumpus, and a jompus. Jompuses are impuses. Wren is a lorpus and a shumpus and a brimpus. Wren is a wumpus and a grimpus and a jompus.",
            'conclusion': 'Wren is a vumpus, a zumpus, or a grimpus.'
        },
        'Example 3': {
            'premises': "All employees who schedule a meeting with their customers will appear in the company today. Everyone who has lunch in the company schedules meetings with their customers. Employees will either have lunch in the company or have lunch at home. If an employee has lunch at home, then he/she is working remotely from home. All employees who are in other countries work remotely from home. No managers work remotely from home. James is either a manager and appears in the company today or neither a manager nor appears in the company today.",
            'conclusion': 'Is it true that James has lunch in the company?'
        }
        # Add more presets as needed
    }
    return render_template('index.html', presets=presets)

@app.after_request
def cleanup_gpu_memory(response):
    # Free GPU memory after each request
    clear_cache()
    # Additional cleanup can go here if needed
    return response

@app.route('/submit', methods=['POST'])
def submit():
    t = time.time()
    model_name = request.form['model']
    output_file = "webapp/" + "output_"+ model_name +".json"
    premises = request.form['premises'].replace("\n", "").strip()
    conclusion = request.form['conclusion']

    if conclusion.startswith("Is it true that "):
        conclusion = conclusion[16:]
    if conclusion.endswith('?'):
        conclusion = conclusion[:-1].strip()+"."
    question_conclusion = "Is it true that " + conclusion[:-1] + "?"
    
    # Validate premises and conclusion
    if not premises or len(premises.split('.')) < 2:
        return jsonify({"error": "Premises must contain at least 1 sentence."}), 400
    
    if not conclusion or not (len(conclusion.split('.')) == 2 and conclusion.endswith('.')):
        if not conclusion.endswith('?'):
            return jsonify({"error": "Conclusion must contain exactly 1 sentence."}), 400


    cache = json.load(open(output_file))
    cache = {frozenset([s.strip()+"." for s in k.split(".") if s.strip() != ""]): v for k, v in cache.items()}
    context = frozenset([s.strip()+"." for s in premises.split(".") if s.strip() != ""])
    
    known = False
    if context in cache:
        if question_conclusion in cache[context]:
            conclusion = question_conclusion
        if conclusion in cache[context]:
            known = True
            result = cache[context][conclusion]["result"]
            log_premises = cache[context][conclusion]["premises"]
            log_conclusion = cache[context][conclusion]["conclusion"]
            correspondance_dict = cache[context][conclusion]["correspondance"]
            entailments = cache[context][conclusion]["entailments"]
            proof_lines = cache[context][conclusion]["proof"]
            errors = cache[context][conclusion]["errors"]
    print("known", known)
    if not known:
        instance = {"text": premises, "question": conclusion}

        result, log_premises, log_conclusion, correspondance_dict, entailments, proof_lines, errors = main(instance, model_name, output_file, spacyModel)

    end_time = time.time()
    print("Time elapsed: ", end_time-t)

    if errors != []:
        print("errors", errors)
        return jsonify({'error': True})

    print("ouaiiis")
    
    output = {
        'premises': [p.strip()+"." for p in premises.split(".") if p.strip() != ""],
        'conclusion': conclusion,
        'log_premises': log_premises,
        'log_conclusion': log_conclusion,
        'correspondance_dict': correspondance_dict,
        'proof_lines': proof_lines,
        'entailments': entailments,
        'result': result,
    }
    print("a")
    session["output"] = output
    print("b")
    return jsonify({'error': False})

@app.route('/results.html')
def results():
    print("results")
    output = session.get("output")
    # Retrieve the output from session or pass it to the template

    proof_lines = session.get("output")["proof_lines"]
    correspondance_dict = session.get("output")["correspondance_dict"]
    entailments = session.get("output")["entailments"]
    log_premises = session.get("output")["log_premises"]
    log_conclusion = session.get("output")["log_conclusion"]

    graph_head, graph_body = build_graph(proof_lines, correspondance_dict, entailments, log_premises+[log_conclusion])

    new_variables = get_new_variables(correspondance_dict)

    premises = session.get("output")["premises"]
    conclusion = session.get("output")["conclusion"]

    output["premises_strings"] = prepare_premises(log_premises, correspondance_dict, premises, new_variables)
    output["conclusion_string"] = prepare_premises([log_conclusion], correspondance_dict, [conclusion], new_variables)[0]
    output["proof_lines"] = prepare_proof_lines(proof_lines, correspondance_dict, entailments, log_premises+[log_conclusion], new_variables)
    output["entailments"] = prepare_entailments(entailments, new_variables, correspondance_dict)
    output["correspondance_dict"] = {new_variables[k]: v for k,v in correspondance_dict.items()}
    return render_template('results.html', output=output, graph_head=graph_head, graph_body=graph_body)

#@app.route('/graph')
def graph():
    proof_lines = session.get("output")["proof_lines"]
    correspondance_dict = session.get("output")["correspondance_dict"]
    entailments = session.get("output")["entailments"]
    log_premises = session.get("output")["log_premises"]
    G = build_graph(proof_lines, correspondance_dict, entailments, log_premises)
    #data = json_graph.node_link_data(G)
    #return jsonify(data)
        # Créer un objet Pyvis Network
    net = Network(directed=True)
    net.from_nx(G)
    graph_html = net.generate_html()


def main(instance, model_name, output_file, spacy_model):
    global LLaMa3_model
    if model_name == "LLaMa3":
        if LLaMa3_model is None:
            LLaMa3_model = initialize_entailment_model("LLaMa3")
        entailment_model = LLaMa3_model
    else:
        entailment_model = Symbolic_model
    
    entailment_cache_file = "cache/" + model_name + "_cache.json"

    result, premises, conclusion, correspondance_dict, entailments_dict, proof_lines, errors = process_instance(instance, spacy_model, entailment_model, entailment_cache_file, "cache/parsing_cache.json", "cache/instances_cache.json")


    with open(output_file, 'r', encoding="utf-8") as f:
        output_data = json.load(f)

    new_instance = {}
    context = frozenset([s.strip()+"." for s in instance["text"].split(".") if s.strip() != ""])
    context = instance["text"]
    question = instance["question"]

    if context not in output_data:
        output_data[context] = {}

    for i in range(len(errors)):
        if "Logic Transform" in errors[i]:
            errors[i] = "Parsing"
        elif "Entailments Error" in errors[i]:
            errors[i] = "Entailments"
        elif "Reasoning Error: maximum recursion depth exceeded" in errors[i]:
            errors[i] = ""
        elif "Reasoning Error" in errors[i]:
            errors[i] = "Reasoning"
    
    if errors != []:
        return result, premises, conclusion, correspondance_dict, entailments_dict, proof_lines, errors

    output_data[context][question] = {
        "result": result,
        "proof": proof_lines,
        "premises": [str(prem) for prem in premises],
        "conclusion": str(conclusion),
        "correspondance": correspondance_dict,
        "entailments": {str(key): [str(val) for val in value] for (key, value) in entailments_dict.items()},
        "errors": errors
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f)

    return output_data[context][question]["result"], output_data[context][question]["premises"], output_data[context][question]["conclusion"], output_data[context][question]["correspondance"], output_data[context][question]["entailments"], output_data[context][question]["proof"], output_data[context][question]["errors"]

if __name__ == '__main__':
    app.run()
