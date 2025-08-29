from collections import defaultdict
import string

def rewrite_entailment(string, new_variables):
    if string.startswith("("):
        return string[:-2]+new_variables[string[-2]]+")"
    else:
        return new_variables[string]

def clean_ent(string):
    if string.startswith("("):
        return string[1] + " " +  string[2:-1]
    

def rewrite_formula(string, new_variables):
    new_string = ''
    for c in string:
        if c in new_variables:
            new_string += new_variables[c]
        else:
            new_string += c
    return new_string
            

def prepare_entailments(entailments, new_variables, correspondance_dict):
    new_entailments = {}
    lines = []
    for k,v in entailments.items():
        for o in v:
            line = rewrite_entailment(k, new_variables) + " ▷ " + rewrite_entailment(o, new_variables)
            full_text = rewrite_entailment(k, correspondance_dict) + " ▷ " + rewrite_entailment(o, correspondance_dict)
            lines.append("<b>"+line+"</b>" + ": " + full_text)
    return lines
    
def get_new_variables(correspondance_dict):
    new_variables = {}
    for i,k in enumerate(correspondance_dict):
        new_variables[k] = str(i)
    return new_variables

def prepare_premises(log_premises, correspondance_dict, premises, new_variables):
    uninstantiations = [get_uninstantiated_version(p, log_premises, correspondance_dict) for p in log_premises]
    log_premises_uninstantiated = [u[0] for u in uninstantiations]
    uninstantation_dicts = [u[1] for u in uninstantiations]
    premises_new_text = []
    instances = get_instances(uninstantation_dicts, correspondance_dict)
    instances = ["<b>Instances:</b> " + ", ".join(i) for i in instances]
    instances = [max(instances, key=len)]
    if instances == "<b>Instances:</b> ":  # No instances
        instances = []
    for i, prem in enumerate(log_premises_uninstantiated):
        prem_new_text = prem
        prem_new_text = ""
        for char in prem:
            if not char in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}:
                char = correspondance_dict[char]
            prem_new_text += char
        if prem_new_text.startswith("(") and prem_new_text.endswith(")"):
            prem_new_text=prem_new_text[1:-1]
        prem_new_text = prem_new_text.replace("→", " → ").replace("∧", " ∧ ").replace("∨", " ∨ ").replace("¬", " ¬ ").replace("▷", " ▷ ").replace("∪", " ∪ ").replace(" .", "").replace("⊻", " ⊻ ")
        premises_new_text.append(prem_new_text)
    return ["<b>"+p+"</b>" + "<br>" + "&nbsp;&nbsp;&nbsp;&nbsp;" + pn + "<br>" + "&nbsp;&nbsp;&nbsp;&nbsp;" + "Logic Version: " + rewrite_formula(log_prem, new_variables) for p, pn, log_prem in zip(premises, premises_new_text, log_premises)] + instances

def get_instances(uninstantation_dicts, correspondance_dict):
    instances = []
    for d in uninstantation_dicts:
        prem_instances = set()
        for k, v in d.items():
            X_text = correspondance_dict[v]
            instantiated_text = correspondance_dict[k]

            original_x_index = X_text.find('X ')

            # The replacement is the difference between the modified text and the original text
            replacement = instantiated_text[original_x_index:len(instantiated_text) - len(X_text)+1]  # The new part replacing 'X'
            prem_instances.add("'"+replacement+"'")
        instances.append(prem_instances)
    return instances

def prepare_proof_lines(proof_lines, correspondance_dict, entailments, log_premises, new_variables):
    new_lines = []
    for line in proof_lines:
        parsed_line = parse_proof_line(line)
        if parsed_line:
            line = ""
            line_number, premises, conclusion, line_type, original_lines = parsed_line
            if line_type == "ax" and "→" in conclusion and conclusion.split("→")[0] in entailments and conclusion.split("→")[1] in entailments[conclusion.split("→")[0]]:
                splits = conclusion.split("→")
                line_type = "ent"
                conclusion = "▷".join(splits)
            
            elif line_type == "ax": #an axiom but not an entailment, we want to get its uninstantiated version if it exists
                corresponding_premise = None
                conc_chars = set([c for c in conclusion if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}])
                for prem in log_premises:
                    prem_chars = set([c for c in prem if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", '⊻'}])
                    if len(conc_chars.intersection(prem_chars)) > 1:
                        corresponding_premise = prem
                        break
                if corresponding_premise is not None:
                    conclusion = corresponding_premise


            if premises == []:
                line = "<b>"+str(line_number)+"</b>" + ". " + rewrite_formula(conclusion, new_variables) + " (" + line_type + " " + ", ".join([str(o) for o in original_lines]) + ")"
            else:
                premises = [rewrite_formula(p, new_variables) for p in premises]
                line["text"] = "</b>"+str(line_number)+"</b>" + ". " + ", ".join(premises) + " ⊢ " + rewrite_formula(conclusion, new_variables) + " (" + line_type + " " + ", ".join([str(o) for o in original_lines]) + ")"



            new_lines.append(line.replace(" )", ")"))
    return new_lines


def get_variables_nb(premise):
    if "∪" in premise:
        premise = premise[:premise.index("∪")]
    return len([c for c in premise if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}])

def get_uninstantiated_version(premise, premises, correspondance_dict):
    if "∪" not in premise:
        return premise, {}
    characters = string.printable[:68] + string.printable[71:-6] + "".join([chr(i) for i in range(200, 400)])
    premises_lengths = [get_variables_nb(p) for p in premises]
    premise_index = premises.index(premise)
    previous_lengths = sum(premises_lengths[:premise_index])
    variables = list(correspondance_dict.keys())
    variables = characters[:len(correspondance_dict)]
    uninstantiated_variables = variables[previous_lengths:previous_lengths+premises_lengths[premise_index]]
    new_premise = ""
    i = 0
    for c in premise.split("∪")[0][1:]:
        if c in correspondance_dict:
            new_premise += uninstantiated_variables[i]
            i+=1
        else:
            new_premise += c
    i=0
    uninstantiation_dict = {}
    for c in premise:
        if c == "∪":
            i=0
        if c in correspondance_dict:
            uninstantiation_dict[c] = uninstantiated_variables[i]
            i+=1
    if new_premise.startswith("(") and new_premise.endswith(")"):
        new_premise=new_premise[1:-1]
    return new_premise, uninstantiation_dict

def parse_proof_line(line):
    line_number = int(line[:line.index(".")])
    premises = line[line.index(".") + 1: line.index("⊢")].strip().split(", ")
    premises = [p for p in premises if p!=""]
    conclusion = line[line.index("⊢") + 2: line.rfind("(")].strip()
    conclusion = conclusion[1:-1] if conclusion[0] == "(" else conclusion
    if "-" in line[line.rfind("("):]:
        line_type = line[line.rfind("(") + 1: line.index("-")].strip()
        origin_lines = line[line.index("-") + 1: -1].strip().split(", ")
        origin_lines = [int(o) for o in origin_lines if o!=""]
    else:
        line_type = line[line.rfind("(") + 1: -1].strip()
        origin_lines = []
    return line_number, premises, conclusion, line_type, origin_lines

def clean_proof_graph(proof_lines, correspondance_dict, entailments, log_premises):
    entailments_proof = {}
    new_lines = []
    uninstantation_dictionary = {}
    uninstantiation_lines = set()
    uninstantiation_new_links = defaultdict(list)
    for line in proof_lines:
        parsed_line = parse_proof_line(line)
        if parsed_line:
            line = {}
            line_dict = {}
            line_number, premises, conclusion, line_type, original_lines = parsed_line
            if line_type == "ax" and "→" in conclusion and conclusion.split("→")[0] in entailments and conclusion.split("→")[1] in entailments[conclusion.split("→")[0]]:
                splits = conclusion.split("→")
                line_type = "ent-rem"
                conclusion = "▷".join(splits)
                entailments_proof[line_number] = conclusion
            
            elif line_type == "ax": #an axiom but not an entailment, we want to get its uninstantiated version if it exists
                corresponding_premise = None
                conc_chars = set([c for c in conclusion if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}])
                for prem in log_premises:
                    prem_chars = set([c for c in prem if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}])
                    if len(conc_chars.intersection(prem_chars)) > 1:
                        corresponding_premise = prem
                        break
                if corresponding_premise is not None:
                    uninstantiated_premise, uninstantiation_dict = get_uninstantiated_version(corresponding_premise, log_premises, correspondance_dict)
                    if uninstantiation_dict != {}:
                        uninstantation_dictionary = {**uninstantation_dictionary, **uninstantiation_dict}
                        conclusion = uninstantiated_premise
                        uninstantiation_lines.add(line_number)
                        uninstantiation_new_links[line_number] = line_number

            
            for prem in premises:
                for char in prem:
                    if not char in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}:
                        line_dict[char] = correspondance_dict[char]
            for char in conclusion:
                if not char in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}:
                    line_dict[char] = correspondance_dict[char]

            if premises == []:
                line["text"] = conclusion
                conclusion_full_text = ""
                for char in conclusion:
                    if char in line_dict:
                        char = line_dict[char]
                    conclusion_full_text += char
                line["full_text"] = conclusion_full_text.replace("⊢", " ⊢ ").replace("→", " → ").replace("∧", " ∧ ").replace("∨", " ∨ ").replace("¬", " ¬ ").replace("▷", " ▷ ").replace("⊻", " ⊻ ")
            else:
                line["text"] = ", ".join(premises) + " ⊢ " + conclusion
                conclusion_full_text = ""
                for char in "".join(premises) + conclusion:
                    if char in line_dict:
                        char = line_dict[char]
                    conclusion_full_text += char
                line["full_text"] = conclusion_full_text.replace("⊢", " ⊢ ").replace("→", " → ").replace("∧", " ∧ ").replace("∨", " ∨ ").replace("¬", " ¬ ").replace("▷", " ▷ ").replace("⊻", " ⊻ ")


            if len(original_lines)==1 and original_lines[0] in uninstantiation_lines:
                uninstantiation_lines.add(line_number)
                uninstantiation_new_links[line_number] = uninstantiation_new_links[original_lines[0]]
                continue

            for i,n in enumerate(original_lines):
                if n in entailments_proof:
                    line["ent"] = entailments_proof[n]
                    original_lines.remove(n)
                    line_type = "ent"
                    break
                if n in uninstantiation_lines:
                    original_lines[i] = uninstantiation_new_links[n]

            line["line_number"] = line_number
            line["line_type"] = line_type
            line["original_lines"] = original_lines
            line["line_dict"] = line_dict
            new_lines.append(line)
    for line in new_lines:
        print(line)
    print(uninstantiation_new_links)
    return new_lines
            


            