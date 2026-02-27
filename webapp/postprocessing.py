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
                line = "<b>"+str(line_number)+"</b>" + ". " + ", ".join(premises) + " ⊢ " + rewrite_formula(conclusion, new_variables) + " (" + line_type + " " + ", ".join([str(o) for o in original_lines]) + ")"


            new_lines.append(line.replace(" )", ")"))
    return new_lines

def find_nearest_original(proof_lines, new_proof_lines, original_line):
    kept_lines = set([l[0] for l in new_proof_lines])
    if original_line in kept_lines:
        return original_line
    else:
        parsed_lines = [parse_proof_line(l) for l in proof_lines]
        for line in parsed_lines:
            line_number, premises, conclusion, line_type, original_lines = line
            if line_number == original_line:
                for o in original_lines:
                    nearest = find_nearest_original(proof_lines, new_proof_lines, o)
                    if nearest is not None:
                        return nearest


def is_instantiation_transition(formula, premises):
    def extract_base(formula):
        if not formula.startswith("(") or not formula.endswith(")"):
            formula = "(" + formula + ")"
        parentheses_count = 0  # Compteur des parenthèses ouvrantes et fermantes
        subformula = ""  # Chaîne pour stocker le sous-ensemble en cours
        for i, char in enumerate(formula):
            subformula += char  # Ajoute le caractère courant à la sous-formule
            # Si le caractère est une parenthèse ouvrante, on incrémente le compteur
            if char == "(":
                parentheses_count += 1
            # Si le caractère est une parenthèse fermante, on décrémente le compteur
            elif char == ")":
                parentheses_count -= 1
            
            # Lorsque le compteur est revenu à 0, cela signifie que la sous-formule est complète
            if parentheses_count == 0:
                return subformula.strip()  # On retourne la sous-formule équilibrée
        
        return None  # Si aucune sous-formule complète n'a été trouvée
    formula_base = extract_base(formula)

    formula_chars = set([c for c in formula_base if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}])
    for prem in premises:
        prem_chars = set([c for c in prem if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", '⊻'}])
        if len(formula_chars.intersection(prem_chars)) > 1 and "∪" in prem:
            premise_set = set([c for c in prem.split("∪")[0] if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", '⊻'}])
            print(len(formula_chars), len(premise_set))
            if len(formula_chars) == len(premise_set):
                return True
    return False

def process_instant_origins(proof_lines, instant_origins):
    #instant origins provides "paths" of the form original line -> line with ∧e -> line with ∧e -> ...
    #for each path, extract the start, end and intermediate lines
    remove, starts, ends = set(), set(), defaultdict(list)
    visited = set()
    for o, l in instant_origins.items():
        if o in visited:
            continue
        path = [o]
        while l in instant_origins:
            path.append(l)
            visited.add(l)
            l = instant_origins[l]
        path.append(l)
        remove.update(path[1:-1])
        starts.add(path[0])
        ends[path[0]].append(path[-1])
    orig_info = {}
    npl = []
    for line_number, premises, conclusion, line_type, original_lines in proof_lines:
        if line_number in remove:
            continue
        elif line_number in starts:
            for end in ends[line_number]:
                orig_info[end] = original_lines
        elif line_number in orig_info:
            npl.append((line_number, premises, conclusion, "inst", orig_info[line_number]))
        else:
            npl.append((line_number, premises, conclusion, line_type, original_lines))
            
    return npl

    
def get_text(formula, correspondance_dict):
    formula_text = ""
    for char in formula:
        if not char in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}:
            char = correspondance_dict[char]
        formula_text += char
    if formula_text.startswith("(") and formula_text.endswith(")"):
        formula_text=formula_text[1:-1]
    formula_text = formula_text.replace("→", " → ").replace("∧", " ∧ ").replace("∨", " ∨ ").replace("¬", " ¬ ").replace("▷", " ▷ ").replace("∪", " ∪ ").replace(" .", "").replace("⊻", " ⊻ ")
    return formula_text

def get_axiom_text(formula, log_premises, premises):
    print(log_premises)
    print(formula)
    print(formula in log_premises)

def prepare_textual_proof(proof_lines, correspondance_dict, entailments, log_premises, new_variables, premises):
    new_lines = []
    new_proof_lines = []
    entailments_lines = set()
    instant_origins = {}
    print("Starting to prepare textual proof...")
    print(log_premises)
    for line in proof_lines:
        print("Processing line:", line)
        parsed_line = parse_proof_line(line)
        if parsed_line:
            line = ""
            line_number, premises, conclusion, line_type, original_lines = parsed_line
            if line_type == "ax" and "→" in conclusion and conclusion.split("→")[0] in entailments and conclusion.split("→")[1] in entailments[conclusion.split("→")[0]]: #repère les entailments parmi les axiomes
                splits = conclusion.split("→")
                line_type = "ent"
                conclusion = "▷".join(splits)
                entailments_lines.add(line_number)
            
            elif line_type == "ax": #an axiom but not an entailment, we want to get its uninstantiated version if it exists
                corresponding_premise = None
                print("Looking for uninstantiated version of:", conclusion)
                conc_chars = set([c for c in conclusion if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}])
                for prem in log_premises:
                    prem_chars = set([c for c in prem if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", '⊻'}])
                    if len(conc_chars.intersection(prem_chars)) > 1:
                        corresponding_premise = prem
                        break
                if corresponding_premise is not None:
                    uninstantiated_premise, uninstantiation_dict = get_uninstantiated_premise(corresponding_premise, log_premises, correspondance_dict)
                    if conclusion != uninstantiated_premise:
                        line_type = "inst"

            #On modifie la ligne selon certaines conditions
            if line_type == "ent": #Si la ligne est un entailment, on la met pas
                pass
            elif line_type == '→e': #Si c'est une deduction dûe à un entailment, on marque comme tel et on enlève l'origine. Si la conclusion a le même texte 
                orig_lines = []
                for o in original_lines:
                    if o in entailments_lines:
                        line_type = "ded-ent"
                    else:
                        orig_lines.append(o)
                if line_type == "ded-ent" and len(orig_lines) == 1:#Si c'est une déduction dûe à des entailments, on vérifie si le texte de la conclusion est le même que le texte de la conclusion d'origine. Si oui il faut fusionner les 2 lignes
                    orig_line = orig_lines[0]
                    for index, (np_line_number, np_premises, np_conclusion, np_line_type, np_original_lines) in enumerate(new_proof_lines):
                        if np_line_number == orig_line:
                            if get_text(np_conclusion, correspondance_dict) == get_text(conclusion, correspondance_dict):
                                new_proof_lines[index] = (line_number, np_premises, np_conclusion, np_line_type, np_original_lines)
                            else:
                                new_proof_lines.append((line_number, premises, conclusion, line_type, orig_lines))
                else:
                    new_proof_lines.append((line_number, premises, conclusion, line_type, orig_lines))
            elif line_type == 'inst': #Si c'est une instanciation, on crée 2 lignes, une pour l'axiome de base et une pour l'instanciation, et on relie les deux
                new_line_number = max(max([int(l[0]) for l in new_proof_lines]+[0]), max([int(l[0]) for l in proof_lines])) + 1
                new_proof_lines.append((str(new_line_number), premises, uninstantiated_premise, 'ax', []))
                new_proof_lines.append((line_number, premises, conclusion, line_type, [new_line_number]))
            elif line_type == 'ax':
                new_proof_lines.append((line_number, premises, conclusion, line_type, original_lines))
            elif line_type == '∧e' and is_instantiation_transition(conclusion, log_premises):
                instant_origins[original_lines[0]] = line_number
                new_proof_lines.append((line_number, premises, conclusion, line_type, original_lines))
            else:
                new_proof_lines.append((line_number, premises, conclusion, line_type, original_lines))
            

    new_proof_lines = process_instant_origins(new_proof_lines, instant_origins)
    new_lines_dict = {}
    for i, line in enumerate(new_proof_lines):
        line_number, premises, conclusion, line_type, original_lines = line
        
        #Je veux pouvoir afficher le texte de l'axiome de base pour les lignes d'axiome
        if line_type == "ax":
            get_axiom_text(conclusion, log_premises, premises)
            print("---")

        new_lines_dict[int(line_number)] = i+1
        line_number = str(i+1)

        prem_new_text = ""
        for char in conclusion:
            if not char in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}:
                char = correspondance_dict[char]
            prem_new_text += char
        if prem_new_text.startswith("(") and prem_new_text.endswith(")"):
            prem_new_text=prem_new_text[1:-1]
        prem_new_text = prem_new_text.replace("→", " → ").replace("∧", " ∧ ").replace("∨", " ∨ ").replace("¬", " ¬ ").replace("▷", " ▷ ").replace("∪", " ∪ ").replace(" .", "").replace("⊻", " ⊻ ")

        if premises == []:
            line = "<b>"+str(line_number)+"</b>" + ". " + rewrite_formula(conclusion, new_variables) + " (" + line_type + " " + ", ".join([str(new_lines_dict[o]) for o in original_lines]) + ")"
            line = "<b>"+str(line_number)+"</b>" + ". " + get_text(conclusion, correspondance_dict) + " (" + line_type + " " + ", ".join([str(new_lines_dict[o]) for o in original_lines]) + ")"
            line = prepare_tp_line(line_number, premises, conclusion, line_type, original_lines, correspondance_dict, new_lines_dict)
        else:
            #premises = [rewrite_formula(p, new_variables) for p in premises]
            line = "<b>"+str(line_number)+"</b>" + ". " + ", ".join(premises) + " ⊢ " + rewrite_formula(conclusion, new_variables) + " (" + line_type + " " + ", ".join([str(new_lines_dict[o]) for o in original_lines]) + ")"
            line = prepare_tp_line(line_number, premises, conclusion, line_type, original_lines, correspondance_dict, new_lines_dict)
        new_lines.append(line.replace(" )", ")"))
    return new_lines


def prepare_tp_line(line_number, premises, conclusion, line_type, original_lines, correspondance_dict, new_lines_dict):
    def get_prefix(line_type, original_lines):
        if line_type == "ax":
            return "Input"
        elif line_type == "ded-ent":
            return f"Entailment (from {', '.join([str(new_lines_dict[o]) for o in original_lines])})"
        elif line_type == "inst":
            return "Instantiation of " + str(new_lines_dict[original_lines[0]])
        else:
            return f"Deduction ({'+'.join([str(new_lines_dict[o]) for o in original_lines])})"
    prefix = get_prefix(line_type, original_lines)
    line = "<b>"+str(line_number)+"</b>" + ". " + prefix + ": " + get_text(conclusion, correspondance_dict)
    if premises != []:
        if len(premises) == 1:
            print(premises[0], conclusion, line_number)
            suffix = f" (supposing that '{get_text(premises[0], correspondance_dict)}' holds)"
        else:
            suffix = f" (supposing that {', '.join([get_text(p, correspondance_dict) for p in premises])} hold)"
        line += suffix
    return line

        

def get_variables_nb(premise):
    if "∪" in premise:
        premise = premise[:premise.index("∪")]
    return len([c for c in premise if c not in {"∧", "∨", "→", "¬", "▷", "∪", "(", ")", "", " ", "⊻"}])

def get_uninstantiated_premise(premise, premises, correspondance_dict):
    characters = string.printable[:68] + string.printable[71:-6] + "".join([chr(i) for i in range(200, 400)])
    premises_lengths = [get_variables_nb(p) for p in premises]
    premise_index = premises.index(premise)
    previous_lengths = sum(premises_lengths[:premise_index])
    variables = list(correspondance_dict.keys())
    variables = characters[:len(correspondance_dict)]
    uninstantiated_variables = variables[previous_lengths:previous_lengths+premises_lengths[premise_index]]
    new_premise = ""
    i = 0
    if "∪" not in premise and premise.startswith("("):
        premise = "("+premise[1:-1]
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
        #print("parsed", parsed_line)
        if parsed_line:
            line = {}
            line_dict = {}
            line_number, premises, conclusion, line_type, original_lines = parsed_line
            #print(len(premises))
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
            #print(len(premises), premises == [])
            if premises == []:
                #print("A")
                line["text"] = conclusion
                conclusion_full_text = ""
                for char in conclusion:
                    if char in line_dict:
                        char = line_dict[char]
                    conclusion_full_text += char
                line["full_text"] = conclusion_full_text.replace("⊢", " ⊢ ").replace("→", " → ").replace("∧", " ∧ ").replace("∨", " ∨ ").replace("¬", " ¬ ").replace("▷", " ▷ ").replace("⊻", " ⊻ ")
            else:
                #print("B")
                #print("premises", premises)
                line["text"] = ", ".join(premises) + " ⊢ " + conclusion
                #print(line["text"])
                conclusion_full_text = ""
                for char in "".join(premises):
                    if char in line_dict:
                        char = line_dict[char]
                    conclusion_full_text += char
                conclusion_full_text += "⊢"
                for char in conclusion:
                    if char in line_dict:
                        char = line_dict[char]
                    conclusion_full_text += char
                #print("full_text", conclusion_full_text)
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

            if len(original_lines) == 1:
                print("TEST?")
                for l in new_lines:
                    if l["line_number"] == original_lines[0]:
                        full_text = l["full_text"]        
                if full_text == line["full_text"]: #entailment with same text as original line, we can remove it and link its children to the original line
                    print("Found entailment with same text as original line, removing it and linking its children to the original line")
                    for new_line in new_lines:
                        if new_line["line_number"] in original_lines:
                            new_line["line_number"] = line_number
                    continue


            line["line_number"] = line_number
            line["line_type"] = line_type
            line["original_lines"] = original_lines
            line["line_dict"] = line_dict
            new_lines.append(line)
    for line in new_lines:
        pass
        #print("lin", line)
    #print("unl", uninstantiation_new_links)
    return new_lines
            


            