import re
from nltk.tag import pos_tag
import os, fnmatch
import torch
import spacy
import benepar
import lemminflect
from fastcoref import spacy_component
from negate import Negator
from coref_utils import resolve_coref_custom
from constituent_treelib import ConstituentTree, BracketedTree, Language, Structure
from datetime import date
from reasoner.logic import Literal, UnaryPredicate, And, Or, Implies, Not, InstancesOr, XOr
from tregex_parsing.main import Node
import string

class SpacyModel():
    def __init__(self):
        #self.nlp = spacy.load("en_core_web_trf")
        self.nlp = spacy.load("en_core_web_lg")
        config = {"attrs": {"tensor": None}}
        self.nlp.add_pipe("doc_cleaner", config=config)
        self.nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
        self.nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda'})
        #self.nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'})
    
    def process(self, text):
        with torch.no_grad():
            return self.nlp(text)

    def process_coref(self, text):
        with torch.no_grad():
            return resolve_coref_custom(self.nlp(text))

    def get_tree(self, sentence):
        return str(ConstituentTree(sentence, nlp=self.nlp))

def tree_preprocessing(lines):
    text = []
    for line in lines:
        new = line.strip()
        new = new.replace("\n", "")
        if "nucleus-satellite:Cause" in new:
            print("HIP")
            new = new.replace("nucleus-satellite:Cause", "satellite-nucleus:Cause")
        elif "satellite-nucleus:Cause" in new:
            print("HOP")
            new = new.replace("satellite-nucleus:Cause", "nucleus-satellite:Cause")
        elif "nucleus-nucleus:Condition" in new or "nucleus-nucleus:Cause" in new:
            new = new.replace("nucleus-nucleus:Condition", "nucleus-nucleus:and")
            new = new.replace("nucleus-nucleus:Cause", "nucleus-nucleus:and")
        if len(new.split(" ")) > 2:
            new = new.split(" ")
            new = [new[0], " ".join([new[1], new[2]]), " ".join([new[3], new[4]])]
        else:
            new = [new]
        text.extend(new)

    return text

def perform_coref(text, spacy_model):
    return spacy_model.process_coref(text)

def coref_preprocessing(lines, spacy_model):
    #Coreference Resolution
    print("COREF PREPROCESSING")
    edu_strings = [line.strip(" ") for line in lines]
    new_text = "".join(edu_strings)
    
    new_text = spacy_model.process_coref(new_text)
    coref_edus = new_text.split("\n")
    print("fini")
    print(new_text)
    return coref_edus

def singularize_np(span):
    out = []
    #print("singu NP", span)
    for token in span:
        if token.tag_ == "NNS": #attention je crois qu'il sait pas singulariser un truc avec une majuscule --> ne pas mettre la 1e lettre en majuscule
            sing = token._.inflect("NN")
            if token.is_oov:
                if token.text[-3:] == "ves":
                    sing = token.text[:-3] + "f"
                elif token.text[-3:] in {"ses", "xes", "zes"} or token.text[-4:] in {"ches", "shes"}:
                    sing = token.text[:-2]
                elif token.text[-3:] == "ies":
                    sing = token.text[:-3] + "y"
                elif token.text[-1:] == "s":
                    sing = token.text[:-1]
            if sing is not None:
                out.append(sing.lower())
            else:
                out.append(token.lower_)
        elif token.tag_ == "NNPS":
            out.append(token._.inflect("NNP"))
        elif token.tag_ == "NNP":
            out.append(token.text)
        else:
            out.append(token.lower_)
    return (" ".join(out))

def singularize_quantif(span, marker_word, det_detected=False): #both inputs are spans !
    print("singu quantif", span, marker_word)
    singular_verb_out = singularize_verbs(span, det_detected)
    #print(span._.parse_string)
    #print(span._.labels)
    #print(marker_word._.labels)
    if type(marker_word) != str:
        for token in span:
            if token == marker_word[0] or (token.i == marker_word[0].i and token.text == marker_word[0].text):
                if "NP" in marker_word._.labels: #Noun alone forms a NP
                    parent = marker_word
                else:
                    parent = marker_word._.parent
                break
        #print("singular np", parent)
        singular_np = singularize_np(parent)
        return singular_verb_out.replace(parent.text, singular_np)
    return singular_verb_out
    
def singularize_verbs(span, det_detected = False):
    #when det_detected is True in input, it means the function was recursively called. This applies only when there is a "are" in a universally quantified sentence, and we want to add "a" after the "is". In this case, we run singularize quantif, which will singularize the noun on a side and singularize verbs on the other (here), but we don't want to consider the NP so we directly turn det_detected at True.
    out = []
    verb_detected = False
    noun_detected = False
    insert_index = -1
    print("singu_vbs de ", span)
    if len(span) == 0:
        return ""
    init_index = span[0].i
    for token in span:
        if token.tag_ == "VBP":
            sing = token._.inflect("VBZ")
            if sing is not None:
                verb_detected = True
                out.append(sing)
                if sing == "is":
                    insert_index = len(out)
            else:
                out.append(token.text)
        elif token.text == "were":
            out.append("was")
            verb_detected = True
            insert_index = len(out)
        else:
            out.append(token.text)
        #print(token, noun_detected, verb_detected, det_detected, insert_index)
        if not noun_detected and verb_detected and not det_detected and insert_index > 0:
            if token.tag_ in {"DT", "PRP$", "WDT", "WP$", "POS", "CD"}:
                det_detected = True
            elif token.tag_ in {"NN", "NNS", "NNP", "NNPS"}:
                ind = token.i - init_index
                noun_detected = True
                noun_span = span[ind:ind + 1]
        #elif token.tag_ == "MD":
    if noun_detected:
        #print("NOUN DETECTED")
        result = singularize_quantif(span, noun_span, det_detected=True).split(" ")
        if out[insert_index] in {"not", "never", "no", "neither", "nor"}:
            insert_index += 1
        if out[insert_index][0].lower() in ["a", "e", "i", "o", "u", "y"]:
            result.insert(insert_index, "an")
        else:
            result.insert(insert_index, "a")
        #print("RESULTAT DE LA SINGU VBS", result)
        out = result
    return (" ".join(out))

def rel_simplify(string):
    #if string in ("Attribution", "belief"):
        #return "belief"
    if string in ("Disjunction", "or"):
        return "or"
    elif string in {"xor", "OR"}:
        return "xor"
    elif string in ("Condition", "Hypothetical", "imply", "Cause"):
        return "imply"
    elif string in ("Universal", "forall"):
        return "forall"
    elif string == "Not":
        return "not"
    return "and"

def clean_text(text):
    text = text.replace(" or .", " .").replace(" and .", " .").replace(", .", ".") #this is a bit too specific, but it's a quick fix
    text = text.strip()

    #Cleaning du text d'un EDU: 
    text = text.strip()
    print("CLEAN TEXT")
    #print(text)
    if re.sub("^(?:(or.[^,]*both.*\.))", "", text) == "":
        return ""

    #Enlever les mots inutiles au début et à la fin
    words = text.lower().split(" ")
    if "either" in words and "or" in words:
        if "both" not in words:
            either_index = words.index("either")
            or_index = words.index("or", either_index)
            words[or_index] = "OR"
            text = " ".join(words)
        else:
            text = re.sub("(?:(, or.[^,]*both.*\.))", "", text) #on enlève les "or both" ou autre, qui sont implied dans la disjonction classique

    text = re.sub(r"(?i)^(?:(?:that|and|or|but|because|if|then|either|also|,) )+","",text)
    text = re.sub(r"(?i)(?: (?:that|and|or|but|because|if|then|either|also|,)(\.| |))+$","",text)
    text = text.replace(" either "," ")
    #Majuscule au début et point à la fin
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if not text[-1] == ".":
        if text[-1] == " ":
            text = text + "."
        else:
            text = text + " ."

    return text

def clean_either(text):
    text = re.sub(r"(?i)^(?:(?:that|and|or|but|because|if|then|either|also|,) )+","",text)
    text = re.sub(r"(?i)(?: (?:that|and|or|but|because|if|then|either|also|,)(\.| |))+$","",text)

    text = text.replace(" either "," ")
    #Majuscule au début et point à la fin
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if not text[-1] == ".":
        if text[-1] == " ":
            text = text + "."
        else:
            text = text + " ."
    #il faudrait rajouter qqch pour la coref implicite ici
    #checker si y'a qqch avant le verbe? = Si c'est une phrase VP?
    # coref_edus = [[string] for string in coref_edus]
    #print(text)
    text = text.replace("both ", "")
    return text

def lifting(text):
    #on prend un texte
    #on extrait le sujet
    #on vérifie s'il est défini
    #si oui, on le remplace par _X (ou de manière plus générale la lift variable)
    #result.append(sentence, original_variable, lift_variable)
    #on fait pareil pour l'objet
    #en fait on le fait pour tous les GN? Juste quand on voit un prénom?
    words = text.split()
    tagged_sent = pos_tag(words)
    in_proper_group = False
    proper_nouns = []
    for word, tag in tagged_sent:
        if in_proper_group and tag == "NNP":
            proper_nouns[-1] = proper_nouns[-1] + " " + word
        elif tag == "NNP":
            in_proper_group = True
            proper_nouns.append(word)
        else:
            in_proper_group = False
    if proper_nouns != []:
        return proper_nouns
    return None

def get_conjunction_splits_old(sentence, conjunctions, spacy_model):
    #when I detect a "and", I take the biggest NP it belongs to.
    #Then I split by and / , which gives me all new_texts
    result = []
    with torch.no_grad():
        doc = spacy_model.process(sentence)
    sent = list(doc.sents)[0]
    print(sent._.parse_string)
    max_np_parent = ""
    print("CONJONCTIONS", conjunctions)
    for word, index in conjunctions:
        interest_conj = doc[index] #it should be the correct one, but possibility of error due to the spacy tokenizer
        ind = index
        while doc[ind].tag_ != "CC":
            ind+=1
        interest_conj = doc[ind]
        np_parent = find_biggest_parent(doc[ind:ind+1])
        print("NP PArent original", np_parent)
        if np_parent == doc[ind:ind+1]:
            print("NP Parent parent", np_parent._.parent)
            #print("Not S Parent", find_biggest_not_s_parent(doc[ind:ind+1]))
            #np_parent = find_biggest_not_s_parent(doc[index:index+1])
            np_parent = np_parent._.parent
            print(np_parent)
            
        
        splits_np = np_parent.text.replace(" " + word + " "," , ").replace("  ", " ").split(",") 
        print("SPLITS_NP", splits_np)

        splits = ["", ""]
        found = False
        for child in np_parent._.children:
            if child != doc[ind:ind+1]:
                if not found:
                    splits[0] = child.text
                else:
                    splits[1] = child.text
                    break
            else:
                found = True
        print("ALTERNATIVE SPLITS", splits)

        if word == "nor":
            negator = Negator(spacy_model, fail_on_unsupported=True)
            splits = [negator.negate_sentence((doc[:np_parent.start].text + " " + split_np.strip() + " " + doc[np_parent.end:].text).replace("neither", "not"), prefer_contractions=False) for split_np in splits_np if split_np != ""]
            word = "and"
        else:
            splits = [doc[:np_parent.start].text + " " + split_np.strip() + " " + doc[np_parent.end:].text for split_np in splits_np if split_np != ""]
            #ça c'est un truc de sauvage il faudrait faire un peu plus subtil

        if len(np_parent) > len(max_np_parent):
            max_np_parent = np_parent
            result = [(word, splits)]
    return result

def get_conjunction_splits(sentence, conjunctions, spacy_model):
    #when I detect a "and", I take its parent
    #Then I split by and / , which gives me all new_texts
    result = []
    with torch.no_grad():
        doc = spacy_model.process(sentence)
    sent = list(doc.sents)[0]
    max_parent = ""
    print(sent._.parse_string)
    print("CONJONCTIONS", conjunctions)
    #for word in doc:
        #print(word, word._.labels, word.tag_)
    for word, index in conjunctions:
        interest_conj = doc[index] #it should be the correct one, but possibility of error due to the spacy tokenizer
        ind = index
        while doc[ind].tag_ != "CC":
            ind+=1

        interest_conj = doc[ind]
        parent = doc[ind:ind+1]._.parent
        print("Parent original", parent)
        if parent == doc[ind:ind+1]:
            print("NP Parent parent", parent._.parent)
            parent = parent._.parent
        
        splits = ["", ""]
        found = False
        marker = False
        
        for child in parent._.children:
            if child != doc[ind:ind+1]:
                if not found:
                    if child.text not in {",", ";", "."}:
                        splits[0] = child
                else:
                    print("ICI ON FAIT DES CHOIX MAINTENANT QUON A TROUVE")
                    if type(splits[0]) == str and splits[0] == "":
                        print("choix A")
                        splits = [child.text]
                        break
                    if child.text == "":
                        print("choix B")
                        splits = [splits[0].text]
                        break
                    print("choix C")
                    print(child, child._.labels)
                    if splits[1] == "":
                        splits[1] = child
                    else:
                        splits[1] = doc[splits[1].start:child.end]
                    #if len(child) == 1 and child[0].tag_ == "CC":
                        #continue
                    #print("avant le truc",splits[0], "---", splits[1])
                    splits, marker = conjunction_split_implicit_coref(splits[0], splits[1])
                    #print("après le truc", splits, marker)
                    break       
            else:
                found = True
        splits = [split.text if type(split) != str else split for split in splits]
        splits = [split for split in splits if split != ""]
        print("parent text", parent.text)
        print("splits de base", splits)
        print("word", word)
        print()
        if not marker and len(splits) > 1:
            #alt_splits = [parent.text.replace(splits[0] + " " + word + " " + splits[1], split) for split in splits]
            #print(alt_splits)
            print(word, index)
            splits = [re.sub(re.escape(splits[0]) + "\W*" + word + "\W*" + re.escape(splits[1]), split, parent.text) for split in splits]
        print("splits !!!!", splits)
        #input(splits)
        
        if word == "nor":
            full_splits = [negate_nor(doc, parent, split, spacy_model) for split in splits if split != ""]
            word = "and"
        else:
            full_splits = [doc[:parent.start].text + " " + split.strip() + " " + doc[parent.end:].text for split in splits if split != ""]
        print("generated splits", full_splits)
        
        if len(parent) > len(max_parent):
            max_parent = parent
            not_s_parent = find_biggest_not_s_parent(doc[ind:ind+1])
            if "VP" in not_s_parent._.labels:
                for par_word in not_s_parent:
                    if par_word.text == "not":
                        if word == "and":
                            word = "or"
                        elif word == "or":
                            word = "and"
            result = [(word, full_splits)]
    #print("FULL SPLITS")
    return result
    
def negate_nor(doc, parent, split, spacy_model):
    negator = Negator(spacy_model, fail_on_unsupported=True)
    try:
        new_split = negator.negate_sentence((doc[:parent.start].text + " " + split.strip() + " " + doc[parent.end:].text).replace("neither ", ""), prefer_contractions=False)
    except RuntimeError:
        new_split = (doc[:parent.start].text + " " + split.strip() + " " + doc[parent.end:].text).replace("neither ", "not ")
    return new_split

def conjunction_split_implicit_coref(left_span, right_span):
    if len(left_span._.labels) == 0 or "S" not in left_span._.labels: #I think this can be changed to just the 2nd condition
        return [left_span.text, right_span.text], False
    
    #cas pbs
        #le right_child est un VP
        #le right child n'a aucun NP en enfant (c'est qu'un grand VP)
        #le right child n'a aucun VP en enfant (c'est qu'une grande S mal identifiée parce qu'il n'y a pas de sujet)

    need_implicit_coref = False

    if "VP" in right_span._.labels:
        need_implicit_coref = True

    right_nps = [child for child in right_span._.children if "NP" in child._.labels]
    right_vps = [child for child in right_span._.children if "VP" in child._.labels]
    if len(right_nps) == 0 or len(right_vps) == 0: #or all(len(right_vp._.children) == 1 for right_vp in right_vps):
        need_implicit_coref = True
    
    if right_span[0].tag_ in {"CC", "IN", "TO"}:
        need_implicit_coref = True

    #print(need_implicit_coref, right_span[0], right_span[0]._.labels)

    if not need_implicit_coref:
        return [left_span.text, right_span.text], False

    subj = [child for child in left_span._.children if "NP" in child._.labels]
    if subj == []:
        return [left_span.text, right_span.text], False

    return[left_span.text, subj[0].text + " " + right_span.text], True

def check_universal_conjunction(span):
    #check a span, if there is an "and" which only has "NP" ancestors up to the original span, then it should be transformed into an "or".
    #This function looks for such an "and" and returns its index if it exists, or an empty list otherwise
    #return a spacy span
    #print("dans univ conjunction")
    start_index = span.start
    for i, word in enumerate(span):
        if word.text == "and":
            current_span = span[word.i-start_index:word.i+1-start_index]
            while current_span != span:
                print(current_span, current_span._.parent, current_span._.parent._.labels[0])
                if "NP" not in current_span._.parent._.labels:
                    print("ciao")
                    break
                current_span = current_span._.parent
            if current_span == span:
                print("ouaiiiis")
                return [word.i]
    return []

def check_universal_negative_conjunction(span):
    #check a span, if there is an "and" which only has "NP" ancestors up to the original span, then it should be transformed into an "or".
    #This function looks for such an "and" and returns its index if it exists, or an empty list otherwise
    #return a spacy span
    #print("dans univ conjunction")
    start_index = span.start
    for i, word in enumerate(span):
        if word.text == "nor":
            current_span = span[word.i-start_index:word.i+1-start_index]
            while current_span != span:
                print(current_span, current_span._.parent, current_span._.parent._.labels[0])
                if "NP" not in current_span._.parent._.labels:
                    print("ciao")
                    break
                current_span = current_span._.parent
            if current_span == span:
                print("ouaiiiis")
                for new_word in word._.parent:
                    if new_word.lower_ == "neither":
                        return [word.i, new_word.i]
                return [word.i]
    return []

    

def find_biggest_NP_parent(span): #find the biggest NP constituency group a span belongs to
    if span._.parent is not None:
        if span._.parent._.labels[0] not in {'VP','S', 'PP', 'ADJP', 'ADVP'}:
            return find_biggest_NP_parent(span._.parent)
        else:
            return span
    else:
        return None
    
def find_biggest_parent(span): #find the biggest constituency group a span belongs to that is not a verbal phrase or the whole sentence
    if span._.parent is not None:
        if span._.parent._.labels[0] not in {'VP','S'}:
            return find_biggest_parent(span._.parent)
        else:
            return span
    else:
        return None

def find_biggest_not_s_parent(span): #find the biggest constituency group a span belongs to that is not the whole sentence
    if span._.parent is not None:
        if not "S" in span._.parent._.labels:
            return find_biggest_not_s_parent(span._.parent)
        else:
            return span
    else:
        if not "S" in span._.labels:
            return span 
        return None

def has_vp_ancestor(span): #returns True if the span has a VP as an ancestor.
    while span._.parent is not None and "VP" not in span._.parent._.labels[0]:
        return has_vp_ancestor(span._.parent)
    if span._.parent is None:
        return False
    else:
        return True

def find_smallest_vp_parent(span):
    while span._.parent is not None and "VP" not in span._.parent._.labels:
        return find_smallest_vp_parent(span._.parent)
    if span._.parent is None:
        return None
    elif "VP" in span._.parent._.labels:
        return span._.parent

def be_sentence(doc):
    for i, word in enumerate(doc):
        if word.lemma_ == "be":
            vp_parent = find_smallest_vp_parent(word)
            if vp_parent is not None:
                for check_word_post in vp_parent:
                    if check_word_post != word and check_word_post.tag_[0] == "V" and find_smallest_vp_parent(check_word_post) == vp_parent: #there is a verb other than "is" which shares the same smallest VP parent, meaning it is used as auxiliary
                        return False
            return True
    return False

def clean_temp():
    for f in os.listdir("temp/edus/"):
        if f.split(".")[-1] == "edus":
            os.remove(os.path.join("temp/edus/", f))
    for f in os.listdir("temp/premises/"):
        if f.split(".")[-1] == "txt":
            os.remove(os.path.join("temp/premises/", f))
    for f in os.listdir("temp/trees/"):
        if f.split(".")[-1] == "tree":
            os.remove(os.path.join("temp/trees/", f))
    if os.path.exists("temp/test.json"):
        os.remove("temp/test.json")

def clean_temp_alt():
    for f in os.listdir("temp/edus_alt/"):
        if f.split(".")[-1] == "edus":
            os.remove(os.path.join("temp/edus_alt/", f))
    for f in os.listdir("temp/premises_alt/"):
        if f.split(".")[-1] == "txt":
            os.remove(os.path.join("temp/premises_alt/", f))
    for f in os.listdir("temp/trees_alt/"):
        if f.split(".")[-1] == "tree":
            os.remove(os.path.join("temp/trees_alt/", f))
    if os.path.exists("temp/test_alt.json"):
        os.remove("temp/test_alt.json")

def flatten_list(l):
    result = []
    for sl in l:
        if type(sl) == list:
            result.extend(flatten_list(sl))
        else:
            result.append(sl)
            
    return result

def find_instance(text, spacy_model):
    with torch.no_grad():
        doc = spacy_model.process(text)
    sent = list(doc.sents)[0]
    for i,word in enumerate(doc):
        print(word.text, word.tag_)
        if word.tag_ in {"NN","NNP","NNS","NNPS"}:
            constituency_ancestor = find_biggest_not_s_parent(word)
            if "VP" in constituency_ancestor._.labels:
                for child in constituency_ancestor:
                    print(child)
                break
            constituency_parent = find_biggest_parent(word)
            if constituency_parent is not None:
                result = constituency_parent.text
                return result
            else:
                break
    return None
def find_instances(text, spacy_model):
    print("FIND INSTANCES EN COURS")
    result = set()
    with torch.no_grad():
        doc = spacy_model.process(text)
    sent = list(doc.sents)[0]
    print(sent._.parse_string)
    for i,word in enumerate(doc):
        if word.tag_ in {"NNP","NNPS", "NN", "NNS"}:
            np_ancestor = find_biggest_NP_parent(word)
            if type(np_ancestor) == type(word): #NP only 1 word long, must be an entity
                if word.tag_ in {"NNP", "NNPS"}:
                    result.add(np_ancestor.text)
            elif np_ancestor is not None:
                """if any([w.tag_.startswith("V") for w in np_ancestor]):
                    continue"""
                the_marker = False
                for np_word in np_ancestor: #we check if the NP is definite. If that is the case, we add it to the result
                    if np_word.lower_ in {"a", "an"}:
                        break
                    elif np_word.lower_ == "the":
                        the_marker = True
                    elif np_word == word: #end of the check of the sequence
                        print(word, word.tag_, the_marker, np_ancestor.text)
                        if word.tag_ in {"NNS", "NNPS", "NN","NNS"} and not the_marker: #if the word is plural but there is no
                            break
                        result.add(np_ancestor.text)
                        break
    instances = set()
    for instance in result: #Just in case an instance appears as part of an other, we only take the biggest available
        if not any([instance in other for other in result if other != instance]):
            instances.add(instance)
    return instances

def find_instances_dev(text, spacy_model):
    result = set()
    with torch.no_grad():
        doc = spacy_model.process(text)
    sent = list(doc.sents)[0]
    print(sent._.parse_string)
    for i,word in enumerate(doc):
        if word.tag_ in {"NN","NNP","NNS","NNPS"}:
            if len(doc) > i+1 and doc[i+1].tag_ in {"NN","NNP","NNS","NNPS"}:
                continue
            constituency_ancestor = find_biggest_not_s_parent(word)
            np_ancestor = find_biggest_NP_parent(word)

            if constituency_ancestor is None:
                continue
            #I think this case disjunction is too complex, and should be just one where we look into the np
            if "VP" in constituency_ancestor._.labels: #the NP is embedded within a VP
                np_ancestor = find_biggest_NP_parent(word)
                bad_instance = False
                the_marker = False
                if type(np_ancestor) != type(word): #several words NP
                    for np_word in np_ancestor: #we check if the NP is indefinite
                        if np_word.lower_ in {"a", "an"}:
                            bad_instance = True
                            break
                        if np_word.lower_ == "the":
                            the_marker = True
                        elif np_word == word:
                            if word.tag_ == "NNS" and not the_marker:
                                bad_instance = True
                            break 
                    if bad_instance: #if the NP is indefinite, we don't want it
                        continue

                smallest_vp_parent = find_smallest_vp_parent(word) 
                for child in smallest_vp_parent._.children: #the children may be either single words (len(labels) == 0) or the ancestor
                    if type(np_ancestor) == type(word): #the NP is a single word, the other children MUST all be the same type 
                        if type(child) != type(np_ancestor):
                            bad_instance = True
                            break
                    else: #the NP is several words long, the other children must either be not the same type of len(labels) == 0
                        if type(child) == type(np_ancestor) and child != np_ancestor and len(child._.labels) > 0:
                            bad_instance = True
                            break
                if not bad_instance:
                    result.add(np_ancestor.text)
            
            else:
                #we have a pure NP
                np_ancestor = find_biggest_NP_parent(word)
                if type(np_ancestor) == type(word): #NP only 1 word long, must be an entity
                    result.add(np_ancestor.text)
                elif np_ancestor is not None:
                    the_marker = False
                    for np_word in np_ancestor: #we check if the NP is definite. If that is the case, we add it to the result
                        if np_word.lower_ in {"a", "an"}:
                            break
                        elif np_word.lower_ == "the":
                            the_marker = True
                        elif np_word == word: #end of the check of the sequence
                            if word.tag_ in {"NNS", "NNPS"} and not the_marker: #if the word is plural but there is no
                                break
                            result.add(np_ancestor.text)
                            break
    instances = set()
    for instance in result: #Just in case an instance appears as part of an other, we only take the biggest available
        if not any([instance in other for other in result if other != instance]):
            instances.add(instance)
    return instances
"""mod = SpacyModel()
r = mod.process_coref("If someone is kind then they are happy.")
print(r)
s = mod.process_coref("I like John and he likes school.")
print(s)"""
#l = ['Bonnie either attends and is very engaged with school events\n', 'and is a student\n', 'who attends the school ,\n', 'or she neither attends and is very engaged with school events nor is a student\n', 'who attends the school .\n']
#t = "Bonnie both is very engaged with school events ."
#print(find_instances(t, mod))
#s = "Bonnie is a young child or teenager who wishes to further Bonnie's academic career and educational opportunities and chaperones high school dances or neither is a young child nor teenager who wishes to further her academic career and educational opportunities ."
#print(get_conjunction_splits(s, [('or', 5), ('and', 14), ('and', 17), ('or', 22), ('nor', 28), ('and', 37)], mod))

def build_instances_disjunction(formulas_list):
    if len(formulas_list) == 1:
        return formulas_list[0]
    else:
        return InstancesOr(formulas_list[0], build_instances_disjunction(formulas_list[1:]))

def in_implies_left(formula, variable, left_formula = None):
    #recursively checks if a certain variable is on the left side of an implication within formula
    if isinstance(formula, Literal):
        if left_formula is not None and formula.char == variable:
            return left_formula
        return False
    elif isinstance(formula, UnaryPredicate):
        if left_formula is not None and formula.name == variable:
            return left_formula
        return False
    elif isinstance(formula, Implies):
        if in_implies_left(formula.left, variable, formula.left) and left_formula is None:
            return formula
        else:
            return in_implies_left(formula.left, variable, formula.left)
    elif isinstance(formula, Not):
        return in_implies_left(formula.inner, variable, left_formula)
    else:
        return in_implies_left(formula.left, variable, left_formula) or in_implies_left(formula.right, variable, left_formula)

def replace_implies_left(formula: Implies, variable):
    #removes variable from the left side of formula, and possibly removes completely the implication if the variable is alone on the left side
    #print(formula, variable)
    if isinstance(formula, Implies):
        if isinstance(formula.left, Literal) and formula.left.char == variable:
            return formula.right
        elif isinstance(formula.left, UnaryPredicate) and formula.left.name == variable:
            return formula.right
        else:
            return Implies(replace_implies_left(formula.left, variable), formula.right)
    elif isinstance(formula, Not):
        return Not(replace_implies_left(formula.inner, variable))
    elif isinstance(formula, Literal):
        return formula
    elif isinstance(formula, UnaryPredicate):
        return formula
    else:
        if isinstance(formula.left, Literal) and formula.left.char == variable:
            return formula.right
        elif isinstance(formula.right, Literal) and formula.right.char == variable:
            return formula.left
        elif isinstance(formula.left, UnaryPredicate) and formula.left.name == variable:
            return formula.right
        elif isinstance(formula.right, UnaryPredicate) and formula.right.name == variable:
            return formula.left
        else:
            if isinstance(formula, And):
                return And(replace_implies_left(formula.left, variable), replace_implies_left(formula.right, variable))
            elif isinstance(formula, Or):
                return Or(replace_implies_left(formula.left, variable), replace_implies_left(formula.right, variable))
            elif isinstance(formula, XOr):
                return XOr(replace_implies_left(formula.left, variable), replace_implies_left(formula.right, variable))

def replace_implies(formula, new_implication, old_implication):
    if isinstance(formula, Implies):
        if formula == old_implication:
            return new_implication
        else:
            return Implies(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))
    elif isinstance(formula, Not):
        return Not(replace_implies(formula.inner, new_implication, old_implication))
    elif isinstance(formula, Literal):
        return formula
    elif isinstance(formula, UnaryPredicate):
        return formula
    elif isinstance(formula, And):
        return And(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))
    elif isinstance(formula, Or):
        return Or(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))
    elif isinstance(formula, XOr):
        return XOr(replace_implies(formula.left, new_implication, old_implication), replace_implies(formula.right, new_implication, old_implication))

def transform_truth(premise, key_truth_quantif):
    valid_candidate = in_implies_left(premise, key_truth_quantif)
    #print("valid candidate", valid_candidate)
    if not valid_candidate:
        return premise
    new_implication = replace_implies_left(valid_candidate, key_truth_quantif)
    #print("new_imp", new_implication)
    return replace_implies(premise, new_implication, valid_candidate)

def tree_transform (rst_tree, dic, first_allowed_letter):
    #Input: the tree object
    #Function: tree_transform
    #Output: Formula object corresponding to the tree
    if rst_tree.children == []:
        if rst_tree.variables == []:
            dic[first_allowed_letter] = rst_tree.text
            new_first_allowed_letter = chr(ord(first_allowed_letter)+1)
            return Literal(first_allowed_letter), dic, new_first_allowed_letter
        else:
            dic[first_allowed_letter] = rst_tree.text
            new_first_allowed_letter = chr(ord(first_allowed_letter)+1)
            return UnaryPredicate(first_allowed_letter, rst_tree.variables[0]), dic, new_first_allowed_letter
    else:
        rel = rel_simplify(rst_tree.data)
        if rel == "forall": 
            return tree_transform(rst_tree.children[0][0], dic, first_allowed_letter)
        elif rel == "imply":
            for (child, nuc) in rst_tree.children:
                if nuc == 0:
                    left, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
                elif nuc == 1:
                    right, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
            return Implies(left, right), dic, first_allowed_letter
        elif rel == "and":
            for i, (child, nuc) in enumerate(rst_tree.children):
                if i == 0:
                    left, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
                elif i == 1:
                    right, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
            return And(left, right), dic, first_allowed_letter
        elif rel == "or":
            for i, (child, nuc) in enumerate(rst_tree.children):
                if i == 0:
                    left, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
                elif i == 1:
                    right, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
            return Or(left, right), dic, first_allowed_letter
        elif rel == "xor":
            for i, (child, nuc) in enumerate(rst_tree.children):
                if i == 0:
                    left, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
                elif i == 1:
                    right, dic, first_allowed_letter = tree_transform(child, dic, first_allowed_letter)
            return XOr(left, right), dic, first_allowed_letter

        elif rel == "not":
           inner, dic, first_allowed_letter = tree_transform(rst_tree.children[0][0], dic, first_allowed_letter)
           return Not(inner), dic, first_allowed_letter
            
    return

def convert_to_logic(tree, dic, first_allowed_letter, spacy_model, character_list, invert=False, neg=False):
    if tree.label() == "ROOT":
        text = " ".join(tree.get_words())
        if neg:
            negator = Negator(spacy_model, fail_on_unsupported=True)
            text = negator.negate_sentence(text, prefer_contractions=False)
        dic[first_allowed_letter] = text
        #new_first_allowed_letter = chr(ord(first_allowed_letter)+1)
        new_first_allowed_letter = character_list[character_list.index(first_allowed_letter)+1]
        if "X " in text or " X" in text:
            return UnaryPredicate(first_allowed_letter, "X"), dic, new_first_allowed_letter
        else:
            return Literal(first_allowed_letter), dic, new_first_allowed_letter
    elif tree.label() == "Attribution":
        #We consider its left child is only one Root.
        prefix = " ".join(tree[0].get_words()[:-1]) + " that" 
        tree[1] = tree[1].prefix_left(prefix)
        return convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
    elif tree.label() == "Punctuation":
        prefix = " ".join(tree[0].get_words()[:-1])
        tree[1] = tree[1].prefix_left(prefix)
        return convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
    else:
        rel = tree.label()
        if rel == "Universal": 
            return convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
        elif rel.lower() in {"implies", "if", "whenever", "when", "once"}:
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            return Implies(left, right), dic, first_allowed_letter
        elif rel.lower() == "or":
            if len(tree) > 2:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(Node("or", *tree[1:]), dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            else:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            if invert:
                return And(left, right), dic, first_allowed_letter
            return Or(left, right), dic, first_allowed_letter
        elif rel == "XOR":
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            return XOr(left, right), dic, first_allowed_letter
        elif rel == "Not":
            inner, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            return inner, dic, first_allowed_letter
        elif rel == "NeitherNor":
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            return And(left, right), dic, first_allowed_letter
        elif rel == "Nor":
            left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, not invert, not neg)
            return And(left, right), dic, first_allowed_letter
        else: #rel = "and"
            if len(tree) > 2:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(Node("and", *tree[1:]), dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            else:
                left, dic, first_allowed_letter = convert_to_logic(tree[0], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
                right, dic, first_allowed_letter = convert_to_logic(tree[1], dic, first_allowed_letter, spacy_model, character_list, invert, neg)
            if invert:
                return Or(left, right), dic, first_allowed_letter
            return And(left, right), dic, first_allowed_letter
    return

def coref_step(step, spacy_model):
    full_text = " \ ".join(step.premises+[step.conclusion])
    resolved_text = spacy_model.process_coref(full_text)
    premises = [prem.strip() for prem in resolved_text.split(" \ ")]
    step.premises = premises[:-1]
    step.conclusion = premises[-1]
    return step

def get_date():
    today = date.today()
    return today.strftime("%m_%d")

def find_most_recent_file(folder, name):
    files = os.listdir(folder)
    matching_files = []
    if name.startswith("None_"):
        name = name[5:]
    if "EntailmentBank" in name:
        name = name.replace("EntailmentBank", "EB")
        name= name.replace("neg_*EB", "EB_neg")
        name= name.replace("hallu_*EB", "EB_hallu")
    print(name)
    for f in files:
        if fnmatch.fnmatch(f, "*"+name+"*"):
            if "ProofWriter" in name and "alt" in name and "alt2" not in name and "alt2" in f:
                continue
            if "ProofWriter" in name and "alt" in name and "alt3" not in name and "alt3" in f:
                continue
            elif "ProofWriter" in name and "alt" not in name and "alt" in f:
                continue
            elif "LLaMa" in name and "LLaMa3" not in name and "LLaMa3" in f:
                continue

            matching_files.append(f)
    print(matching_files)
    #input("HOP")
    matching_files = [m.replace("FOLIO-LLaMa", "FOLIO-06_01-LLaMa") for m in matching_files]
    matching_files = [m.replace("FOLIO-Mistral", "FOLIO-06_01-Mistral") for m in matching_files]
    matching_files = [m.replace("FOLIO-T5", "FOLIO-06_01-T5") for m in matching_files]
    matching_files = [m.replace("FOLIO-Symbolic", "FOLIO-06_01-Symbolic") for m in matching_files]
    matching_files = [m.replace("FOLIO-GPT", "FOLIO-06_01-GPT") for m in matching_files]
    matching_files = [m.replace("ProntoQA-Symbolic", "ProntoQA-06_01-Symbolic") for m in matching_files]
    matching_files = [m.replace("ProntoQA-LLaMa", "FOLIO-06_01-LLaMa") for m in matching_files]
    matching_files = [m.replace("ProntoQA-Mistral", "FOLIO-06_01-Mistral") for m in matching_files]
    matching_files = [m.replace("ProntoQA-T5", "FOLIO-06_01-T5") for m in matching_files]
    matching_files = [m.replace("ProofWriter-LLaMa", "ProofWriter-06_01-LLaMa") for m in matching_files]
    matching_files = [m.replace("ProofWriter_alt-LLaMa", "ProofWriter_alt-06_01-LLaMa") for m in matching_files]
    matching_files = [m.replace("ProofWriter_alt2-LLaMa", "ProofWriter_alt2-06_01-LLaMa") for m in matching_files]
    matching_files = [m.replace("ProofWriter-Symbolic", "ProofWriter-06_01-Symbolic") for m in matching_files]
    matching_files = [m.replace("ProofWriter_alt-Symbolic", "ProofWriter_alt-06_01-Symbolic") for m in matching_files]
    matching_files = [m.replace("ProofWriter_alt2-Symbolic", "ProofWriter_alt2-06_01-Symbolic") for m in matching_files]
    matching_files.sort(reverse=True, key = lambda x: x.replace(".jsonl", ""))
    print(matching_files)
    #input()
    return matching_files[0]

def clear_cache():
    torch.cuda.empty_cache()