import networkx as nx
import matplotlib.pyplot as plt
from webapp.postprocessing import clean_proof_graph
from bs4 import BeautifulSoup
from networkx.readwrite import json_graph
from pyvis.network import Network
import json

def build_graph(proof_lines, correspondance_dict, entailments, log_premises):
    graph = nx.DiGraph()
    proof_lines = clean_proof_graph(proof_lines, correspondance_dict, entailments, log_premises)
    for line in proof_lines:
        if line["line_type"] == "ent-rem":
            continue
        if "ent" not in line:
            line["ent"] = ""
        graph.add_node(line["line_number"], text=line["full_text"], type=line["line_type"], dict=line["line_dict"], ent=line["ent"])
        for or_line in line["original_lines"]:
            graph.add_edge(or_line, line["line_number"])
    ancestors = set([len(nx.ancestors(graph, node)) for node in graph.nodes])
    sorted_ancestors = sorted(ancestors)

    # Créer un dictionnaire qui associe chaque élément à son indice
    index_dict = {element: index for index, element in enumerate(sorted_ancestors)}
    for node in graph.nodes:
        descendants = nx.descendants(graph, node)
        graph.nodes[node]['descendants_nb'] = len(descendants)
        ancestors = nx.ancestors(graph, node)
        graph.nodes[node]['ancestors_nb'] = index_dict[len(ancestors)]


    # Créer un objet Pyvis Network
    net = Network(directed=True)
    net.from_nx(graph)


    # Personnaliser les propriétés des nœuds
    for node in net.nodes:
        # Utiliser l'attribut 'full_text' pour afficher sur le nœud
        full_text = graph.nodes[node['id']].get('text', '')
        full_text = full_text.replace(" → ", "\n→ ")
        full_text = full_text.replace(" ∧ ", "\n∧ ")
        full_text = full_text.replace(" ∨ L", "\n∨ L")
        node['label'] = full_text  # Afficher le texte sur le nœud
        node['size'] = 50  # Taille ajustée pour un affichage lisible
        #node['shape'] = "circle"
        # Personnaliser l'apparence du texte dans le nœud
        node['font'] = {
            'size': 40,  # Taille du texte
            'color': 'black',  # Couleur du texte
            'face': 'arial',  # Police du texte
            'background': 'white',  # Fond du texte
            'align': 'center'  # Centrer le texte dans le nœud
        }
        node['level'] = graph.nodes[node['id']].get('ancestors_nb', '')

    # Personnaliser les propriétés des arêtes (facultatif)
    for edge in net.edges:
        edge['color'] = 'green'
        edge['width'] = 5

    #Change color for entailment edges
    for edge in net.edges:
        if graph.nodes[edge['to']]['type'] == "ent":
            edge['color'] = 'DimGray'
            edge['dashed'] = True

    net.set_options(json.dumps({
            "layout": {
                "hierarchical": {
                    "enabled": True,  # Activer la disposition hiérarchique
                    "levelSeparation": 400,  # Séparation entre les niveaux
                    "nodeSpacing": 300,  # Espacement entre les nœuds
                    "treeSpacing": 1000,  # Espacement entre les sous-graphes
                    "direction": 'LR',  # Direction: 'LR' pour de gauche à droite, 'UD' pour de haut en bas,
                    "sortMethod": "directed",
                }
            },
            "physics": {
                "enabled": False  # Désactive la simulation physique
            }}))

    graph_html = net.generate_html()

    # Use BeautifulSoup to parse the graph HTML and extract <head> and <body>
    soup = BeautifulSoup(graph_html, 'html.parser')

    # Extract the <head> and <body> sections
    graph_head = str(soup.head)  # Get the entire <head> section as string
    graph_body = str(soup.body)  # Get the entire <body> section as string


    return graph_head, graph_body