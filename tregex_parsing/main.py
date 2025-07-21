import nltk
from nltk import Tree
import copy
from collections import defaultdict

class Node(Tree):
    def __init__(self, label, *children):
        super().__init__(label, children)
        assert(all(type(c)==Node for c in list(self)))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.label() == other.label() and list(self) == list(other)

    def init_indexes(self, indexes=[]):
        self.indexes = indexes
        for i, child in enumerate(list(self)):
            child.init_indexes(indexes+[i])

    def get_node(self, indexes):
        node = self
        for i in indexes:
            node = node[i]
        return node

    def get_children(self, label):
        return [c for c in self if c.label() == label]

    def get_V_children(self):
        return [c for c in self if c.label().startswith("V")]

    def get_N_children(self):
        return [c for c in self if c.label().startswith("N")]

    def get_adj_children(self):
        return [c for c in self if c.label() in {"JJ", "JJR", "JJS", "ADJP"}]

    def get_labels_path(self, target_indexes):
        labels = [node.label()]
        node = self
        for i in target_indexes:
            node = node[i]
            labels.append(node.label())
        return labels

    def replace_subtree(self, subtree, indexes):
        # Traverse up to the parent of the target node
        if indexes:
            result = copy.deepcopy(self)
            parent = result.get_node(indexes[:-1])
            index = indexes[-1]
            parent[index] = copy.deepcopy(subtree)
            return result
        else:
            return copy.deepcopy(subtree)

    def remove_subtrees(self, subtrees):
        result = copy.deepcopy(self)
        deleted_indexes = defaultdict(int)
        for subtree in subtrees:
            if tuple(subtree[:-1]) in deleted_indexes:
                subtree[-1] -= deleted_indexes[tuple(subtree[:-1])]
            parent = result.get_node(subtree[:-1])
            index = subtree[-1]
            del parent[index]
            deleted_indexes[tuple(subtree[:-1])] += 1

            """#Deletion done, now need to prepare cleaning
            if len(parent) == 1 and parent.label() == parent[0].label():
                pparent = subtree[:-2]
                if pparent:
                    pparent_node = result.get_node(pparent)
                else:
                    pparent_node = result
                pparent_node[parent.indexes[-1]] = copy.deepcopy(parent[0])"""

            #Cleaning: remove empty chain
            if len(parent) == 0:
                pparent_index = subtree[:-1]
                while len(result.get_node(pparent_index)) <= 1:
                    if pparent_index == []:
                        return result
                    pparent_index = pparent_index[:-1]
                pparent_node = result.get_node(pparent_index)
                del pparent_node[parent.indexes[len(pparent_index)]]
        return result

    def introduce_subtree(self, subtree, indexes):
        #Introduce a subtree at the given indexes in the tree - indexes is the corresponds to the indexes in the new tree
        result = copy.deepcopy(self)
        result.init_indexes()
        if type(indexes) == list:
            parent = result.get_node(indexes[:-1])
            index = indexes[-1]
        elif indexes == "right":
            parent = result
            while len(parent[-1]) > 0 and len(parent[-1][-1]) > 0:
                parent = parent[-1]
                index = len(parent)
        parent.insert(index, copy.deepcopy(subtree))
        return result

    def __deepcopy__(self, memo):
        """Override deepcopy to ensure subnodes remain of type Node."""
        label = copy.deepcopy(self._label, memo)
        children = [copy.deepcopy(child, memo) for child in self]
        new_node = Node(label, *children)
        if hasattr(self, 'indexes'):
            new_node.indexes = copy.deepcopy(self.indexes, memo)
        return new_node

    def get_rightest_descendant(self, label):
        #Returns the rightest descendant of the node with the given label
        label_children = self.get_children(label)
        if len(label_children) == 0:
            return self

        return label_children[-1].get_rightest_descendant(label)

    def get_words(self):
        words = []
        if len(self) == 0:
            return [self.label()]
        else:
            for c in self:
                words += c.get_words()
            return words

    def remove_following(self, indexes):
        #Remove all the nodes following the node at the given indexes
        result = copy.deepcopy(self)
        for i in range(len(indexes)):
            parent = result.get_node(indexes[:i])
            current_index = indexes[i]
            if current_index+1 < len(parent):
                del parent[current_index+1:]
        return result

    def replace_word(self, original_word, replacement_word):
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.label().lower() == original_word.lower():
                node.set_label(replacement_word)
            queue.extend(c for c in node)
        return self

    def get_leftest_leaf(self):
        if len(self) == 0:
            return self
        return self[0].get_leftest_leaf()

    def clean_duplicates(self):
        #Traverses the tree. If a node has only one child and this child has the same label, we replace the node by its child
        queue = [self]
        while queue:
            node = queue.pop(0)
            if len(node) == 1 and node.label() == node[0].label():
                parent = self.get_node(node.indexes[:-1])
                index = node.indexes[-1]
                parent[index] = node[0]
                self.init_indexes()
            queue.extend(c for c in node)
        return self

    def clean_tree(self):
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.label() == "NML":
                node.set_label("NP")
            queue.extend(c for c in node)
        return self

    def print_parsed(self):
        #If a node is ROOT, print all its words (" ".join(node.get_words()))
        #Otherwise, print the label and indicate children with parentheses
        if self.label() == "ROOT":
            return " ".join(self.get_words())
        else:
            return self.label()+"("+(", ".join(c.print_parsed() for c in self))+")"
            
    def prefix_left(self, prefix):
        #Add a prefix to the left of all ROOT children
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.label() == "ROOT":
                node.insert(0, Node(prefix))
            else:
                queue.extend(c for c in node)
        return self
        
    @classmethod
    def fromstring(cls, tree_str):
        """Create a Node tree from a string representation."""
        nltk_tree = Tree.fromstring(tree_str)
        return cls.convert(nltk_tree)

    @classmethod
    def convert(cls, nltk_tree):
        """Convert an NLTK tree to an instance of this Node class."""
        if isinstance(nltk_tree, Tree):
            label = nltk_tree.label()
            children = [cls.convert(child) for child in nltk_tree]
            return cls(label, *children)  # Use *children to unpack the list for the constructor
        else:
            # Convert terminal nodes (strings) to Node instances
            return cls(nltk_tree)
    
    def convert_to_logic(tree, dic, first_allowed_letter, spacy_model, invert=False, neg=False):
        if self.label() == "ROOT":
            text = " ".join(self.get_words())
            if neg:
                negator = Negator(spacy_model, fail_on_unsupported=True)
                text = negator.negate_sentence(text, prefer_contractions=False)
            dic[first_allowed_letter] = text
            new_first_allowed_letter = chr(ord(first_allowed_letter)+1)
            if "X " in text or " X" in text:
                return UnaryPredicate(first_allowed_letter, "X"), dic, new_first_allowed_letter

            else:
                return Literal(first_allowed_letter), dic, new_first_allowed_letter
        elif self.label() == "Attribution":
            #We consider its left child is only one Root
            prefix = " ".join(self[0].get_words()[:-1]) + " that" 
            self[1] = self[1].prefix_with(prefix)
            return self[1].convert_to_logic(dic, first_allowed_letter, invert, neg)
        else:
            rel = self.label()
            if rel == "Universal": 
                return self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
            elif rel in {"Imply", "If", "if"}:
                left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, not invert, neg)
                right, dic, first_allowed_letter = self[1].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                return Implies(left, right), dic, first_allowed_letter
            elif rel == "or":
                if len(self) > 2:
                    left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                    right, dic, first_allowed_letter = Node("or", *self[1:]).convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                else:
                    left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                    right, dic, first_allowed_letter = self[1].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                if invert:
                    return And(left, right), dic, first_allowed_letter
                return Or(left, right), dic, first_allowed_letter
            elif rel == "XOR":
                left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                right, dic, first_allowed_letter = self[1].convert_to_logic(dic, first_allowed_letterspacy_model, invert, neg)
                return XOr(left, right), dic, first_allowed_letter
            elif rel == "not":
                inner, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, not invert, not neg)
                return inner, dic, first_allowed_letter
            elif rel == "NeitherNor":
                left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, not invert, not neg)
                right, dic, first_allowed_letter = self[1].convert_to_logic(dic, first_allowed_letter, spacy_model, not invert, not neg)
                return And(left, right), dic, first_allowed_letter
            elif rel == "Nor":
                left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                right, dic, first_allowed_letter = self[1].convert_to_logic(dic, first_allowed_letter, spacy_model, not invert, not neg)
                return And(left, right), dic, first_allowed_letter
            else: #rel = "and"
                if len(self) > 2:
                    left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                    right, dic, first_allowed_letter = Node("and", *self[1:]).convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                else:
                    left, dic, first_allowed_letter = self[0].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                    right, dic, first_allowed_letter = self[1].convert_to_logic(dic, first_allowed_letter, spacy_model, invert, neg)
                if invert:
                    return Or(left, right), dic, first_allowed_letter
                return And(left, right), dic, first_allowed_letter
        return

class Condition:
    def __init__(self, operator, tregex):
        self.operator=operator
        self.tregex=tregex
        assert(type(tregex)==Tregex)

    def __str__(self):
        return self.operator+" "+str(self.tregex)

    def fulfilledBy(self, node, variables, originalTree):
        if self.operator=='<':
            return any(self.tregex.matches(c, variables, originalTree) for c in node)
        if self.operator=='!<':
            return not any(self.tregex.matches(c, variables, originalTree) for c in node)
        if self.operator=='<,':
            return any(self.tregex.matches(c, variables, originalTree) for c in node[:1])
        if self.operator=='<-':
            return any(self.tregex.matches(c, variables, originalTree) for c in node[-1:])
        if self.operator=='<:':
            if len(node) == 1:
                return self.tregex.matches(node[0], variables, originalTree)
            return False
        if self.operator=='<<':
            return any(self.tregex.matchesAnyDescendant(c, variables, originalTree) for c in node)
        if self.operator=='<<,':
            return any(self.tregex.matchesAnyLefmostDescendant(c, variables, originalTree) for c in node[:1])
        if self.operator=='<<:':
            if len(node) == 1:
                return self.tregex.matchesAnySingleDescendant(node[0], variables, originalTree)
            return False
        if self.operator=='!<<':
            return not any(self.tregex.matchesAnyDescendant(c, variables, originalTree) for c in node)
        if self.operator.startswith('<+'):
            intermediateTregex = Tregex(self.operator[2:], None)
            return any(self.tregex.matchesChainDescendant(c, variables, originalTree, intermediateTregex) for c in node)
        if self.operator=='$':
            return self.tregex.matchesAnySister(node, variables, originalTree)
        if self.operator=='!$':
            return not self.tregex.matchesAnySister(node, variables, originalTree)
        if self.operator=='$..':
            return self.tregex.matchesAnyRSister(node, variables, originalTree)
        if self.operator=='!$..':
            return not self.tregex.matchesAnyRSister(node, variables, originalTree)
        if self.operator=='$,,':
            return self.tregex.matchesAnyLSister(node, variables, originalTree)
        if self.operator=='!$,,':
            return not self.tregex.matchesAnyLSister(node, variables, originalTree)
        if self.operator=='$+':
            return self.tregex.matchesImmRSister(node, variables, originalTree)
        if self.operator=="!$+":
            return not self.tregex.matchesImmRSister(node, variables, originalTree)
        if self.operator=='>':
            return self.tregex.matchesParent(node, variables, originalTree)
        if self.operator=='!>':
            return not self.tregex.matchesParent(node, variables, originalTree)
        if self.operator=='>>':
            return self.tregex.matchesAnyAscendant(node, variables, originalTree)
        if self.operator=='!>>':
            return not self.tregex.matchesAnyAscendant(node, variables, originalTree)
        if self.operator=='>>,1':
            return self.tregex.matchesAnyAscendantLeft(node, variables, originalTree)
        if self.operator=='!>>,1':
            return not self.tregex.matchesAnyAscendantLeft(node, variables, originalTree)
        if self.operator=='text=':
            return self.tregex.matchesText(node, variables, originalTree)
         
    def __str__(self):
        return self.operator+str(self.tregex)

class Tregex:
    def __init__(self, label, variable, *conditions):
        self.label=label
        self.conditions=conditions
        self.variable=variable        
        assert(all(type(c)==Condition for c in self.conditions))

    def __str__(self):
        return (self.label if self.label else "NoLabel")+('='+self.variable if self.variable else '')+'('+(' and '.join(str(c) for c in self.conditions))+')'  

    def matches(self, node, variables, originalTree):
        newVariables={}
        result=(self.label is None or any(node.label().lower()==lab.lower() for lab in self.label.split("|"))) and all(c.fulfilledBy(node, newVariables, originalTree) for c in self.conditions)
        if result:
            variables.update(newVariables)
            if self.variable:
                variables[self.variable]=node.indexes
        return result

    def matchesAnyDescendant(self, node, variables, originalTree):
        return self.matches(node, variables, originalTree) or any(self.matchesAnyDescendant(c,variables, originalTree) for c in node)

    def matchesAnyLefmostDescendant(self, node, variables, originalTree):
        return self.matches(node, variables, originalTree) or any(self.matchesAnyLefmostDescendant(c, variables, originalTree) for c in node[:1])

    def matchesAnySingleDescendant(self, node, variables, originalTree):
        return self.matches(node, variables, originalTree) or (len(node) == 1 and self.matchesAnySingleDescendant(node[0], variables, originalTree))

    def matchesAnySister(self, node, variables, originalTree):
        sisterNodes = [c for c in originalTree.get_node(node.indexes[:-1]) if c.indexes[-1] != node.indexes[-1]]
        return any(self.matches(c, variables, originalTree) for c in sisterNodes)

    def matchesAnyRSister(self, node, variables, originalTree):
        if node.indexes != [] and len(originalTree.get_node(node.indexes[:-1])) > node.indexes[-1]+1:
            RSisterNodes = originalTree.get_node(node.indexes[:-1])[node.indexes[-1]+1:]
            return any(self.matches(c, variables, originalTree) for c in RSisterNodes)
        return False

    def matchesAnyLSister(self, node, variables, originalTree):
        if node.indexes != [] and node.indexes[-1] > 0:
            LSisterNodes = originalTree.get_node(node.indexes[:-1])[:node.indexes[-1]]
            return any(self.matches(c, variables, originalTree) for c in LSisterNodes)
        return False

    def matchesImmRSister(self, node, variables, originalTree):
        if node.indexes != [] and len(originalTree.get_node(node.indexes[:-1])) > node.indexes[-1]+1:
            RSisterNode = originalTree.get_node(node.indexes[:-1])[node.indexes[-1]+1]
            return self.matches(RSisterNode, variables, originalTree)
        return False
            
    def matchesChainDescendant(self, node, variables, originalTree, intermediateLabel):
        return self.matches(node, variables, originalTree) or (intermediateLabel.matches(node, variables, originalTree) and any(self.matchesChainDescendant(c, variables, originalTree, intermediateLabel) for c in node))

    def matchesParent(self, node, variables, originalTree):
        return self.matches(originalTree.get_node(node.indexes[:-1]), variables, originalTree)

    def matchesAnyAscendant(self, node, variables, originalTree):
        return self.matches(node, variables, originalTree) or (node.indexes != [] and self.matchesAnyAscendant(originalTree.get_node(node.indexes[:-1]), variables, originalTree))

    def matchesAnyAscendantLeft(self, node, variables, originalTree):
        if node.indexes == []:
            return False
        current_parent = originalTree.get_node(node.indexes[:-1])
        if self.matches(current_parent, variables, originalTree) and node.indexes[-1] == 0:
            return True
        else:
            return self.matchesAnyAscendantLeft(current_parent, variables, originalTree)

    def matchesText(self, node, variables, originalTree):
        if " ".join(node.get_words()).lower() == self.label.lower():
            return len(node)!= 1 or (len(node) == 1 and len(node[0]) == 0) #If the node only has one child, we want the 2nd smallest node to match that text
        return False
        
def findMatches(tregex, node, originalTree):
    variables={}
    if tregex.matches(node, variables, originalTree):
        yield variables
    else:
        for c in node:
            yield from findMatches(tregex, c, originalTree)

def findMatch(tregex, node, originalTree):
    variables = {}
    if tregex.matches(node, variables, originalTree):
        pass
    yield variables
