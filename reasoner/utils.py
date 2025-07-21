from reasoner.logic import Implies, Formula, Literal, Not, Or
from collections import defaultdict

class Sequent:
    def __init__(self, hypotheses:set, conclusion, depth=0):
        self.hypotheses = frozenset(hypotheses)
        self.conclusion = conclusion
        self.depth = depth

    def __repr__(self) -> str:
        return f"{list(self.hypotheses)} ⊢ {self.conclusion} ({self.depth})"
    
    def __eq__(self, other):
        return self.hypotheses == other.hypotheses and self.conclusion == other.conclusion

    def set_depth(self, depth):
        self.depth = depth

    def sequent_imp(self, other):
        return isinstance(other, Sequent) and self.conclusion == other.conclusion and self.hypotheses.issubset(other.hypotheses)

    def is_implies(self):
        return isinstance(self.conclusion, Implies)
    
class Interest(Sequent):
    def __init__(self, hypotheses, conclusion, depth=0):
        super().__init__(hypotheses, conclusion, depth)
        #Here depth is more like an inverse depth, but it has no impact

    def is_discharged(self, facts):
        return self in facts

class SequentSet:
    def __init__(self, elements = set()):
        self.dict = {}
        self.depths = {}
        for e in elements:
            self.dict[e] = {frozenset()}
            self.depths[(e, frozenset())] = 0

    def __contains__(self, element):
        #Is more than strict containment, is more "logical containment"
        #Checks if element can be immediately deduced from the set (present with the same or less hypotheses)
        if element.conclusion in self.dict:
            for hypotheses in self.dict[element.conclusion]:
                if hypotheses <= element.hypotheses:
                    return True
        return False
    
    def add(self, element):
        if element.conclusion not in self.dict:
            self.dict[element.conclusion] = {element.hypotheses}
            self.depths[(element.conclusion, element.hypotheses)] = element.depth
        else:
            if element.hypotheses in self.dict[element.conclusion]:
                self.depths[(element.conclusion, element.hypotheses)] = min(self.depths[(element.conclusion, element.hypotheses)], element.depth)
            else:
                weaker_hypotheses_set = set()
                for hypotheses in self.dict[element.conclusion]:
                    if element.hypotheses < hypotheses:
                        weaker_hypotheses_set.add(hypotheses)
                self.dict[element.conclusion] = self.dict[element.conclusion] - weaker_hypotheses_set
                self.dict[element.conclusion].add(element.hypotheses)
                self.depths[(element.conclusion, element.hypotheses)] = element.depth
                for hypo in weaker_hypotheses_set:
                    del self.depths[(element.conclusion, hypo)]
    
    def __repr__(self) -> str:
        return str(self.dict)
    
    def __len__(self):
        return sum([len(self.dict[truc]) for truc in self.dict])
    
    def len_alt(self):
        return (sum([len(self.dict[truc]) for truc in self.dict])/len(self.dict), len(self.dict), len(self.depths))
    
    def get_discharging_fact(self, interest):
        #Finds the most general fact (with the least hypotheses) that discharges the interest
        most_general_hypotheses = interest.hypotheses
        for hypotheses in self.dict[interest.conclusion]:
            if hypotheses <= interest.hypotheses and len(hypotheses) < len(most_general_hypotheses):
                most_general_hypotheses = hypotheses
            elif hypotheses <= interest.hypotheses and len(hypotheses) == len(most_general_hypotheses):
                if self.depths[(interest.conclusion, hypotheses)] < self.depths[(interest.conclusion, most_general_hypotheses)]:
                    most_general_hypotheses = hypotheses
        return Sequent(most_general_hypotheses, interest.conclusion, self.depths[(interest.conclusion, most_general_hypotheses)])
    
    def get_depth(self, fact: Sequent):
        if (fact.conclusion, fact.hypotheses) in self.depths:
            return self.depths[(fact.conclusion, fact.hypotheses)]
        elif fact.conclusion in self.dict:
            min_depth = float("inf")
            for hypotheses in self.dict[fact.conclusion]:
                if hypotheses <= fact.hypotheses:
                    min_depth = min(min_depth, self.depths[(fact.conclusion, hypotheses)])
            return min_depth
    
    def discharges_interest(self, interest):
        return interest in self

    def search_type_back_forw(self, target_type: type, interest: Interest):
        #searches for sequents that are a certain type and have hypotheses that are a subset of the interest's hypotheses
        results = []
        target_hypotheses = interest.hypotheses
        for conclusion in self.dict:
            if isinstance(conclusion, target_type):
                for hypotheses in self.dict[conclusion]:
                    if hypotheses <= target_hypotheses:
                        depth = self.depths[(conclusion, hypotheses)]
                        results.append(Sequent(hypotheses, conclusion, depth))
        return results
    
    def search(self, target_conclusion):
        #returns all facts with the same conclusion as the target 
        results = []
        if target_conclusion in self.dict:
            for hypotheses in self.dict[target_conclusion]:
                depth = self.depths[(target_conclusion, hypotheses)]
                results.append(Sequent(hypotheses, target_conclusion, depth))
        return results
    
    def search_implications_left(self, target):
        results = []
        for conclusion in self.dict:
            if isinstance(conclusion, Implies) and conclusion.left == target.conclusion:
                for hypotheses in self.dict[conclusion]:
                    depth = self.depths[(conclusion, hypotheses)]
                    results.append(Sequent(hypotheses, conclusion, depth))

        return results

    def search_neg_disjunctions(self, target: Sequent):
        results = []
        neg_target = target.conclusion.inner if isinstance(target.conclusion, Not) else Not(target.conclusion)
        for conclusion in self.dict:
            if isinstance(conclusion, Or) and (conclusion.left == neg_target or conclusion.right == neg_target):
                for hypotheses in self.dict[conclusion]:
                    if hypotheses <= target.hypotheses or target.hypotheses <= hypotheses:
                        depth = self.depths[(conclusion, hypotheses)]
                        results.append(Sequent(hypotheses, conclusion, depth))
        return results

class InterestSet(SequentSet):
    def __contains__(self, element: Interest):
        #Different from the SequentSet containment
        #Just checks if the interest is present in the set?
        #Or something broader? Like if (A,B) is present then (A) is present?
        #This seems to be logical: si on s'intéresse déjà avec A et B en hypothèses, alors on s'intéresse au même truc avec juste A en hypothèse?
        #Bah oui mais bof en fait on s'en fout un peu
        return (element.hypotheses, element.conclusion) in self.depths
    
    def add(self, interest: Interest):
        if interest.conclusion in self.dict:
            self.dict[interest.conclusion].add(interest.hypotheses)
        else:
            self.dict[interest.conclusion] = {interest.hypotheses}

        if (interest.hypotheses, interest.conclusion) not in self.depths:
            self.depths[(interest.hypotheses, interest.conclusion)] = interest.depth
        else:
            self.depths[(interest.hypotheses, interest.conclusion)] = min(self.depths[(interest.hypotheses, interest.conclusion)], interest.depth)

    def remove_discharged_interests(self, target_fact):
        results = []
        hypotheses_to_remove = set()
        if target_fact.conclusion in self.dict:
            for hypotheses in self.dict[target_fact.conclusion]:
                if target_fact.hypotheses <= hypotheses:
                    hypotheses_to_remove.add(hypotheses)
                    depth = self.depths[(hypotheses, target_fact.conclusion)]
                    results.append(Interest(hypotheses, target_fact.conclusion, depth))
            
            self.dict[target_fact.conclusion] = self.dict[target_fact.conclusion] - hypotheses_to_remove
            for hypo in hypotheses_to_remove:
                del self.depths[(hypo, target_fact.conclusion)]
        return results


    def search_neg_interests(self, fact: Sequent):
        results = []
        for conclusion in self.dict:
            if isinstance(conclusion, Implies):
                continue
            for hypotheses in self.dict[conclusion]:
                neg_prem = conclusion.inner if isinstance(conclusion, Not) else Not(conclusion)
                if neg_prem not in hypotheses and (hypotheses == fact.hypotheses or fact.hypotheses == hypotheses | {neg_prem}):
                    depth = self.depths[(hypotheses, conclusion)]
                    results.append(Interest(hypotheses, conclusion, depth))
        return results

class InterestLink():
    def __init__(self, premises: list, conclusion: Interest, rule_symbol, block_premises = None):
        self.premises = premises
        self.conclusion = conclusion
        self.discharged_depth = 0
        self.rule_symbol = rule_symbol
        if block_premises is None:
            block_premises = []
        self.block_premises = block_premises
        self.discharging_facts = []

    def __repr__(self) -> str:
        return f"{self.premises} '->' {self.conclusion} - {self.discharged_depth} - {self.rule_symbol}" 

    def discharge_depth(self, facts):
        if Sequent(self.conclusion.hypotheses, self.conclusion.conclusion) in facts:
            self.discharged_depth = facts.get_depth(Sequent(self.conclusion.hypotheses, self.conclusion.conclusion))
            return 0
        for premise in self.premises:
            if not premise in facts:
                return 0
        self.discharging_facts = self.get_discharging_facts(facts)
        self.discharged_depth = max([facts.get_depth(discharging_fact) for discharging_fact in self.discharging_facts]) + 1

        return self.discharged_depth
    
    def get_discharging_facts(self, facts):
        discharging_facts = []
        for premise in self.premises:
            discharging_facts.append(facts.get_discharging_fact(premise))
        return discharging_facts

    def generate_fact(self):
        return Sequent(self.conclusion.hypotheses, self.conclusion.conclusion, self.discharged_depth)
    
        
class InterestLinkSet:
    def __init__(self, elements = None):
        if elements is None:
            elements = set()
        self.set = elements
        self.dict = {}
        for e in elements:
            for premise in e.premises:
                hypotheses, conclusion = premise.hypotheses, premise.conclusion
            if conclusion not in self.dict:
                self.dict[conclusion] = {hypotheses: {e}}
            else:
                if hypotheses not in self.dict[conclusion]:
                    self.dict[conclusion][hypotheses] = {e}
                else:
                    self.dict[conclusion][hypotheses].add(e)

    def __repr__(self) -> str:
        return str(self.set)
    
    def __contains__(self, element):
        return element in self.set

    def add(self, element):
        if element not in self.set:
            self.set.add(element)

            for premise in element.premises:
                hypotheses, conclusion = premise.hypotheses, premise.conclusion
                if conclusion not in self.dict:
                    self.dict[conclusion] = {hypotheses: {element}}
                else:
                    if hypotheses not in self.dict[conclusion]:
                        self.dict[conclusion][hypotheses] = {element}
                    else:
                        self.dict[conclusion][hypotheses].add(element)

    def search(self, target_interest):
        results = set()
        if target_interest.conclusion in self.dict:
            for hypotheses in self.dict[target_interest.conclusion]:
                if target_interest.hypotheses <= hypotheses:
                    results = results.union(self.dict[target_interest.conclusion][hypotheses])
        return results

class PriorityQueue:
    def __init__(self):
        self.interests = defaultdict(list)
        self.facts = defaultdict(list)
        self.back_forw_interests = defaultdict(list)
        self.forw_back_facts = defaultdict(list)

        self.already_seen = set()

    def __len__(self):
        return sum([len(self.interests[depth]) for depth in self.interests]) + sum([len(self.facts[depth]) for depth in self.facts]) + sum([len(self.back_forw_interests[depth]) for depth in self.back_forw_interests]) + sum([len(self.forw_back_facts[depth]) for depth in self.forw_back_facts])
    
    def is_empty(self):
        return len(self) == 0
    
    def add(self, element, priority):
        if (element.hypotheses, element.conclusion, priority) in self.already_seen:
            return
        if priority == 1:
            dict = self.interests
        elif priority == 2:
            dict = self.facts
        elif priority == 3:
            dict = self.back_forw_interests
        elif priority == 4:
            dict = self.forw_back_facts
        
        dict[element.depth].append(element)
        self.already_seen.add((element.hypotheses, element.conclusion, priority))

    def pop(self):
        if sum([len(self.interests[depth]) for depth in self.interests]) != 0:
            for depth in sorted(self.interests.keys()):
                if self.interests[depth] != []:
                    return self.interests[depth].pop(0), 1
        elif sum([len(self.facts[depth]) for depth in self.facts]) != 0:
            for depth in sorted(self.facts.keys()):
                if self.facts[depth] != []:
                    return self.facts[depth].pop(0), 2
        elif sum([len(self.back_forw_interests[depth]) for depth in self.back_forw_interests]) != 0:
            for depth in sorted(self.back_forw_interests.keys()):
                if self.back_forw_interests[depth] != []:
                    return self.back_forw_interests[depth].pop(0), 3
        elif sum([len(self.forw_back_facts[depth]) for depth in self.forw_back_facts]) != 0:
            for depth in sorted(self.forw_back_facts.keys()):
                if self.forw_back_facts[depth] != []:
                    return self.forw_back_facts[depth].pop(0), 4
        return None

        

def get_proof_dict(actions):
    proof_dict = {}
    for element in actions:
        premises = element["premises"]
        conclusion = element["conclusion"]
        rule_symbol = element["action"]
        proof_dict[(conclusion.hypotheses, conclusion.conclusion)] = ([(premise.hypotheses, premise.conclusion) for premise in premises], rule_symbol)

    return proof_dict

def get_proof_tree(proof_dict, target):
    seen = set()
    lines = []
    queue = [target]
    while queue != []:
        current = queue.pop(0)
        premises, rule = proof_dict[current]
        if current not in seen:
            for premise in premises:
                queue.insert(0,premise)

        if (premises, current, rule) in lines:
            lines = [line for line in lines if line != (premises, current, rule)]
        lines.append((premises, current, rule))
        seen.add(current)
    lines = lines[::-1]
    new_lines = []
    for i, line in enumerate(lines):
        premises, (hypotheses, conclusion), rule = line
        lines_nbs = []
        for j in range(i):
            for premise in premises:
                if premise == lines[j][1]:
                    lines_nbs.append(j+1)
        new_lines.append((i+1, set(hypotheses), conclusion, rule, lines_nbs))
    lines_to_keep = set([len(new_lines)])
    for line in enumerate(new_lines):
        lines_nb = line[1][4]
        for line_nb in lines_nb:
            lines_to_keep.add(line_nb)
    new_lines = [line for line in new_lines if line[0] in lines_to_keep]
    lines_dict = {line[0]: i+1 for i, line in enumerate(new_lines)}
    final_lines= []
    for line in new_lines:
        string = str(lines_dict[line[0]]) + ". " + ", ".join((str(e) for e in line[1])) + " ⊢ " + str(line[2]) + " (" + line[3]
        if line[4] != []:
            string += " - " + ", ".join((str(lines_dict[e]) for e in line[4]))
        string += ")"
        final_lines.append(string)

    return final_lines

def flatten_formula(l):
    flattened = []
    if isinstance(l, list):
        for item in l:
            if isinstance(item, list):
                flattened.extend(flatten_formula(item))
            else:
                flattened.append(item)
        return flattened
    return [l]