from reasoner.rules import *
from reasoner.logic import *
from reasoner.utils import *
from func_timeout import func_timeout, FunctionTimedOut

#This implementation follows Pollock's Algorithm (https://johnpollock.us/ftp/OSCAR-web-page/PAPERS/Natural-Deduction.pdf)
#L'idée est de faire de la déduction bi-directionnelle: on utilise les conclusions désirées pour remonter jusqu'aux prémisses et les prémisses pour faire des conclusions

class Proof():
    def __init__(self, hyps, goal, subproofs=None):
        self.forward_rules = [EliminationAndLeft(), EliminationAndRight(), EliminationImplies(), EliminationOr2()]
        self.backward_rules = [IntroductionAnd(), IntroductionOr()]
        self.forw_back_rules = [TransferNot()]
        self.back_forw_rules = [EliminationOr(), IntroductionImplies(), EliminationImplies()]

        self.goal = Interest({}, goal, 0)
        self.hyps = hyps
        self.facts = SequentSet(hyps)
        self.active_interests = InterestSet()
        self.interest_links = InterestLinkSet()

        self.active_interests.add(self.goal)
        
        self.queue = PriorityQueue()
        self.queue.add(self.goal, 1)
        #print("hyphyp", hyps)
        for hyp in hyps:
            self.queue.add(Sequent({}, hyp, 0), 2)

        self.actions = [{"premises": [], "conclusion": Sequent({}, hyp), "action": "ax"} for hyp in hyps]
        
        if subproofs is None:
            self.subproofs = {}
        else:
            self.subproofs = subproofs

    def __str__(self):
        return str(self.hyps) + " ⊢? " + str(self.goal.conclusion)

    def __hash__(self):
        return hash((self.goal.hypotheses, self.goal.conclusion))
    
    def __eq__(self, other):
        return self.goal == other.goal and self.hyps == other.hyps

    def add_fact(self, fact, premises, rule_symbol):
        if fact not in self.facts:
            self.facts.add(fact)
            if isinstance(premises, tuple):
                premises = list(premises)
            else:
                premises = [premises]

            self.actions.append({"premises": premises, "conclusion": fact, "action": rule_symbol})
            self.queue.add(fact, 2) #2 is the priority for facts

    def discharge_interest_link(self, interest_link):
        new_fact = interest_link.generate_fact()

        #print("discharging", interest_link.rule_symbol, interest_link, self.facts, "block", interest_link.block_premises)
        if interest_link.block_premises != []:
            all_proven = True
            for block_premise in interest_link.block_premises:
                new_proof = Proof(self.hyps | block_premise.hypotheses, block_premise.conclusion, self.subproofs)
                if new_proof == self:
                    all_proven = False
                    continue #avoid infinite loop
                if new_proof in self.subproofs:
                    subproof_proven = self.subproofs[new_proof]
                else:
                    subproof_proven = new_proof.prove()
                    #print("subproof finie", subproof_proven)
                    '''proof_dict = get_proof_dict(new_proof.actions, new_proof)
                    proof_lines = get_proof_tree(proof_dict, (new_proof.goal.hypotheses - new_proof.hyps, new_proof.goal.conclusion))
                    for line in proof_lines:
                        print(line)'''
                    #input()
                    self.subproofs[new_proof] = subproof_proven
                    a = len(self.subproofs)
                    self.subproofs.update(new_proof.subproofs)
                    #print("'''''''''''")
                    #self.subproofs | new_proof.subproofs
                    #print("in", self)
                    #print(new_proof)
                    #input()

                if subproof_proven != True:
                    all_proven = False
                    break
            
            """print(interest_link)
            print("done with blocks")
            print("all proven?",all_proven)"""
            #input()
            if all_proven:
                return
        self.queue.add(new_fact, 2)
        self.facts.add(new_fact)
        self.actions.append({"premises": interest_link.discharging_facts, "conclusion": new_fact, "action": interest_link.rule_symbol})
        
    
    def one_step_forward(self, fact):
        new_facts = []
        for rule in self.forward_rules:
            matches = rule.forward_match(self.facts, fact)
            for match in matches:
                new_fact = rule.forward_apply(match)
                if not new_fact in self.facts:
                    new_facts.append((new_fact, match, rule.symbol))
        return new_facts

    def one_step_backward(self, interest):
        if interest.is_discharged(self.facts):
            return [],[]
        
        new_interests, new_interest_links = [], []
        for rule in self.backward_rules:
            #print("RULE:",rule, "INTEREST:",interest)
            if rule.backward_match(interest):
                new_interest, new_interest_link = rule.backward_apply(interest)
                new_interests.extend(new_interest)
                new_interest_links.extend(new_interest_link)
                #print("NEW INTEREST LINKS:",new_interest_link)
        #input()
        return new_interests, new_interest_links
    
    def one_step_forw_back(self, fact):
        new_interests, new_interest_links = [], []
        for rule in self.forw_back_rules:
            if rule.forward_match(fact):
                forw_back_matches = rule.forw_back_match(self.active_interests, fact)
                for forw_back_match in forw_back_matches:
                    new_interest, new_interest_link, new_facts_hypo = rule.forw_back_apply(fact, forw_back_match)
                    for new_fact_hypo in new_facts_hypo:
                        if not new_fact_hypo in self.facts:
                            self.facts.add(new_fact_hypo)
                            self.queue.add(new_fact_hypo, 2)
                            self.actions.append({"premises": [], "conclusion": new_fact_hypo, "action": "ax"})

                    new_interests.extend(new_interest)
                    new_interest_links.append(new_interest_link)

        return new_interests, new_interest_links

    def one_step_back_forw(self, interest):
        if interest.is_discharged(self.facts):
            return [], []
        
        new_interests, new_interest_links = [], []
        for rule in self.back_forw_rules:
            if rule.backward_match(interest):
                back_forw_matches = rule.back_forw_match(self.facts, interest)
                for back_forw_match in back_forw_matches:
                    new_interest, new_interest_link, new_facts_hypo = rule.back_forw_apply(interest, back_forw_match)
                    for new_fact_hypo in new_facts_hypo:
                        if not new_fact_hypo in self.facts:
                            self.facts.add(new_fact_hypo)
                            self.queue.add(new_fact_hypo, 2)
                            self.actions.append({"premises": [], "conclusion": new_fact_hypo, "action": "ax"})

                    new_interests.extend(new_interest)
                    if new_interest_link is not None:
                        new_interest_links.append(new_interest_link)

        return new_interests, new_interest_links

    def contains_contradiction(self):
        contrad_proof = ContradictionCheckProof(self.hyps)
        output = contrad_proof.prove()
        #if output == True:
            #contrad_proof.print_proof_tree()
        return output

    def print_proof_tree(self):
        proof_dict = get_proof_dict(self.actions)
        proof_lines = get_proof_tree(proof_dict, (self.goal.hypotheses, self.goal.conclusion))
        print("Proof:")
        for line in proof_lines:
            print(line)
        return proof_lines                

    def verify(self):
        #print("IN VERIFY")
        #print(self.hyps, "⊢", self.goal.conclusion)
        try:
            contrads = False
            #contrads = func_timeout(20, self.contains_contradiction)
        except FunctionTimedOut:
            contrads = False
        if contrads == True:
            raise ValueError("The set of hypotheses is contradictory")
        try:
            result = func_timeout(60, self.prove)
        except FunctionTimedOut:
            result = "Uncertain"
        if result == True:
            proof_lines = self.print_proof_tree()
            return True, proof_lines
        
        contrad_proof = Proof(self.hyps, self.goal.conclusion.negate())
        contrad_result = False
        #print("Checking contradiction", contrad_proof.goal.conclusion, "from", contrad_proof.hyps)
        try:
            contrad_result = func_timeout(60, contrad_proof.prove)
        except FunctionTimedOut:
            contrad_result = "Uncertain"
        if contrad_result == True:
            proof_lines = contrad_proof.print_proof_tree()
            return False, proof_lines
        return "Uncertain", []
    
    def prove(self):
        """print("Proving...")
        print("Hypotheses:", self.hyps)
        print("Goal:", self.goal.conclusion)
        print("-----")"""
        while not self.queue.is_empty():
            if self.facts.discharges_interest(self.goal): #peut etre remplacé par self.goal not in active_interests?
                #print("Done", self.facts)
                return True
            #print(len(self.queue))
            current, priority = self.queue.pop()
            """print("facts:", len(self.facts))
            print("interests:", len(self.active_interests))
            print("queue", len(self.queue))
            print(priority)
            print("---")"""
            """print("---")
            print("facts:", self.facts)
            print("Examining:",current, priority)
            print("Links:",self.interest_links)
            print("Interests:",self.active_interests)"""
            #input()
            #print("-------")
            #print("Examining:",current, priority, "----", len(self.queue))
            """if current in examined and priority in {1,2}:
                print("????")
            examined.add(current)
            input()"""
            queue_before = len(self.queue)

            if priority == 2:
                assert type(current) == Sequent
                matching_interest_links = self.interest_links.search(current)
                for interest_link in matching_interest_links:
                    if not interest_link.discharged_depth and interest_link.discharge_depth(self.facts):
                        self.discharge_interest_link(interest_link)
                
                self.active_interests.remove_discharged_interests(current)

                new_facts = self.one_step_forward(current)
                for new_fact, new_premises, rule_symbol in new_facts:  
                    self.add_fact(new_fact, new_premises, rule_symbol)

                if current.is_implies() and new_facts == []:
                    new_interest = Interest(current.hypotheses, current.conclusion.left, 0)
                    self.active_interests.add(new_interest)
                    self.queue.add(new_interest, 1) #add as a simple "backward" interest


                self.queue.add(current, 4) 

            else:
                assert priority == 1 or priority == 4 or priority == 3
                if priority == 1:
                    assert type(current) == Interest
                    if current not in self.active_interests:
                        continue
                    new_interests, new_interests_link = self.one_step_backward(current)
                elif priority == 3:
                    assert type(current) == Interest
                    if current not in self.active_interests:
                        continue
                    new_interests, new_interests_link = self.one_step_back_forw(current)
                else:
                    assert priority == 4
                    assert type(current) == Sequent
                    new_interests, new_interests_link = self.one_step_forw_back(current)

                for new_interest_link in new_interests_link:
                    if new_interest_link.discharge_depth(self.facts):
                        self.discharge_interest_link(new_interest_link)
                    else:
                        self.interest_links.add(new_interest_link)
                
                for interest in new_interests:
                    if not self.facts.discharges_interest(interest):
                        self.active_interests.add(interest)
                        self.queue.add(interest, 1) #add as a simple "backward" interest
                
                if priority == 1:
                    self.queue.add(current, 3) #add as a "back-forw" interest, in case the "backward" interest is not discharged

            
            #queue_after = len(self.queue)
            #if queue_after - queue_before > 50:
            """print("Gap queue:", queue_after - queue_before, priority, queue_after, queue_before, len(self.facts))
            if queue_after - queue_before > 1000:
                print(current.hypotheses - self.hyps, "⊢", current.conclusion)"""
        return "Uncertain"

class ContradictionCheckProof(Proof):
    def __init__(self, hyps):
        super().__init__(hyps, Literal("False"))
        #print(len(self.queue))
        for hyp in hyps:
            interest, interest_link = self.add_contrad_interest(Sequent(hyps, hyp, 0))
            self.active_interests.add(interest)
            self.interest_links.add(interest_link)
            self.queue.add(interest, 1)
        #print(len(self.queue))
        #print("Contradiction check proof initialized")

    def add_contrad_interest(self, fact):
        if fact.hypotheses == self.hyps:
            neg_fact = fact.conclusion.negate()
            interest = Interest({}, neg_fact, 0)
            interest_link = InterestLink([fact, interest], Sequent({}, Literal("False")), "¬")
            return interest, interest_link
        return None
    
    def add_fact(self, fact, premises, rule_symbol):
        self.facts.add(fact)
        if isinstance(premises, tuple):
            premises = list(premises)
        else:
            premises = [premises]

        self.actions.append({"premises": premises, "conclusion": fact, "action": rule_symbol})

        self.queue.add(fact, 2)
        contrad = self.add_contrad_interest(fact)
        if contrad is not None:
            int, int_link = contrad
            self.active_interests.add(int)
            self.interest_links.add(int_link)
            self.queue.add(int, 1)

    def discharge_interest_link(self, interest_link):
        super().discharge_interest_link(interest_link)
        new_fact = interest_link.generate_fact()
        contrad = self.add_contrad_interest(new_fact)
        if contrad is not None:
            int, int_link = contrad
            self.active_interests.add(int)
            self.interest_links.add(int_link)
            self.queue.add(int, 1)

class ForwardProof(Proof):
    def __init__(self, hyps, goal, subproofs=None):
        super().__init__(hyps, goal, subproofs)
        self.forw_back_rules = []

    def discharge_interest_link(self, interest_link):
        new_fact = interest_link.generate_fact()
        
        if interest_link.block_premises != []:
            #print(interest_link.rule_symbol)
            #print("???")
            #input()
            all_proven = True
            for block_premise in interest_link.block_premises:
                new_proof = ForwardProof(block_premise.hypotheses, block_premise.conclusion, self.subproofs)
                if new_proof in self.subproofs:
                    #print("J'ai déjà regardé ça")
                    subproof_proven = self.subproofs[new_proof]
                else:
                    subproof_proven = new_proof.prove()
                    #print("subproof finie", subproof_proven)
                    '''proof_dict = get_proof_dict(new_proof.actions, new_proof)
                    proof_lines = get_proof_tree(proof_dict, (new_proof.goal.hypotheses - new_proof.hyps, new_proof.goal.conclusion))
                    for line in proof_lines:
                        print(line)'''
                    #input()
                    self.subproofs[new_proof] = subproof_proven
                    a = len(self.subproofs)
                    self.subproofs.update(new_proof.subproofs)
                    #print("'''''''''''")
                    #self.subproofs | new_proof.subproofs
                    #print("in", self)
                    #print(new_proof)
                    #input()

                if subproof_proven != True:
                    all_proven = False
                    break
            
            """print(interest_link)
            print("done with blocks")
            print("all proven?",all_proven)"""
            #input()
            if all_proven:
                return