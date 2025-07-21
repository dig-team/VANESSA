from reasoner.logic import Literal, And, Or, Implies, Not
from reasoner.utils import InterestLink, Sequent, Interest


class Rule():
    def __init__(self):
        pass

class IntroductionAnd(Rule):
    def __init__(self):
        self.symbol = "∧i"
        pass

    def backward_match(self, interest: Interest):
        return isinstance(interest.conclusion, And)

    def backward_apply(self, interest: Interest):
        new_interests = [Interest(interest.hypotheses, interest.conclusion.left, interest.depth+1),
        Interest(interest.hypotheses, interest.conclusion.right, interest.depth+1)]
        return new_interests, [InterestLink(new_interests, interest, self.symbol)]

class EliminationAndLeft(Rule):
    def __init__(self):
        self.symbol = "∧e"
        pass

    def forward_match(self, _, fact: Sequent):
        matches = [fact] if isinstance(fact.conclusion, And) else []
        return matches

    def forward_apply(self, sequent: Sequent):
        return Sequent(sequent.hypotheses, sequent.conclusion.left, sequent.depth+1)

class EliminationAndRight(Rule):
    def __init__(self):
        self.symbol = "∧e"
        pass

    def forward_match(self, _, fact: Sequent):
        matches = [fact] if isinstance(fact.conclusion, And) else []
        return matches

    def forward_apply(self, sequent: Sequent):
        return Sequent(sequent.hypotheses, sequent.conclusion.right, sequent.depth+1)

class IntroductionOr(Rule):
    def __init__(self):
        self.symbol = "∨i"
        pass

    def backward_match(self, interest: Interest):
        return isinstance(interest.conclusion, Or)

    def backward_apply(self, interest: Interest):
        new_interests = [Interest(interest.hypotheses, interest.conclusion.left, interest.depth+1),
        Interest(interest.hypotheses, interest.conclusion.right, interest.depth+1)]
        return new_interests, [InterestLink([new_interest], interest, self.symbol) for new_interest in new_interests]

class EliminationOr(Rule):
    def __init__(self):
        self.symbol = "∨e"
        pass

    def backward_match(self, interest: Interest):
        return True
    
    def back_forw_match(self, facts, interest: Interest):
        return facts.search_type_back_forw(Or, interest)
    
    def back_forw_apply(self, interest: Interest, match: Sequent):
        new_hypotheses = interest.hypotheses | match.hypotheses
        new_sub_interest_left = Interest(new_hypotheses | {match.conclusion.left}, interest.conclusion, interest.depth+1)
        new_sub_interest_right = Interest(new_hypotheses | {match.conclusion.right}, interest.conclusion, interest.depth+1)
        new_sub_interest_link = InterestLink([new_sub_interest_left, new_sub_interest_right, match], interest, self.symbol)
        new_facts_hypo_left = Sequent(new_hypotheses | {match.conclusion.left}, match.conclusion.left, 0)
        new_facts_hypo_right = Sequent(new_hypotheses | {match.conclusion.right}, match.conclusion.right, 0)

        if new_sub_interest_left == interest or new_sub_interest_right == interest:
            return [], None, []
        return [new_sub_interest_left, new_sub_interest_right], new_sub_interest_link, [new_facts_hypo_left, new_facts_hypo_right]
    
class EliminationOr2(Rule):
    def __init__(self):
        self.symbol = "∨e2"

    def forward_match(self, facts, fact: Sequent):
        if isinstance(fact.conclusion, Or):
            left_neg = fact.conclusion.left.inner if isinstance(fact.conclusion.left, Not) else Not(fact.conclusion.left)
            right_neg = fact.conclusion.right.inner if isinstance(fact.conclusion.right, Not) else Not(fact.conclusion.right)
            match_candidates = facts.search(left_neg) + facts.search(right_neg)
            return [(fact, match) for match in match_candidates]
        else:
            match_candidates = facts.search_neg_disjunctions(fact)
            return([(match, fact) for match in match_candidates])
        
    def forward_apply(self, match: list):
        disjunction, fact = match
        new_depth = max(disjunction.depth, fact.depth) + 1
        new_hypotheses = disjunction.hypotheses | fact.hypotheses
        fact_neg = fact.conclusion.inner if isinstance(fact.conclusion, Not) else Not(fact.conclusion)
        if fact_neg == disjunction.conclusion.left:
            return Sequent(new_hypotheses, disjunction.conclusion.right, new_depth)
        elif fact_neg == disjunction.conclusion.right:
            return Sequent(new_hypotheses, disjunction.conclusion.left, new_depth)

class EliminationImplies(Rule):
    def __init__(self):
        self.symbol = "→e"
        pass

    def forward_match(self, facts, fact: Sequent):
        if isinstance(fact.conclusion, Implies):
            match_candidates = facts.search(fact.conclusion.left)
            return [(fact, match) for match in match_candidates]
        else:
            match_candidates = facts.search_implications_left(fact)
            return [(match, fact) for match in match_candidates]

    def forward_apply(self, match: list):
        implication, fact = match
        new_depth = max(implication.depth, fact.depth) + 1
        new_hypotheses = implication.hypotheses | fact.hypotheses
        return Sequent(new_hypotheses, implication.conclusion.right, new_depth)
    
    def backward_match(self, interest: Interest):
        return not isinstance(interest.conclusion, Implies)
    
    def back_forw_match(self, facts, interest: Interest):
        implications = facts.search_type_back_forw(Implies, interest)
        result = [imp for imp in implications if imp.conclusion.right == interest.conclusion]
        return result
    
    def back_forw_apply(self, interest: Interest, match: Sequent):
        new_hypotheses = interest.hypotheses | match.hypotheses
        new_sub_interest = Interest(new_hypotheses, match.conclusion.left, interest.depth+1)
        new_sub_interest_link = InterestLink([match, new_sub_interest], interest, self.symbol)
        return [new_sub_interest], new_sub_interest_link, []
        
class IntroductionImplies(Rule):
    def __init__(self):
        self.symbol = "→i"
        pass

    def backward_match(self, sequent: Sequent):
        return isinstance(sequent.conclusion, Implies)
    
    def back_forw_match(self, facts, interest: Interest):
        return [interest]

    def back_forw_apply(self, interest: Interest, match: Interest):
        new_sub_interest = Interest(interest.hypotheses | {interest.conclusion.left}, interest.conclusion.right, interest.depth+1)
        new_block_interest = Interest(interest.hypotheses, interest.conclusion.right, interest.depth+1)
        new_sub_interest_link = InterestLink([new_sub_interest], interest, self.symbol,[new_block_interest])
        new_facts_hypo = Sequent(interest.hypotheses | {interest.conclusion.left}, interest.conclusion.left, 0)
        return [new_sub_interest], new_sub_interest_link, [new_facts_hypo]

class TransferNot(Rule):
    def __init__(self):
        self.symbol = "¬"
        pass

    def forward_match(self, fact: Sequent):
        return not isinstance(fact.conclusion, Implies)
    
    def forw_back_match(self, interests, fact: Sequent):
        return interests.search_neg_interests(fact)
    
    def forw_back_apply(self, fact: Sequent, match: Interest):
        neg_prem = match.conclusion.inner if isinstance(match.conclusion, Not) else Not(match.conclusion)
        new_hypotheses = fact.hypotheses | {neg_prem}
        neg_conclusion = fact.conclusion.inner if isinstance(fact.conclusion, Not) else Not(fact.conclusion)
        new_interest = Interest(new_hypotheses, neg_conclusion, match.depth+1)
        new_block_interests = [Interest(new_hypotheses-{neg_prem}, neg_conclusion, match.depth+1), Interest(new_hypotheses-{neg_prem}, fact.conclusion, match.depth+1)]
        new_interest_link = InterestLink([fact, new_interest], match, self.symbol, new_block_interests)
        new_hypo_fact = Sequent(new_hypotheses, neg_prem, 0)

        return [new_interest], new_interest_link, [new_hypo_fact]