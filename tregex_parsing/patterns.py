from tregex_parsing.main import Node, Condition, Tregex, findMatches, findMatch
from lemminflect import getLemma, getInflection, getAllLemmas
import copy

def CC_processing(tree, spacyModel):
    oldTree = Node("r")
    CC = "ROOT <<: (S < (S ?$.. CC & $.. S))"
    CCTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", "S", Condition("<", Tregex("S", None, Condition("$..", Tregex("S", None)))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CCTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            mainS = root.get_node(match["S"][len(match["Root"]):])
            SChildren = mainS.get_children("S")
            minSChildrenIndex, maxSChildrenIndex = SChildren[0].indexes[-1], SChildren[-1].indexes[-1]
            CC = mainS.get_children("CC")

            root_label = "And"
            if CC:
                root_label = [c.get_words() for c in CC]
                if any("either" in words for words in root_label):
                    eitherIndex = [i for i, words in enumerate(root_label) if "either" in words]
                    root_label = "XOR"
                    root = root.remove_subtrees([CC[i].indexes for i in eitherIndex])
                    root.init_indexes()
                    mainS = root.get_node(match["S"][len(match["Root"]):])
                    SChildren = mainS.get_children("S")
                    minSChildrenIndex, maxSChildrenIndex = SChildren[0].indexes[-1], SChildren[-1].indexes[-1]

                elif any("or" in words for words in root_label):
                    root_label = "Or"
                    eitherTregex = Tregex(None, "either", Condition("<<:", Tregex("either", None)))
                    for m in findMatches(eitherTregex, SChildren[0], root):
                        root = root.remove_subtrees([m["either"]])
                        root.init_indexes()
                        root_label = "XOR"
                        mainS = root.get_node(match["S"][len(match["Root"]):])
                        SChildren = mainS.get_children("S")

                elif any("nor" in words for words in root_label):
                    if any("neither" in words for words in root_label):
                        root_label = "NeitherNor"
                        neitherTregex = Tregex(None, "neither", Condition("<<:", Tregex("neither", None)))
                        for m in findMatches(neitherTregex, mainS, root):
                            root = root.remove_subtrees([m["neither"]])
                            root.init_indexes()
                            mainS = root.get_node(match["S"][len(match["Root"]):])
                            SChildren = mainS.get_children("S")
                    else:
                        root_label = "Nor"
                else:
                    root_label = " ".join([" ".join(words) for words in root_label])
                    root_label = " ".join(sorted(set(root_label.split()), key=root_label.split().index)) #eliminates duplicates
                        
            new_roots = []
            for child in SChildren:
                newMainS = Node("S", *(c for c in mainS if c.indexes[-1]<minSChildrenIndex or c.indexes[-1]>maxSChildrenIndex))
                for i, c in enumerate(child):
                    newMainS = newMainS.introduce_subtree(c, [minSChildrenIndex+i])
                newRoot = root.replace_subtree(newMainS, match["S"][len(match["Root"]):])
                words = " ".join(newRoot.get_words())
                newRoot = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))
                new_roots.append(newRoot)
            new_root = Node(root_label, *new_roots)
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def PunctSplit_processing(tree):
    oldTree = Node("r")
    PunctSplit_Tregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", "S", Condition("<<", Tregex(":", "Punc", Condition("$+", Tregex("S", "NewS")))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(PunctSplit_Tregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            
            left = root.remove_subtrees([[match["NewS"][len(match["Root"]):]]])
            if left.get_words()[-1] != ".":
                S_node = root.get_node(match["NewS"][len(match["Root"]):])
                left = left.introduce_subtree(Node(".", Node(".")), match["S"][len(match["Root"]):]+[len(S_node)])

            NewS = root.get_node(match["NewS"][len(match["Root"]):])
            if NewS.get_words()[-1] != ".":
                NewS = NewS.introduce_subtree(Node(".", Node(".")), [len(NewS)])
            right = Node("ROOT", NewS)
            new_root = Node("Punctuation", left, right)
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def NonResPrepClause_processing(tree):
    oldTree = Node("r")
    NonResPrepClause = "ROOT <<: (S << (NP <, NP & < (/,/ $+ (SBAR <, (WHPP $+ S & <, IN & < − WHNP) & ?$+ /,/))))"
    NonResPrepClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", None, Condition("<,", Tregex("NP", "RelNP")), Condition("<", Tregex(",", "Comma", Condition("$+", Tregex("SBAR", None, Condition("<,", Tregex("WHPP", None, Condition("$+", Tregex("S", "RelS")), Condition("<,", Tregex("IN", "RelPrep")), Condition("<-", Tregex("WHNP", None)))))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(NonResPrepClauseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            RelNP = tree.get_node(match["RelNP"])
            RelS = tree.get_node(match["RelS"])
            RelPrep = tree.get_node(match["RelPrep"])

            parentNP = tree.get_node(match["RelNP"][:-1])
            commaIndex = match["Comma"][-1]
            new_left = root.remove_subtrees([c.indexes[len(root.indexes):] for c in parentNP[commaIndex:]])

            complementary_right = Node("PP", RelPrep, RelNP)
            new_right = RelS.introduce_subtree(complementary_right, [-1, len(RelS[-1])])
            new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(RelS)])
            
            new_root = Node("And", new_left, Node("ROOT", new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def NonResWhereClause_processing(tree):
    oldTree = Node("r")
    NonResWhereClause = "ROOT <<: (S << (/.*/ < (NP|PP $+ (/,/ $+ (SBAR <, (WHADVP $+ S & <<: WRB) & ?$+ /,/)))))"
    #Takes a sentence with a relative clause starting with where and transforms it into two sentences
    NonResWhereClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex(None, None, Condition("<", Tregex("NP|PP", "RelP", Condition("$+", Tregex(",", None, Condition("$+", Tregex("SBAR", None, Condition("<,", Tregex("WHADVP", None, Condition("$+", Tregex("S", "RelS")), Condition("<<:", Tregex("WRB", None)))))))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(NonResWhereClauseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            RelP = tree.get_node(match["RelP"]) #The NP or PP that is related
            RelS = tree.get_node(match["RelS"]) #The relative clause (after the where)

            parentP = tree.get_node(match["RelP"][:-1]) #get the parent of the related NP or PP
            PIndex = match["RelP"][-1]
            new_left = root.remove_subtrees([c.indexes[len(root.indexes):] for c in parentP[PIndex+1:]]) #Remove every right-sister of the related NP or PP (removes comma and the relative clause, but keeps eventual other elements that come afterwards in the sentence) 

            if RelP.label() == "NP": #If the related NP is a NP, we will write "in NP" in the new sentence
                complementary_right = Node("PP", Node("IN", Node("in")), RelP)
            elif RelP.label() == "PP":
                complementary_right = RelP

            #Insert the related NP/PP into the relative clause (and add a period at the end)
            new_right = RelS.introduce_subtree(complementary_right, "right") #instead of "right" (introduce at the rightest descendant), we should use the rightest VP (as is the case in Whom processing)
            new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(RelS)])
            
            #Create the new root (And) and replace the old one
            #Each newly created tree has a ROOT node
            new_root = Node("And", new_left, Node("ROOT", new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def NonResWhomClause_processing(tree):
    oldTree = Node("r")
    PPTregex = Tregex("VP", None, Condition("<+VP", Tregex("PP", "pp_rel")))
    NonResWhomClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", None, Condition("<,", Tregex("NP", "RelNP")), Condition("<", Tregex(",", "Comma", Condition("$+", Tregex("SBAR", None, Condition("<,", Tregex("WHNP", None, Condition("$+", Tregex("S", "RelS", Condition("<,", Tregex("NP", None)), Condition("<-", Tregex("VP", "vp_rel")))), Condition("<<:", Tregex("WP", None, Condition("<:", Tregex("whom", None)))))))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(NonResWhomClauseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            RelNP = tree.get_node(match["RelNP"])
            RelS = tree.get_node(match["RelS"])
            relVP = tree.get_node(match["vp_rel"])

            pp_rel = None
            for PP in findMatch(PPTregex, relVP, relVP):
                if PP != {}:    
                    pp_rel = tree.get_node(PP["pp_rel"])
                break 

            parentP = tree.get_node(match["RelNP"][:-1])
            commaIndex = match["Comma"][-1]
            new_left = root.remove_subtrees([c.indexes[len(root.indexes):] for c in parentP[commaIndex:]])

            if pp_rel is not None:
                new_indexes = pp_rel.indexes[len(RelS.indexes):-1] + [pp_rel.indexes[-1]]
                new_right = RelS.introduce_subtree(RelNP, new_indexes)
            else:
                rightestVP = relVP.get_rightest_descendant("VP")
                new_indexes = rightestVP.indexes[len(RelS.indexes):] + [1]
                new_right = RelS.introduce_subtree(RelNP, new_indexes)
            
            new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(RelS)])
            
            new_root = Node("And", new_left, Node("ROOT", new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def NonResWhoseClause_processing(tree):
    oldTree = Node("r")
    NonResWhoseClause = "ROOT <<: (S << (NP < ( NP $+ (/,/ $+ (SBAR <, (WHNP $+ S & <, (/WP \\$/ $+ /.*/=subject)) & ?$+ /,/)))))"
    NonResWhoseClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", None, Condition("<", Tregex("NP", "RelNP", Condition("$+", Tregex(",", None, Condition("$+", Tregex("SBAR", None, Condition("<,", Tregex("WHNP", "Whose", Condition("$+", Tregex("S", "RelS")), Condition("<,", Tregex("WP$", None)))))))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(NonResWhoseClauseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            RelNP = tree.get_node(match["RelNP"])
            RelS = tree.get_node(match["RelS"])
            Whose = tree.get_node(match["Whose"])

            parentNP = tree.get_node(match["RelNP"][:-1])
            NPIndex = tree.get_node(match["RelNP"]).indexes[-1]
            new_left = root.remove_subtrees([c.indexes[len(root.indexes):] for c in parentNP[NPIndex+1:]])

            WhoseComplement = Whose[1:]
            newSubject = Node("NP", RelNP.introduce_subtree(Node("POS", Node("'s")), [len(RelNP)]))
            for c in WhoseComplement:
                newSubject = newSubject.introduce_subtree(c, [len(newSubject)])
            new_right = RelS.introduce_subtree(newSubject, [0])
            new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])
            
            new_root = Node("And", new_left, Node("ROOT", new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def NonResWhoWhichClause_processing(tree):
    oldTree = Node("r")
    NonResWhoWhichClause = "ROOT <<: (S << (NP <, NP & < (/,/ $+ (SBAR <, (WHNP $+ S & <<: WP|WDT) & ?$+ /,/))))"
    NonResWhoWhichClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", None, Condition("<,", Tregex("NP", "RelNP")), Condition("<", Tregex(",", "Comma", Condition("$+", Tregex("SBAR", None, Condition("<,", Tregex("WHNP", None, Condition("$+", Tregex("S", "RelS")), Condition("<<:", Tregex("WP|WDT", None)))))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(NonResWhoWhichClauseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            RelNP = tree.get_node(match["RelNP"])
            RelS = tree.get_node(match["RelS"])

            parentNP = tree.get_node(match["RelNP"][:-1])
            commaIndex = match["Comma"][-1]
            new_left = root.remove_subtrees([c.indexes[len(root.indexes):] for c in parentNP[commaIndex:]])

            new_right = RelS.introduce_subtree(RelNP, [0])
            new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])
            
            new_root = Node("And", new_left, Node("ROOT", new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def NonResAppo_processing(tree):
    oldTree = Node("r")
    NonResAppo = "ROOT <<: (S < VP & << ( NP $+ (/,/ $+ (NP !$ CC & ?$+ /,/))))"
    NonResAppoTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("VP", None)), Condition("<<", Tregex("NP", "NP1", Condition("$+", Tregex(",", "Comma", Condition("$+", Tregex("NP", "NP2", Condition("!$", Tregex("CC", None)))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(NonResAppoTregex, tree, tree):
            root = tree.get_node(match["Root"])
            NP1 = tree.get_node(match["NP1"])
            NP2 = tree.get_node(match["NP2"])

            #Should run NER and use the closest to a person as the main NP
            mainNP, appoNP = NP1, NP2
            
            #We first insert the appositive NP in the main NP in stead of the original NP1, and remove everything after it (NP1 index now corresponds to that of the main NP)
            new_left = root.introduce_subtree(mainNP, NP1.indexes[len(root.indexes):])
            new_left = new_left.remove_subtrees([c.indexes[:-1]+[i] for i,c in enumerate(new_left.get_node(match["NP1"][len(root.indexes):-1])) if i > NP1.indexes[-1]])

            #We create a new sentence
            #Should do something to detect plurality, so that we can use "are" instead of "is"
            new_right = Node("S", mainNP, Node("VP", Node("VBZ", Node("is")), appoNP))
            new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])
            
            new_root = Node("And", new_left, Node("ROOT", new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def PreAdvClause_processing(tree):
    PreAdvClause = "ROOT <<: (S < (SBAR < (S < (NP $.. VP)) $.. (NP $.. VP)))"
    PreAdvClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("SBAR", "SBAR", Condition("<", Tregex("S", "S", Condition("<+S", Tregex("NP", None, Condition("$..", Tregex("VP", None)))))), Condition("$..", Tregex("NP", None, Condition("$..", Tregex("VP", None)))))))))
    for match in findMatches(PreAdvClauseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])
        SBARIndexes = match["SBAR"]

        SBAR_left_words = tree.get_node(SBARIndexes)
        SBAR_left_words = SBAR_left_words.remove_subtrees([S.indexes[-1:]])
        SBAR_left_words = SBAR_left_words.get_words()
        
        new_node_label = " ".join(SBAR_left_words)

        commaNextToSBAR = tree.get_node(SBARIndexes[:-1])[SBARIndexes[-1]+1].label() == ","
        if commaNextToSBAR:
            if new_node_label.lower() == "if" and "then" in tree.get_node(SBARIndexes[:-1])[SBARIndexes[-1]+2].get_words():
                new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):-1]+[i] for i in range(SBARIndexes[-1]+3)])
            else:
                new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):-1]+[i] for i in range(SBARIndexes[-1]+2)])
        else:
            if new_node_label.lower() == "if" and "then" in tree.get_node(SBARIndexes[:-1])[SBARIndexes[-1]+1].get_words():
                new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):-1]+[i] for i in range(SBARIndexes[-1]+2)])
                #new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):], SBARIndexes[len(root.indexes):-1]+[SBARIndexes[-1]+1]])
            else:
                new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):-1]+[i] for i in range(SBARIndexes[-1]+1)])
        #We create a new sentence
        new_right = S.introduce_subtree(Node(".", Node(".")), [len(S)])

        new_root = Node(new_node_label, Node("ROOT", new_right), new_left)
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def PrePurpAdvClauseIOT_processing(tree):
    """
    I don't know if we should do this split
    Should be an implication? An "and"? 
    Weird to be an implication, but at the same time is more than a simple "and"
    """
    PrePurpAdvClauseIOT = "ROOT <<: (S < (SBAR < (S <<, (VP <<, /(T|t)o/)) $.. (NP $.. VP)))"
    PrePurpAdvClauseIOTTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("SBAR", "SBAR", Condition("<", Tregex("S", "S", Condition("<<,", Tregex("VP", None, Condition("<<,", Tregex("TO", "TO")))))), Condition("$..", Tregex("NP", None, Condition("$..", Tregex("VP", None)))))))))
    for match in findMatches(PrePurpAdvClauseIOTTregex, tree, tree):
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])
        SBARIndexes = match["SBAR"]
        TOIndexes = match["TO"]

        #We get the part after the SBAR (main clause)
        commaNextToSBAR = tree.get_node(SBARIndexes[:-1])[SBARIndexes[-1]+1].label() == ","
        if commaNextToSBAR:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):], SBARIndexes[len(root.indexes):-1]+[SBARIndexes[-1]+1]])
        else:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):]])

        #Take the inside of the S Sentence without the TO
        new_right = S.remove_subtrees([TOIndexes[len(S.indexes):]])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(S)])

        new_root = Node("Purpose", Node("ROOT", new_right), new_left)
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def PrePurpAdvClauseT_processing(tree):
    """
    Same problem as earlier
    I don't know if we should do this split
    Should be an implication? An "and"? 
    Weird to be an implication, but at the same time is more than a simple "and"
    """
    PrePurpAdvClauseT = "ROOT <<: (S < (S <<, (VP <<, /(T|t)o/) $.. (NP $.. VP)))"
    PrePurpAdvClauseTTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("S", "S", Condition("<<,", Tregex("VP", None, Condition("<<,", Tregex("TO", "TO")))))), Condition("$..", Tregex("NP", None, Condition("$..", Tregex("VP", None)))))))
    for match in findMatches(PrePurpAdvClauseTTregex, tree, tree):
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])
        TOIndexes = match["TO"]
        SIndexes = match["S"]

        #We get the part after the SBAR (main clause)
        commaNextToS = tree.get_node(SIndexes[:-1])[SIndexes[-1]+1].label() == ","
        if commaNextToS:
            new_left = root.remove_subtrees([SIndexes[len(root.indexes):], SIndexes[len(root.indexes):-1]+[SIndexes[-1]+1]])
        else:
            new_left = root.remove_subtrees([SIndexes[len(root.indexes):]])

        #Take the inside of the S Sentence without the TO
        new_right = S.remove_subtrees([TOIndexes[len(S.indexes):]])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(S)])

        new_root = Node("Purpose", Node("ROOT", new_right), new_left)
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def PrePartPhrase_processing1(tree):
    '''
    Again, I'm not sure if we should do this split'''
    PrePartPhrase = "ROOT <<: (S <: (VP <<, VBG|VBN ) $.. (NP $.. VP))"
    PrePartPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("VP", "VP", Condition("<<,", Tregex("VBG|VBN", None)))), Condition("$..", Tregex("NP", "NP", Condition("$..", Tregex("VP", None)))))))
    for match in findMatches(PrePartPhraseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        VP = tree.get_node(match["VP"])
        SIndexes = match["VP"][:-1]
        #We get the part after the Participle Phrase (main clause)
        commaNextToS = tree.get_node(SIndexes[:-1])[SIndexes[-1]+1].label() == ","
        if commaNextToS:
            new_left = root.remove_subtrees([SIndexes[len(root.indexes):], SIndexes[len(root.indexes):-1]+[SIndexes[-1]+1]])
        else:
            new_left = root.remove_subtrees([SIndexes[len(root.indexes):]])

        #Append the VP with the NP as subject, and create a S node
        new_VP = Node("VP", Node("VBZ", Node("is")), VP)
        new_right = Node("S", NP, new_VP, Node(".", Node(".")))

        new_root = Node("Subordinate", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def PrePartPhrase_processing2(tree):
    PrePartPhrase = "ROOT <<: (S < (PP|ADVP <+PP|ADVP (S <: (VP <<, VBG|VBN )) $.. (NP $.. VP)))"
    PrePartPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("PP|ADVP", "PartP", Condition("<+PP|ADVP", Tregex("S", "S", Condition("<", Tregex("VP", "VP", Condition("<<,", Tregex("VBG|VBN", None)))))), Condition("$..", Tregex("NP", "NP", Condition("$..", Tregex("VP", None)))))))))
    for match in findMatches(PrePartPhraseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        VP = tree.get_node(match["VP"])
        PIndexes = match["PartP"]
        #We get the part after the Participle Phrase (main clause)
        commaNextToP = tree.get_node(PIndexes[:-1])[PIndexes[-1]+1].label() == ","
        if commaNextToP:
            new_left = root.remove_subtrees([PIndexes[len(root.indexes):], PIndexes[len(root.indexes):-1]+[PIndexes[-1]+1]])
        else:
            new_left = root.remove_subtrees([PIndexes[len(root.indexes):]])

        #Append the VP with the NP as subject, and create a S node
        new_VP = Node("VP", Node("VBZ", Node("is")), VP)
        new_label = tree.get_node(match["PartP"]).get_children("IN")[0][0].label()
        new_right = Node(new_label, NP, new_VP, Node(".", Node(".")))

        new_root = Node("Subordinate", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree
    
def CoordVerbPhrase_processing(tree, spacyModel):
    oldTree = Node("r")
    CoordVerbPhrase = "ROOT <<: (S < ( NP $.. (VP <+(VP) (VP > VP ?$.. CC & $.. VP))))"
    #CoordVerbPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None, Condition("<+VP", Tregex("VP", None, Condition("$..", Tregex("VP", None)), Condition(">", Tregex("VP", "mainVP")))))))))))
    CoordVerbPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("VP", None, Condition("<+VP", Tregex("VP", None, Condition("$..", Tregex("VP", None)), Condition(">", Tregex("VP", "mainVP")))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordVerbPhraseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
            VPChildren = mainVP.get_children("VP")
            minVPChildrenIndex, maxVPChildrenIndex = VPChildren[0].indexes[-1], VPChildren[-1].indexes[-1]
            
            CC = mainVP.get_children("CC")
            root_label = "And"

            if CC:
                root_label = [c.get_words() for c in CC]
                if any("either" in words for words in root_label):
                    eitherIndex = [i for i, words in enumerate(root_label) if "either" in words]
                    root_label = "XOR"
                    root = root.remove_subtrees([CC[i].indexes for i in eitherIndex])
                    root.init_indexes()
                    mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
                    VPChildren = mainVP.get_children("VP")
                    minVPChildrenIndex, maxVPChildrenIndex = VPChildren[0].indexes[-1], VPChildren[-1].indexes[-1]

                elif any("or" in words for words in root_label):
                    root_label = "Or"
                    eitherTregex = Tregex(None, "either", Condition("<<:", Tregex("either|Either", None)))
                    for m in findMatches(eitherTregex, VPChildren[0], root):
                        root = root.remove_subtrees([m["either"]])
                        root.init_indexes()
                        root_label = "XOR"
                        mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
                        VPChildren = mainVP.get_children("VP")
                
                elif any("nor" in words for words in root_label):
                    if any("neither" in words for words in root_label):
                        root_label = "NeitherNor"
                        neitherTregex = Tregex(None, "neither", Condition("<<:", Tregex("neither|Neither", None)))
                        for m in findMatches(neitherTregex, mainVP, root):
                            root = root.remove_subtrees([m["neither"]])
                            root.init_indexes()
                            mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
                            VPChildren = mainVP.get_children("VP")
                            minVPChildrenIndex, maxVPChildrenIndex = VPChildren[0].indexes[-1], VPChildren[-1].indexes[-1]
                    else:
                        root_label = "Nor"
                else:
                    bothTregex = Tregex(None, "both", Condition("<<:", Tregex("both|Both", None)))
                    for m in findMatches(bothTregex, mainVP, root):
                        root = root.remove_subtrees([m["both"]])
                        root.init_indexes()
                        mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
                        VPChildren = mainVP.get_children("VP")
                        minVPChildrenIndex, maxVPChildrenIndex = VPChildren[0].indexes[-1], VPChildren[-1].indexes[-1]
                    root_label = " ".join([" ".join(words) for words in root_label])
                    root_label = " ".join(sorted(set(root_label.split()), key=root_label.split().index)) #eliminates duplicates

            new_roots = []
            for child in VPChildren:
                newMainVP = Node("VP", *(c for c in mainVP if c.indexes[-1] == child.indexes[-1] or c.indexes[-1]<minVPChildrenIndex or c.indexes[-1]>maxVPChildrenIndex ))
                if len(newMainVP) == 1:
                    newMainVP = newMainVP[0]
                newRoot = root.replace_subtree(newMainVP, mainVP.indexes[len(root.indexes):])
                try:
                    words = " ".join(newRoot.get_words())
                    newRoot = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))
                except LanguageError:
                    pass
                new_roots.append(newRoot)
            new_root = Node(root_label, *new_roots)
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
            return tree
    return tree

def CoordVerbPhrase_cleaning(tree):
    oldTree = Node("r")
    CoordVerbPhrase = "ROOT <<: (S < ( NP $.. (VP <+(VP) (VP > VP $.. CC & $.. VP))))"
    CoordVerbPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None, Condition("<+VP", Tregex("VP", None, Condition("$", Tregex("CC", None)), Condition(">", Tregex("VP", "mainVP")))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordVerbPhraseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
            VPChildren = mainVP.get_children("VP")
            if len(VPChildren) > 1:
                continue
            root.init_indexes()
            VPChild = VPChildren[0]
            VPChildIndex = VPChild.indexes[-1]
            
            diverseVChildren = [v for v in mainVP.get_V_children() if v != VPChild]
            if diverseVChildren == []:
                tree.init_indexes()
                continue

            CC = mainVP.get_children("CC")
            root_label = [c.get_words() for c in CC]

            if len(CC) == 1:
                if VPChildIndex < CC[0].indexes[-1]:
                    VP_right = Node("VP", *(c for c in mainVP if c.indexes[-1] > CC[0].indexes[-1]))
                    new_mainVP = mainVP.remove_subtrees([[c.indexes[-1]] for c in mainVP if c.indexes[-1] > CC[0].indexes[-1]])
                    new_mainVP = new_mainVP.introduce_subtree(VP_right, [len(new_mainVP)])
                    new_root = root.replace_subtree(new_mainVP, mainVP.indexes[len(root.indexes):])

                else:
                    VP_left = Node("VP", *(c for c in mainVP if c.indexes[-1] < CC[0].indexes[0]))
                    new_mainVP = mainVP.remove_subtrees([[c.indexes[-1]] for c in mainVP if c.indexes[-1] < CC[0].indexes[-1]])
                    new_mainVP = new_mainVP.introduce_subtree(VP_left, [0])
                    new_root = root.replace_subtree(new_mainVP, mainVP.indexes[len(root.indexes):])

            elif len(CC) > 0:
                if VPChildIndex < CC[-1].indexes[-1]:
                    VP_right = Node("VP", *(c for c in mainVP if c.indexes[-1] > CC[-1].indexes[-1]))
                    new_mainVP = mainVP.remove_subtrees([[c.indexes[-1]] for c in mainVP if c.indexes[-1] > CC[-1].indexes[-1]])
                    new_mainVP = new_mainVP.introduce_subtree(VP_right, [len(new_mainVP)])
                    new_root = root.replace_subtree(new_mainVP, mainVP.indexes[len(root.indexes):])
                
                else:
                    VP_left = Node("VP", *(c for c in mainVP if c.indexes[-1] < CC[-1].indexes[-1] and c not in CC))
                    new_mainVP = mainVP.remove_subtrees([[c.indexes[-1]] for c in mainVP if c.indexes[-1] < CC[-1].indexes[-1]])
                    new_mainVP = new_mainVP.introduce_subtree(VP_left, [0])
                    for cc in CC[:-1]:
                        new_mainVP = new_mainVP.introduce_subtree(cc, [0])
                    new_root = root.replace_subtree(new_mainVP, mainVP.indexes[len(root.indexes):])
                
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def CoordVerbPhrase_cleaning2(tree):
    oldTree = Node("r")
    CoordVerbPhrase = "ROOT <<: (S < ( NP $.. (VP <+(VP) (VP > VP $.. CC & $.. VP))))"
    CoordVerbPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("VP", "mainVP", Condition("$..", Tregex("CC", None, Condition("$..", Tregex("VP", "mainVP")))), Condition("!>", Tregex("VP", None)))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordVerbPhraseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
            parent = root.get_node(match["mainVP"][len(match["Root"]):-1])

            new_node = Node("VP", parent)
            new_root = root.replace_subtree(new_node, mainVP.indexes[len(match["Root"]):-1])
            
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def NonResPostPartPhrase_processing1(tree):
    '''
    Again, I'm not sure if we should do this split'''
    NonResPostPartPhrase = "ROOT <<: (S < (NP $.. (VP <+ (VP) (NP|PP $.. (S <: (VP <<, VBG|VBN )))"
    NonResPostPartPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", "NP", Condition("$..", Tregex("VP", None, Condition("<+VP", Tregex("NP|PP", None, Condition("$..", Tregex("S", None, Condition("$,", Tregex(",", None)), Condition("<:", Tregex("VP", "VP", Condition("<<,", Tregex("VBG|VBN", None)))))))))))))))
    for match in findMatches(NonResPostPartPhraseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        VP = tree.get_node(match["VP"])
        SIndexes = match["VP"][:-1]
        #We get the part after the Participle Phrase (main clause)
        commaNextToS = tree.get_node(SIndexes[:-1])[SIndexes[-1]-1].label() == ","
        if commaNextToS:
            new_left = root.remove_subtrees([SIndexes[len(root.indexes):-1]+[SIndexes[-1]-1], SIndexes[len(root.indexes):]])
        else:
            new_left = root.remove_subtrees([SIndexes[len(root.indexes):]])

        #Append the VP with the NP as subject, and create a S node
        new_VP = VP.introduce_subtree(Node("VBZ", Node("is")), [0])
        new_right = Node("S", NP, new_VP, Node(".", Node(".")))

        new_root = Node("Subordinate", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def NonResPostPartPhrase_processing2(tree):
    '''
    Again, I'm not sure if we should do this split'''
    NonResPostPartPhrase = "ROOT <<: (S < (NP $.. (VP <+ (VP) (NP|PP $.. (S <: (VP <<, VBG|VBN )))"
    NonResPostPartPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", "NP", Condition("$..", Tregex("VP", None, Condition("<+VP", Tregex("NP|PP", None, Condition("$..", Tregex("PP|ADVP", "PartP", Condition("$,", Tregex(",", None)), Condition("<+PP|ADVP", Tregex("S", None, Condition("<:", Tregex("VP", "VP", Condition("<<,", Tregex("VBG|VBN", None)))))))))))))))))
    for match in findMatches(NonResPostPartPhraseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        VP = tree.get_node(match["VP"])
        PIndexes = match["PartP"]
        #We get the part after the Participle Phrase (main clause)
        commaNextToS = tree.get_node(PIndexes[:-1])[PIndexes[-1]-1].label() == ","
        if commaNextToS: #always true
            new_left = root.remove_subtrees([PIndexes[len(root.indexes):-1]+[PIndexes[-1]-1], PIndexes[len(root.indexes):]])
        else:
            new_left = root.remove_subtrees([PIndexes[len(root.indexes):]])

        #Append the VP with the NP as subject, and create a S node
        new_VP = VP.introduce_subtree(Node("VBZ", Node("is")), [0])
        new_label = tree.get_node(PIndexes).get_children("IN")[0][0].label()
        new_right = Node("S", NP, new_VP, Node(".", Node(".")))

        new_root = Node("Subordinate", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def PostPurpAdvClauseIOT_processing(tree):
    """
    I don't know if we should do this split
    Should be an implication? An "and"? 
    Weird to be an implication, but at the same time is more than a simple "and"
    """
    PostPurpAdvClauseIOT = "ROOT <<: (S < (NP $.. (VP < +(VP) (SBAR < (S <<, (VP <<, /(T|t)o/))))))"
    PostPurpAdvClauseIOTTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None, Condition("<+VP", Tregex("SBAR", "SBAR", Condition("<", Tregex("S", "S", Condition("<<,", Tregex("VP", "VP", Condition("<<,", Tregex("TO", "TO")))))))))))))))
    for match in findMatches(PostPurpAdvClauseIOTTregex, tree, tree):
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])
        SBARIndexes = match["SBAR"]
        TOIndexes = match["TO"]

        #We get the part before the SBAR (main clause)
        commaNextToSBAR = tree.get_node(SBARIndexes[:-1])[SBARIndexes[-1]-1].label() == ","
        if commaNextToSBAR:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):-1]+[SBARIndexes[-1]-1], SBARIndexes[len(root.indexes):]])
        else:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):]])

        #Take the inside of the S Sentence without the TO
        new_right = S.remove_subtrees([TOIndexes[len(S.indexes):]])
        if len(new_right.get_node(match["VP"][len(match["S"]):])) == 1 and new_right.get_node(match["VP"][len(match["S"]):])[0].label() == "VP":
            new_right = new_right.replace_subtree(new_right.get_node(match["VP"][len(S.indexes):])[0], match["VP"][len(S.indexes):]) #Clean VP(VP(...)) to VP(...)
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(S)])

        new_root = Node("Purpose", Node("ROOT", new_right), new_left)
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def PostPurpAdvClauseT_processing(tree):
    """
    I don't know if we should do this split
    Should be an implication? An "and"? 
    Weird to be an implication, but at the same time is more than a simple "and"
    """
    PostPurpAdvClauseIOT = "ROOT <<: (S < (NP $.. (VP < +(VP) (NP|PP $.. (S <<, (VP <<, /(T|t)o/))))))"
    PostPurpAdvClauseIOTTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None, Condition("<+VP", Tregex("NP|PP", None, Condition("$..", Tregex("S", "S", Condition("<<,", Tregex("VP", "VP", Condition("<<,", Tregex("TO", "TO")))))))))))))))
    for match in findMatches(PostPurpAdvClauseIOTTregex, tree, tree):
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])
        TOIndexes = match["TO"]

        #We get the part before the S (main clause)
        commaNextToSBAR = tree.get_node(S.indexes[:-1])[S.indexes[-1]-1].label() == ","
        if commaNextToSBAR:
            new_left = root.remove_subtrees([S.indexes[len(root.indexes):-1]+[S.indexes[-1]-1], S.indexes[len(root.indexes):]])
        else:
            new_left = root.remove_subtrees([S.indexes[len(root.indexes):]])

        #Take the inside of the S Sentence without the TO
        new_right = S.remove_subtrees([TOIndexes[len(S.indexes):]])
        if len(new_right.get_node(match["VP"][len(match["S"]):])) == 1 and new_right.get_node(match["VP"][len(match["S"]):])[0].label() == "VP":
            new_right = new_right.replace_subtree(new_right.get_node(match["VP"][len(S.indexes):])[0], match["VP"][len(S.indexes):]) #Clean VP(VP(...)) to VP(...)
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(S)])

        new_root = Node("Purpose", Node("ROOT", new_right), new_left)
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def DirectAttrPost_processing(tree):
    '''
    Pattern inutile'''
    DirectAttrPost = "ROOT <<: (S < (NP $.. (VP=vp <+(VP) (SBAR=sbar [,, /``/=start | <<, /``/=start] [.. /''/=end | <<- /''/=end]))))"
    pass

def RepAttrPre_processing(tree):
    RepAttrPre = "ROOT <<: (S < (NP $.. (VP=vp <+(VP) (SBAR < S))))"
    RepAttrPreTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", "VP", Condition("<+VP|ADJP", Tregex("SBAR", "SBAR", Condition("<", Tregex("S|SBAR", "S")))))))))))
    for match in findMatches(RepAttrPreTregex, tree, tree):
        #FIRST NEED TO CHECK THAT VP IS AN ATTRIBUTION VERB
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])
        SBARIndexes = match["SBAR"]

        #We get the part before the SBAR (attribution clause)
        commaNextToSBAR = tree.get_node(SBARIndexes[:-1])[SBARIndexes[-1]-1].label() == ","
        if commaNextToSBAR:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):-1]+[SBARIndexes[-1]-1], SBARIndexes[len(root.indexes):]])
        else:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):]])
            
        if S.label() == "SBAR":
            SBAR = tree.get_node(match["SBAR"])
            SBAR_children = SBAR.get_children("SBAR")
            for child in SBAR_children:
                if len(child.get_children("S")) == 1:
                    Schild = child.get_children("S")[0]
                    SBAR = SBAR.replace_subtree(Schild, child.indexes[len(SBAR.indexes):])
            SBAR.set_label("S")
            new_right = SBAR
        
        else:
            new_right = S
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(S)])
        new_root = Node("Attribution", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def PostAttrClause_processing(tree):
    PostAttrClause = "ROOT <<: (S < (NP $.. (VP <+(VP) (SBAR=sbar < (S=s < (NP $.. VP))))))"
    PostAttrClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None, Condition("<+VP", Tregex("SBAR", "SBAR", Condition("<", Tregex("S", "S", Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None)))))))))))))))
    for match in findMatches(PostAttrClauseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])
        SBARIndexes = match["SBAR"]

        #We first insert the appositive NP in the main NP in stead of the original NP1, and remove everything after it (NP1 index now corresponds to that of the main NP)
        commaNextToSBAR = tree.get_node(SBARIndexes[:-1])[SBARIndexes[-1]-1].label() == ","
        if commaNextToSBAR:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):-1]+[SBARIndexes[-1]-1], SBARIndexes[len(root.indexes):]])
        else:
            new_left = root.remove_subtrees([SBARIndexes[len(root.indexes):]])

        #We create a new sentence
        #Should do something to detect plurality, so that we can use "are" instead of "is"
        new_right = S.introduce_subtree(Node(".", Node(".")), [len(S)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree

def DirectAttrPre_processing(tree):
    pass

def RepAttrPost_processing(tree):
    RepAttrPost = "ROOT <<: (S < (S|SBAR|SBARQ=s $.. (NP=np [$,, VP=vpb | $.. VP=vpa])))"
    RepAttrPostTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("S|SBAR|SBARQ", "S", Condition("$..", Tregex("NP", "NP", Condition("$", Tregex("VP", "VP")))))))))
    for match in findMatches(RepAttrPostTregex, tree, tree):
        #FIRST NEED TO CHECK THAT VP IS AN ATTRIBUTION VERB
        root = tree.get_node(match["Root"])
        S = tree.get_node(match["S"])

        #We get the part after the SBAR (attribution clause)
        commaNextToS = tree.get_node(S.indexes[:-1])[S.indexes[-1]+1].label() == ","
        if commaNextToS:
            new_left = root.remove_subtrees([S.indexes[len(root.indexes):], S.indexes[len(root.indexes):-1]+[S.indexes[-1]+1]])
        else:
            new_left = root.remove_subtrees([S.indexes[len(root.indexes):]])

        new_right = S
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(S)])
        new_root = Node("Attribution", Node("ROOT", new_right), new_left)
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def EmbPartPhrase_processing(tree):
    #EmbPartPhrase = "ROOT <<: (S < VP &<< (NP|PP <, ( NP ?$+ PP & $ + + (/,/ $+ (VP [<, (ADVP|PP $+ VBG|VBN) | <, VBG|VBN] & ?$+ /,/)))))"
    #EmbPartPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", "S", Condition("<", Tregex("VP", None, Condition("<<", Tregex("NP|PP", "MainNP", Condition("<,", Tregex("NP", "RelNP", Condition("$..", Tregex(",", "leftComma", Condition("$+", Tregex("VP", "VP", Condition("<", Tregex("VBG|VBN", "VBGN" None)), Condition("$+", Tregex(",", None)))))))))))))))
    return tree
    for match in findMatches(EmbPartPhraseTregex, tree, tree):
        if (match["VBGN"][-1] == 1 and tree.get_node(match["VBGN"][:-1])[0].label() not in {"ADVP", "PP"}) or match["VBGN"][-1] != 0:
            continue
        root = tree.get_node(match["Root"])
        MainNP = tree.get_node(match["MainNP"])
        RelNP = tree.get_node(match["RelNP"])
        S = tree.get_node(match["S"])
        VP = tree.get_node(match["VP"])

        new_left = root.remove_subtrees([match["leftComma"][len(root.indexes):-1]+[i] for i in range(match["leftComma"][-1], len(MainNP))])
        if len(new_left[0]) == 1 and new_left[0].label() == "NP":
            new_left = new_left.replace_subtree(new_left[0][0], [0])

        new_right = Node("S", RelNP, Node("VP", Node("VBZ", Node("is")), VP))
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def ResWhoWhichClause_processing(tree):
    '''
    I don't really like this one, it's hard to transform into two proper sentences.
    Best thing to do would be to transform into universal quantification, but that we don't want to instantiate?
    If the verb is be, then a simple "and" is enough.
    If not then we should somehow transform the sentence into a universal quantification
    For now, we will just split the sentence in two (but we lose a lot of information on coreference)
    Other option is just to do nothing
    This will be useful in the case of universal quantification, and could be skipped in the general case
    '''
    ResWhoWhich = "ROOT <<: (S << (NP <, ( NP $.. (SBAR <, (WHNP $+ S & <<: WP|WDT) & ?$+ /,/))))"
    ResWhoWhichTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", None, Condition("<,", Tregex("NP", "NP", Condition("$..", Tregex("SBAR", "SBAR", Condition("<,", Tregex("WHNP", "WHNP", Condition("$+", Tregex("S", "S")), Condition("<<:", Tregex("WP|WDT", None)))))))))))))
    for match in findMatches(ResWhoWhichTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        WHNP = tree.get_node(match["WHNP"])
        S = tree.get_node(match["S"])

        new_left = root.remove_subtrees([match["SBAR"][len(root.indexes):]])

        if len(new_left.get_node(match["SBAR"][len(root.indexes):-1])) == 1:
            new_left = new_left.replace_subtree(new_left.get_node(match["NP"][len(root.indexes):]), match["NP"][len(root.indexes):-1])

        new_right = S.introduce_subtree(NP, [0])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()

    return tree
        
def ResWhoseClause_processing(tree):
    ResWhoseClause = "ROOT <<: (S << (NP=head < (NP=np $+ (SBAR=sbar <, (WHNP $+ S=s & <, (/WP\\$/ $+ /.*/=nn))))))"
    ResWhoseClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", None, Condition("<", Tregex("NP", "NP", Condition("$+", Tregex("SBAR", "SBAR", Condition("<,", Tregex("WHNP", "WHNP", Condition("$+", Tregex("S", "S")), Condition("<,", Tregex("WP$", None, Condition("$+", Tregex(None, None)))))))))))))))
    for match in findMatches(ResWhoseClauseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        RelNP = tree.get_node(match["NP"])
        Whose = tree.get_node(match["WHNP"])
        S = tree.get_node(match["S"])

        new_left = root.remove_subtrees([match["SBAR"][len(root.indexes):]])
        if len(new_left.get_node(match["SBAR"][len(root.indexes):-1])) == 1:
            new_left = new_left.replace_subtree(new_left.get_node(match["NP"]), match["NP"][len(root.indexes):-1])

        WhoseComplement = Whose[1:]
        newSubject = Node("NP", RelNP.introduce_subtree(Node("POS", Node("'s")), [len(RelNP)]))
        for c in WhoseComplement:
            newSubject = newSubject.introduce_subtree(c, [len(newSubject)])
        new_right = S.introduce_subtree(newSubject, [0])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def RedRelClause_processing(tree):
    ''''
    Same issue as earlier
    Not sure how to consider the optional PP and IN
    '''
    RedRelClause = "ROOT <<: (S << (NP <, (NP=np $.. (SBAR=sbar <: (S=s < (VP ?< (PP=prep ?<: IN=in)))))))"
    RedRelClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", "NP", Condition("<,", Tregex("NP", None, Condition("$..", Tregex("SBAR", "SBAR", Condition("<:", Tregex("S", "S", Condition("<", Tregex("VP", "VP")))))))))))))
    for match in findMatches(RedRelClauseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        S = tree.get_node(match["S"])
        VP = tree.get_node(match["VP"])
        
        new_left = root.remove_subtrees([match["SBAR"][len(root.indexes):]])
        if len(new_left.get_node(match["SBAR"][len(root.indexes):-1])) == 1:
            new_left = new_left.replace_subtree(new_left.get_node(match["NP"]), match["NP"][len(root.indexes):-1])

        new_right = (S.introduce_subtree(VP.introduce_subtree(NP, [-1]), VP.indexes[-1])).remove_subtrees([VP.indexes[-1]+1])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def ResWhomClause_processing(tree):
    ResWhomClause = "ROOT <<: (S << (NP=head <, NP=np & < (SBAR=sbar <, (WHNP $+ (S=s <, NP=np2 & <- (VP=vp ?<+(VP) PP=prep)) & <<: (WP <: whom)))))"
    #ResWhomClauseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", "S", Condition("<<", Tregex("NP", None, Condition("<,", Tregex("NP", "NP", Condition("<", Tregex("SBAR", "SBAR", Condition("<,", Tregex("WHNP", None, Condition("$+", Tregex("S", "S", Condition("<,", Tregex("NP", None,)) Condition("<-", Tregex("VP", "VP")))), Condition("<<:", Tregex("WP", None, Condition("<:", Tregex("whom", None)))))))))))))))
    for match in findMatches(ResWhomClauseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        RelNP = tree.get_node(match["NP"])
        S = tree.get_node(match["S"])
        VP = tree.get_node(match["VP"])

        new_left = root.remove_subtrees([match["SBAR"][len(root.indexes):]])
        if len(new_left.get_node(match["SBAR"][len(root.indexes):-1])) == 1:
            new_left = new_left.replace_subtree(RelNP, match["NP"][len(root.indexes):-1])
        
        new_right = (S.introduce_subtree(VP.introduce_subtree(NP, [-1]), VP.indexes[-1])).remove_subtrees([VP.indexes[-1]+1])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def PrepPhaseCompVerb_processing(tree):
    '''
    No need to split
    '''

def ResPostPartPhrase_processing1(tree):
    '''
    Wondering if it is worth splitting this. Same issues as other Restrictive clauses
    '''
    ResPostPartPhrase = "ROOT <<: (S=s < VP=mainverb &<< (NP|PP=head <, (NP=np $+ (VP=vp [<, (ADVP|PP $+ VBG|VBN=vbgn) | <, VBG|VBN=vbgn] )) & > (PP !> S)]))"
    ResPostPartPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("VP", None, Condition("<<", Tregex("NP|PP", None, Condition("<,", Tregex("NP", "NP", Condition("$+", Tregex("VP", "VP", Condition("<", Tregex("VBG|VBN", None)))))), Condition(">", Tregex("PP", None, Condition("!>", Tregex("S", None)))))))))))
    for match in findMatches(ResPostPartPhraseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        VP = tree.get_node(match["VP"])

        new_left = root.remove_subtrees([match["VP"][len(root.indexes):]])
        if len(new_left.get_node(match["NP"][len(root.indexes):-1])) == 1 and new_left.get_node(match["NP"][len(root.indexes):-1]).label() == "NP":
            new_left = new_left.replace_subtree(new_left.get_node(match["NP"][len(root.indexes):-1]), match["NP"][len(root.indexes):-1])
        
        newVP = Node("VP", Node("VBZ", Node("is")), VP)
        new_right = NP.introduce_subtree(newVP, [len(NP)])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def ResPostPartPhrase_processing2(tree):
    '''
    Wondering if it is worth splitting this. Same issues as other Restrictive clauses
    '''
    ResPostPartPhrase = "ROOT <<: (S=s < VP=mainverb &<< (NP|PP=head <, (NP=np $+ (VP=vp [<, (ADVP|PP $+ VBG|VBN=vbgn) | <, VBG|VBN=vbgn] )) & > (VP > S))"
    ResPostPartPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("VP", None)), Condition("<<", Tregex("NP|PP", None, Condition("<,", Tregex("NP", "NP", Condition("$+", Tregex("VP", "VP", Condition("<", Tregex("VBG|VBN", None)))))), Condition(">", Tregex("VP", None, Condition(">", Tregex("S", None)))))))))
    for match in findMatches(ResPostPartPhraseTregex, tree, tree):
        root = tree.get_node(match["Root"])
        NP = tree.get_node(match["NP"])
        VP = tree.get_node(match["VP"])

        new_left = root.remove_subtrees([match["VP"][len(root.indexes):]])
        if len(new_left.get_node(match["NP"][len(root.indexes):-1])) == 1 and new_left.get_node(match["NP"][len(root.indexes):-1]).label() == "NP":
            new_left = new_left.replace_subtree(new_left.get_node(match["NP"][len(root.indexes):-1]), match["NP"][len(root.indexes):-1])
        
        newVP = Node("VP", Node("VBZ", Node("is")), VP)
        new_right = NP.introduce_subtree(newVP, [len(NP)])
        new_right = new_right.introduce_subtree(Node(".", Node(".")), [len(new_right)])

        new_root = Node("And", new_left, Node("ROOT", new_right))
        tree = tree.replace_subtree(new_root, match["Root"])
        tree.init_indexes()
    return tree

def PreAdjAdvPhrase_processing(tree):
    '''
    No need to split
    '''

def PostEmbAdjAdvPhrase_processing(tree):
    '''
    No need to split
    '''

def LeadNounPhrase_processing(tree):
    '''
    No need to split
    '''

def PrePrepPhraseComma_processing(tree):
    '''
    No need to split
    '''

def PostEmbPrepPhraseComma_processing(tree):
    '''
    No need to split? Maybe check as it is similar to adverbial clauses but to see
    '''

def CoordNounPhraseObj_cleaning(tree):
    oldTree = Node("r")
    CoordNounPhraseObj = "ROOT <<: ( S < (NP $.. (VP << (NP|UCP=np1 < (NP ?$.. CC & $.. NP=np2)))))"
    CoordNounPhraseObjTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None, Condition("<<", Tregex("NP|UCP", "mainNP", Condition("<", Tregex("NP|NN|NNS|NNP|NNPS|NML", None, Condition("$..", Tregex("NP|NN|NNS|NNP|NNPS|NML", None)))))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordNounPhraseObjTregex, tree, tree):
            root = tree.get_node(match["Root"])
            mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
            NPChildren = mainNP.get_children("NP")
            if len(NPChildren) > 1:
                continue

            diverseNChildren = [n for n in mainNP.get_N_children() if n.label()!= "NP"]
            if diverseNChildren == []:
                continue

            root.init_indexes()
            newMainNP = mainNP
            for c in diverseNChildren:
                newC = Node("NP", c)
                newMainNP = newMainNP.replace_subtree(newC, [c.indexes[-1]])
            new_root = root.replace_subtree(newMainNP, mainNP.indexes[len(root.indexes):])
                
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def CoordNounPhraseObj_processing(tree, spacyModel):
    oldTree = Node("r")
    CoordNounPhraseObj = "ROOT <<: ( S < (NP $.. (VP << (NP|UCP=np1 < (NP ?$.. CC & $.. NP=np2)))))"
    CoordNounPhraseObjTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", None, Condition("$..", Tregex("VP", None, Condition("<<", Tregex("NP|UCP", "mainNP", Condition("<", Tregex("NP", None, Condition("$..", Tregex("CC", None, Condition("$..", Tregex("NP", None)))))))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordNounPhraseObjTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
            NPChildren = mainNP.get_children("NP")
            minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]
            
            if not any([c.label() in {"CC", ","} for c in mainNP[minNPChildrenIndex:maxNPChildrenIndex]]):
                continue
                
            CC = mainNP.get_children("CC")
            root_label = "And"
            if CC:
                root_label = [c.get_words() for c in CC]
                if any("either" in words for words in root_label):
                    eitherIndex = [i for i, words in enumerate(root_label) if "either" in words]
                    root_label = "XOR"
                    root = root.remove_subtrees([CC[i].indexes for i in eitherIndex])
                    root.init_indexes()
                    mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
                    NPChildren = mainNP.get_children("NP")
                    minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]

                elif any("or" in words for words in root_label):
                    root_label = "Or"
                    eitherTregex = Tregex(None, "either", Condition("<<:", Tregex("either|Either", None)))
                    for m in findMatches(eitherTregex, NPChildren[0], root):
                        root = root.remove_subtrees([m["either"]])
                        root.init_indexes()
                        root_label = "XOR"
                        mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
                        NPChildren = mainNP.get_children("NP")
                
                elif any("nor" in words for words in root_label):
                    if any("neither" in words for words in root_label):
                        root_label = "NeitherNor"
                        neitherTregex = Tregex(None, "neither", Condition("<<:", Tregex("neither|Neither", None)))
                        for m in findMatches(neitherTregex, mainNP, root):
                            root = root.remove_subtrees([m["neither"]])
                            root.init_indexes()
                            mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
                            NPChildren = mainNP.get_children("NP")
                            minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]
                    else:
                        root_label = "Nor"
                else:
                    bothTregex = Tregex(None, "both", Condition("<<:", Tregex("both|Both", None)))
                    for m in findMatches(bothTregex, mainNP, root):
                        root = root.remove_subtrees([m["both"]])
                        root.init_indexes()
                        mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
                        NPChildren = mainNP.get_children("NP")
                        minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]
                    root_label = " ".join([" ".join(words) for words in root_label])
                    root_label = " ".join(sorted(set(root_label.split()), key=root_label.split().index)) #eliminates duplicates

            new_roots = []
            for child in NPChildren:
                newMainNP = Node("NP", *(c for c in mainNP if c.indexes[-1] == child.indexes[-1] or c.indexes[-1]<minNPChildrenIndex or c.indexes[-1]>maxNPChildrenIndex ))
                if len(newMainNP) == 1:
                    newMainNP = newMainNP[0]
                newRoot = root.replace_subtree(newMainNP, mainNP.indexes[len(root.indexes):])
                """try:
                    words = " ".join(newRoot.get_words())
                    print(words)
                    newRoot = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))
                except LanguageError:
                    pass"""
                new_roots.append(newRoot)
            new_root = Node(root_label, *new_roots)
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def CoordNounPhraseSubj_cleaning(tree):
    oldTree = Node("r")
    CoordNounPhrase = "ROOT <<: ( S < (NP $.. (VP << (NP|UCP=np1 < (NP ?$.. CC & $.. NP=np2)))))"
    CoordNounPhraseTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP|UCP", "mainNP", Condition("$..", Tregex("VP", None)), Condition("<", Tregex("NP|NN|NNS|NNP|NNPS|NML", None, Condition("$..", Tregex("CC", None, Condition("$..", Tregex("NP|NN|NNS|NNP|NNPS|NML", None)))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordNounPhraseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
            NPChildren = mainNP.get_children("NP")
            if len(NPChildren) > 1:
                continue

            diverseNChildren = [n for n in mainNP.get_N_children() if n.label()!= "NP"]
            if diverseNChildren == []:
                continue

            root.init_indexes()
            newMainNP = mainNP
            for c in diverseNChildren:
                newC = Node("NP", c)
                newMainNP = newMainNP.replace_subtree(newC, [c.indexes[-1]])
            new_root = root.replace_subtree(newMainNP, mainNP.indexes[len(root.indexes):])
                
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def CoordNounPhraseSubj_processing(tree, spacyModel):
    oldTree = Node("r")
    CoordNounPhraseSubj = "ROOT <<: (S < (NP=np1 < (NP ?$.. CC & $.. NP=np2) $.. VP ))"
    CoordNounPhraseSubjTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP|UCP", "mainNP", Condition("<", Tregex("NP", None, Condition("$..", Tregex("NP", None)))), Condition("$..", Tregex("VP", None)))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordNounPhraseSubjTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
            NPChildren = mainNP.get_children("NP")
            minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]
            
            if not any([c.label() in {"CC", ","} for c in mainNP[minNPChildrenIndex:maxNPChildrenIndex]]):
                continue
                

            CC = mainNP.get_children("CC")
            root_label = "And"

            if CC:
                root_label = [c.get_words() for c in CC]
                if any("either" in words for words in root_label):
                    eitherIndex = [i for i, words in enumerate(root_label) if "either" in words]
                    root_label = "XOR"
                    root = root.remove_subtrees([CC[i].indexes for i in eitherIndex])
                    root.init_indexes()
                    mainNP = root.get_node(match["mainNP"])
                    NPChildren = mainNP.get_children("mainNP")
                    minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]

                elif any("or" in words for words in root_label):
                    root_label = "Or"
                    eitherTregex = Tregex(None, "either", Condition("<<:", Tregex("either|Either", None)))
                    for m in findMatches(eitherTregex, NPChildren[0], root):
                        root = root.remove_subtrees([m["either"]])
                        root.init_indexes()
                        root_label = "XOR"
                        mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
                        NPChildren = mainNP.get_children("NP")
                
                elif any("nor" in words for words in root_label):
                    if any("neither" in words for words in root_label):
                        root_label = "NeitherNor"
                        neitherTregex = Tregex(None, "neither", Condition("<<:", Tregex("neither|Neither", None)))
                        for m in findMatches(neitherTregex, mainNP, root):
                            root = root.remove_subtrees([m["neither"]])
                            root.init_indexes()
                            mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
                            NPChildren = mainNP.get_children("NP")
                            minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]
                    else:
                        root_label = "Nor"
                else:
                    bothTregex = Tregex(None, "both", Condition("<<:", Tregex("both|Both", None)))
                    for m in findMatches(bothTregex, mainNP, root):
                        root = root.remove_subtrees([m["both"]])
                        root.init_indexes()
                        mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
                        NPChildren = mainNP.get_children("NP")
                        minNPChildrenIndex, maxNPChildrenIndex = NPChildren[0].indexes[-1], NPChildren[-1].indexes[-1]
                    root_label = " ".join([" ".join(words) for words in root_label])
                    root_label = " ".join(sorted(set(root_label.split()), key=root_label.split().index)) #eliminates duplicates

            new_roots = []
            for child in NPChildren:
                newMainNP = Node("NP", *(c for c in mainNP if c.indexes[-1] == child.indexes[-1] or c.indexes[-1]<minNPChildrenIndex or c.indexes[-1]>maxNPChildrenIndex ))
                if len(newMainNP) == 1:
                    newMainNP = newMainNP[0]
                newRoot = root.replace_subtree(newMainNP, mainNP.indexes[len(root.indexes):])
                """try:
                    words = " ".join(newRoot.get_words())
                    newRoot = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))
                except LanguageError:
                    pass"""
                new_roots.append(newRoot)
            new_root = Node(root_label, *new_roots)
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def CoordAdjPhrase_processing(tree):
    oldTree = Node("r")
    CoordAdjPhrase = "ROOT <<: (S << (ADJP < (JJ $.. CC $.. JJ)))"
    CoordAdjPhraseTregex = Tregex("ROOT", "Root", Condition("!>>", Tregex("And|and", None)), Condition("<<:", Tregex("S", None, Condition("<<", Tregex("ADJP", "mainADJP", Condition("<", Tregex("JJ|ADJP", None, Condition("$..", Tregex("JJ|ADJP", None)))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordAdjPhraseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            mainADJP = root.get_node(match["mainADJP"][len(match["Root"]):])
            ADJPChildren = mainADJP.get_adj_children()
            minJJChildrenIndex, maxJJChildrenIndex = ADJPChildren[0].indexes[-1], ADJPChildren[-1].indexes[-1]
            
            CC = mainADJP.get_children("CC")
            root_label = "And"
            if CC:
                root_label = [c.get_words() for c in CC]
                if any("either" in words for words in root_label):
                    eitherIndex = [i for i, words in enumerate(root_label) if "either" in words]
                    root_label = "XOR"
                    root = root.remove_subtrees([CC[i].indexes for i in eitherIndex])
                    root.init_indexes()
                    mainADJP = root.get_node(match["mainADJP"][len(match["Root"]):])
                    ADJPChildren = mainADJP.get_adj_children()
                    minJJChildrenIndex, maxJJChildrenIndex = ADJPChildren[0].indexes[-1], ADJPChildren[-1].indexes[-1]

                elif any("or" in words for words in root_label):
                    root_label = "Or"
                    eitherTregex = Tregex(None, "either", Condition("<<:", Tregex("either|Either", None)))
                    for m in findMatches(eitherTregex, ADJPChildren[0], root):
                        root = root.remove_subtrees([m["either"]])
                        root.init_indexes()
                        root_label = "XOR"
                        mainADJP = root.get_node(match["mainADJP"][len(match["Root"]):])
                        ADJPChildren = mainADJP.get_adj_children()
                
                elif any("nor" in words for words in root_label):
                    if any("neither" in words for words in root_label):
                        root_label = "NeitherNor"
                        neitherTregex = Tregex(None, "neither", Condition("<<:", Tregex("neither|Neither", None)))
                        for m in findMatches(neitherTregex, mainADJP, root):
                            root = root.remove_subtrees([m["neither"]])
                            root.init_indexes()
                            mainADJP = root.get_node(match["mainADJP"][len(match["Root"]):])
                            ADJPChildren = mainADJP.get_adj_children()
                            minJJChildrenIndex, maxJJChildrenIndex = ADJPChildren[0].indexes[-1], ADJPChildren[-1].indexes[-1]
                    else:
                        root_label = "Nor"
                else:
                    bothTregex = Tregex(None, "both", Condition("<<:", Tregex("both|Both", None)))
                    for m in findMatches(bothTregex, mainADJP, root):
                        root = root.remove_subtrees([m["both"]])
                        root.init_indexes()
                        mainADJP = root.get_node(match["mainADJP"][len(match["Root"]):])
                        ADJPChildren = mainADJP.get_adj_children()
                        minJJChildrenIndex, maxJJChildrenIndex = ADJPChildren[0].indexes[-1], ADJPChildren[-1].indexes[-1]
                    root_label = " ".join([" ".join(words) for words in root_label])
                    root_label = " ".join(sorted(set(root_label.split()), key=root_label.split().index)) #eliminates duplicates
            new_roots = []
            for child in ADJPChildren:
                newMainADJP = Node("ADJP", *(c for c in mainADJP if c.indexes[-1] == child.indexes[-1] or c.indexes[-1]<minJJChildrenIndex or c.indexes[-1]>maxJJChildrenIndex ))
                if len(newMainADJP) == 1:
                    newMainADJP = newMainADJP[0]
                newRoot = root.replace_subtree(newMainADJP, mainADJP.indexes[len(root.indexes):])
                new_roots.append(newRoot)
            new_root = Node(root_label, *new_roots)
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def CoordAdjPhrase_cleaning(tree):
    oldTree = Node("r")
    CoordAdjPhraseTregex = Tregex("ROOT", "Root", Condition("!>>", Tregex("And|and", None)), Condition("<<:", Tregex("S", None, Condition("<<", Tregex("JJ|ADJP", "Adj", Condition("$+", Tregex(",", None, Condition("$+", Tregex("JJ|ADJP", None, Condition("!>", Tregex("ADJP", None)))))))))))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(CoordAdjPhraseTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            ADJParent = root.get_node(match["Adj"][len(match["Root"]):-1])
            ADJPChildren = ADJParent.get_adj_children()
            minJJChildrenIndex, maxJJChildrenIndex = ADJPChildren[0].indexes[-1], ADJPChildren[-1].indexes[-1]
            new_ADJP = Node("ADJP", *(c for c in ADJParent if c.indexes[-1] >= minJJChildrenIndex and c.indexes[-1] <= maxJJChildrenIndex))
            new_root = root.remove_subtrees([c.indexes for c in ADJParent if c.indexes[-1] >= minJJChildrenIndex and c.indexes[-1] <= maxJJChildrenIndex])
            new_root = new_root.introduce_subtree(new_ADJP, match["Adj"][len(match["Root"]):])
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree


def be_sentence(tree):
    BeSentenceTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("VP", None, Condition("<<", Tregex("VB|VBD|VBG|VBN|VBP|VBZ", None, Condition("<", Tregex("am|is|are|was|were|been|being|be", None)), Condition("!$..", Tregex("VP|VBG|VBN",None)))))))))
    try:
        next(findMatches(BeSentenceTregex, tree, tree))
        return True
    except StopIteration:
        return False

def simple_present(tree):
    SimplePresentTregex = Tregex("VP", None, Condition("<", Tregex("VB|VBP|VBZ", None, Condition("!$+", Tregex("VP|VBG|VBN", None)))))
    try:
        next(findMatches(SimplePresentTregex, tree, tree))
        return True
    except StopIteration:
        return False

def indefinite_plural_NP(tree, pluralNN):
    #first we remove everything right of pluralNN, then check if there is a determiner
    newTree = tree.remove_following(pluralNN)
    IndefinitePluralNPTregex = Tregex("DT|PRP$|POS|CD", None)
    try:
        next(findMatches(IndefinitePluralNPTregex, newTree, newTree))
        return False
    except StopIteration:
        pass
    #then we check if there are NNP or NNPS in the NP
    IndefinitePluralNPTregex = Tregex("NNP|NNPS", None)
    try:
        next(findMatches(IndefinitePluralNPTregex, newTree, newTree))
        return False
    except StopIteration:
        return True

def singularize_NP(tree):
    NNSTregex = Tregex("NNS|NNPS", "PluralNoun", Condition("!>>", Tregex("VP", None)))
    introductions = []
    for match in findMatches(NNSTregex, tree, tree):
        noun = tree.get_node(match["PluralNoun"]).get_words()[0]
        lemma = singularize_noun(noun)
        newLabel = "NN" if tree.get_node(match["PluralNoun"]).label == "NNS" else "NNP"
        parent = tree.get_node(match["PluralNoun"][:-1])
        if parent == tree:
            insert_index = match["PluralNoun"]
            for j in range(match["PluralNoun"][-1]-1, -1, -1):
                current_label = tree.get_node(match["PluralNoun"][:-1]+[j]).label()
                if current_label in {"JJ", "JJR", "JJS", "ADJP"}:
                    insert_index = match["PluralNoun"][:-1]+[j]
                if current_label not in {"JJ", "JJR", "JJS", "ADJP", "DT", "CD", "CC", ","} and insert_index != match["PluralNoun"]:
                    insert_index = match["PluralNoun"][:-1]+[j+1]
                    break
        else:
            insert_index = match["PluralNoun"][:-1]+[0]
        insert_first_letter = tree.get_node(insert_index).get_words()[0][0]
        if insert_first_letter.lower() in {"a","e","i","o","u"}:
            determiner = "an"
        else:
            determiner = "a"

        if insert_index == match["PluralNoun"]:
            tree = tree.replace_subtree(Node("NP", Node("DT", Node(determiner)), Node(newLabel, Node(lemma))), match["PluralNoun"])
        else:
            tree = tree.replace_subtree(Node(newLabel, Node(lemma)), match["PluralNoun"])
            introductions.append((Node("DT", Node(determiner)), insert_index))
    for intro in introductions[::-1]:
        tree = tree.introduce_subtree(intro[0], intro[1])
    tree.init_indexes()
    return tree

def singularize_noun(noun):
    lemma = getAllLemmas(noun, "NOUN")
    try:
        lemma = lemma["NOUN"][0]
    except (IndexError, KeyError):
        if noun[-3:] == "ves":
            lemma = noun[:-3] + "f"
        elif noun[-3:] in {"ses", "xes", "zes"} or noun[-4:] in {"ches", "shes"}:
            lemma = noun[:-2]
        elif noun[-3:] == "ies":
            lemma = noun[:-3] + "y"
        elif noun[-1:] == "s":
            lemma = noun[:-1]
        else:
            return getLemma(noun, "NOUN")[0]
    return lemma

def singularize_verbs(tree):
    #Ideally, tree is a VP
    #We want to singularize the verbs in the VP
    #In any VP led by "be", we also want to singularize the object

    #Base case: any verb in non-3rd person present form (or base form & not preceeded by a modal verb in case parsing is not perfect) is singularized
    VBPTregex = Tregex("VBP|VB", "Verb", Condition("!$,,", Tregex("MD", None)), Condition("!>>", Tregex("VP", None, Condition("$,,", Tregex("MD", None)))))
    for match in findMatches(VBPTregex, tree, tree):
        verb = tree.get_node(match["Verb"]+[0]).get_words()[0]
        lemma = getAllLemmas(verb, "VERB")
        try:
            lemma = lemma["VERB"][0]
            if lemma == "be":
                continue
            singular_lemma = getInflection(lemma, "VBZ")[0]
        except IndexError:
            continue
        tree = tree.replace_subtree(Node("VBZ", Node(singular_lemma)), match["Verb"])

    #VP led by "be"
    VBPTregex = Tregex("VBP|VB", "Verb", Condition("!$..", Tregex("VP|VBG|VBN",None)), Condition("!$,,", Tregex("MD", None)))
    for match in findMatches(VBPTregex, tree, tree):
        verb = tree.get_node(match["Verb"]+[0]).get_words()[0]
        lemma = getAllLemmas(verb, "VERB")
        try:
            lemma = lemma["VERB"][0]
            if lemma != "be":
                continue
            if verb == "were":
                singular_lemma = "was"
                form = "VBD"
            else:
                singular_lemma = "is"
                form = "VBZ"
            parent = tree.get_node(match["Verb"][:-1])
            parent.init_indexes()
            newParent = singularize_be_object(parent, match["Verb"][-1])
            tree = tree.replace_subtree(newParent, match["Verb"][:-1])
            tree.init_indexes()
        except IndexError:
            #print("COULD NOT FIND LEMMA FOR", verb)
            continue
        tree = tree.replace_subtree(Node(form, Node(singular_lemma)), match["Verb"])
    
    #Case of "were/are" as auxilary verbs
    wereTregex = Tregex("VBD|VBP", "Verb", Condition("<", Tregex("were|are", "be")))
    for match in findMatches(wereTregex, tree, tree):
        if tree.get_node(match["be"]).get_words()[0] == "were":
            singular_lemma = "was"
            tree = tree.replace_subtree(Node("VBD", Node(singular_lemma)), match["Verb"])
        else:
            singular_lemma = "is"
            tree = tree.replace_subtree(Node("VBZ", Node(singular_lemma)), match["Verb"])

    return tree
        
def singularize_be_object(parent, be_index):
    #tree is the main VP
    #be_index is the index of the node containing the "be" verb
    tree = copy.deepcopy(parent)
    tree = tree.remove_subtrees([[i] for i in range(0, be_index+1)])
    tree.init_indexes()
    regex = Tregex("NNS", "pluralNN", Condition("!$,,", Tregex("DT|PRP$|POS|CD", None)), Condition("!$+", Tregex("NN|NNS|NNP|NNPS",  None)), Condition(">", Tregex("NP|NML", "object", Condition("!<<", Tregex("NP|NML", None)))), Condition("!>>", Tregex("S", None)))
    matchLevel = 100
    introductions = []
    # we take every low-level NP and extract the last plural noun
    for pluralmatch in findMatches(regex, tree, tree):
        if len(pluralmatch["object"]) > matchLevel:
            continue
        matchLevel = len(pluralmatch["object"])
        pluralNN = tree.get_node(pluralmatch["pluralNN"])
        obj = tree.get_node(pluralmatch["object"])
        parent = tree.get_node(pluralmatch["pluralNN"][:-1])
        if parent == obj:
            insert_index = pluralmatch["pluralNN"]
        else:
            insert_index = pluralmatch["pluralNN"][:-1]+[0]

        insert_first_letter = tree.get_node(insert_index).get_words()[0][0]
        if insert_first_letter.lower() in {"a","e","i","o","u"}:
            determiner = "an"
        else:
            determiner = "a"
        noun = pluralNN.get_words()[0]
        lemma = singularize_noun(noun)
        if parent == obj:
            tree = tree.replace_subtree(Node("NP", Node("DT", Node(determiner)), Node("NN", Node(lemma))), pluralmatch["pluralNN"])
        else:
            tree = tree.replace_subtree(Node("NN", Node(lemma)), pluralmatch["pluralNN"])
            introductions.append((Node("DT", Node(determiner)), insert_index))
    
    for intro in introductions[::-1]:
        tree = tree.introduce_subtree(intro[0], intro[1])
    tree.init_indexes()

    for child in parent[:be_index+1][::-1]:
        tree = tree.introduce_subtree(child, [0])
    tree.init_indexes()
    return tree

def singularize_quantif(tree, marker):
    #tree is the main NP
    #marker gives the indexes that contains the marker in the current tree
    NP_parent = tree.get_node(marker[:-1])
    NP_parent.init_indexes()
    tree = tree.replace_subtree(singularize_NP(NP_parent), marker[:-1])

    tree.init_indexes()
    tree = singularize_verbs(tree)
    return tree

#Universal Quantification
def universal_marks(tree):
    oldTree = Node("r")
    universal_markers = {"every","everyone", "everything", "everybody", "everywhere", "all", "each", "any", "anyone", "anything", "anybody", "anywhere", "people"}
    UnivMarkerTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", "univNP", Condition("<<", Tregex(None, "marker", Condition("<", Tregex("every|everyone|everything|everybody|everywhere|all|each|any|anyone|anything|anybody|anywhere|people", None)), Condition("!$,,", Tregex("RB", None)))))))), Condition("!>>", Tregex("Implies|If|if", None)), Condition("!>>", Tregex("Universal", None)))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(UnivMarkerTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            univNP = root.get_node(match["univNP"][len(match["Root"]):])

            #Rewrite the tree as an implication
            #Could add singularization here
            
            #Left side: "X is a" (+ marker) + univNP-marker
            marker = tree.get_node(match["marker"]).get_words()[0]
            if marker[-4:] == "body" or marker[-3:] == "one" or marker.lower() == "people":
                property = "person"
            elif marker[-5:] == "thing":
                property = "thing"
            elif marker[-5:] == "where":
                property = "place"
            else:
                property = ""
            new_univNP = copy.deepcopy(univNP)
            new_univNP.init_indexes()
            new_univNP = singularize_quantif(new_univNP, match["marker"][len(match["univNP"]):])
            if not marker.lower() == "people":
                new_univNP = new_univNP.remove_subtrees([match["marker"][len(match["univNP"]):]])
            new_univNP = new_univNP.replace_word("people", "person")

            #Directly rewriting the left side
            if len(new_univNP) == 0:
                new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X"))), Node("VP", Node("VBZ", Node("is")), Node("NP", Node("DT", Node("a")), Node("NN", Node(property)))), Node(".", Node("."))))
            else:
                if property == "" or marker.lower() == "people":
                    new_univNP.init_indexes()
                    if new_univNP.get_words()[0].lower() not in {"a", "an"}:
                        leftest_NPTregex = Tregex("NP", "leftest", Condition("!<<", Tregex("NP", None)))
                        leftestMatch = next(findMatches(leftest_NPTregex, new_univNP, new_univNP))
                        first_word = new_univNP.get_node(leftestMatch["leftest"]).get_words()[0]
                        if first_word.lower() not in {"a", "an"}:
                            first_letter = first_word[0]
                            if first_letter.lower() in {"a","e","i","o","u"}:
                                determiner = "an"
                            else:
                                determiner = "a"
                            new_univNP = new_univNP.introduce_subtree(Node("DT", Node(determiner)), leftestMatch["leftest"]+[0])

                    new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X"))), Node("VP", Node("VBZ", Node("is")), new_univNP), Node(".", Node("."))))
                else:
                    if new_univNP.get_words()[0] in {"a", "an"}: #should not happen
                        new_univNP.init_indexes()
                        leftest = new_univNP.get_leftest_leaf()
                        new_univNP = new_univNP.remove_subtrees([leftest.indexes])
                    new_univNP = new_univNP.introduce_subtree(Node("NP", Node("DT", Node("a")), Node("NN", Node(property))), [0])
                    new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X"))), Node("VP", Node("VBZ", Node("is")), new_univNP), Node(".", Node("."))))
            #Right side: start of sentence + "X" + end of sentence
            new_right = root.replace_subtree(Node("NP", Node("NNP", Node("X"))), match["univNP"][len(match["Root"]):])
            new_right.init_indexes()
            new_right = singularize_verbs(new_right)
            #words = " ".join(new_right.get_words())
            #new_right = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))

            new_root = Node("Universal", Node("Implies", new_left, new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def neg_universal_marks(tree):
    oldTree = Node("r")
    NegUnivMarkerTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", "univNP", Condition("<<", Tregex(None, "marker", Condition("<", Tregex("no|nobody|nowhere|nothing|noone", None)))))))), Condition("!>>", Tregex("Implies|If|if", None)), Condition("!>>", Tregex("Universal", None)))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(NegUnivMarkerTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            univNP = root.get_node(match["univNP"][len(match["Root"]):])

            #Rewrite the tree as an implication
            #Could add singularization here
            
            #Left side: "X is a" (+ marker) + univNP-marker
            marker = tree.get_node(match["marker"]).get_words()[0]
            if marker[-4:] == "body" or marker[-3:] == "one":
                property = "person"
            elif marker[-5:] == "thing":
                property = "thing"
            elif marker[-5:] == "where":
                property = "place"
            else:
                property = ""

            new_univNP = copy.deepcopy(univNP)
            new_univNP.init_indexes()
            new_univNP = singularize_quantif(new_univNP, match["marker"][len(match["univNP"]):])
            try:
                dotNextToNo = tree.get_node(match["marker"][:-1])[match["marker"][-1]+1].get_words() == ["."]
            except IndexError:
                dotNextToNo = False
            if dotNextToNo:
                tree.init_indexes()
                continue
            
            try:
                oneNextToNo = tree.get_node(match["marker"][:-1])[match["marker"][-1]+1].get_words() == ["one"]
            except IndexError:
                oneNextToNo = False
            if oneNextToNo:
                new_univNP = new_univNP.remove_subtrees([match["marker"][len(match["univNP"]):-1]+[match["marker"][-1]+1]])
                property = "person"
            new_univNP = new_univNP.remove_subtrees([match["marker"][len(match["univNP"]):]])

            if len(new_univNP) == 0:
                new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X"))), Node("VP", Node("VBZ", Node("is")), Node("NP", Node("DT", Node("a")), Node("NN", Node(property)))), Node(".", Node("."))))
            else:
                if property == "":
                    new_univNP.init_indexes()
                    if new_univNP.get_words()[0].lower() not in {"a", "an"}: #Not sure we should check this, it should never happen
                        leftest_NPTregex = Tregex("NP", "leftest", Condition("!<<", Tregex("NP", None)))
                        leftest_match = next(findMatches(leftest_NPTregex, new_univNP, new_univNP))
                        first_letter = new_univNP.get_node(leftest_match["leftest"]).get_words()[0][0]
                        if first_letter.lower() in {"a","e","i","o","u"}:
                            determiner = "an"
                        else:
                            determiner = "a"
                        new_univNP = new_univNP.introduce_subtree(Node("DT", Node(determiner)), leftest_match["leftest"]+[0])

                    new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X"))), Node("VP", Node("VBZ", Node("is")), new_univNP), Node(".", Node("."))))
                else:
                    if new_univNP.get_words()[0] in {"a", "an"}: #should not happen, but just in case
                        new_univNP.init_indexes()
                        leftest = new_univNP.get_leftest_leaf()
                        new_univNP = new_univNP.remove_subtrees([leftest.indexes])
                    new_univNP = new_univNP.introduce_subtree(Node("NP", Node("DT", Node("a")), Node("NN", Node(property))), [0])
                    new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X"))), Node("VP", Node("VBZ", Node("is")), new_univNP), Node(".", Node("."))))

            #Right side: negate(start of sentence + "X" + end of sentence)
            new_right = root.replace_subtree(Node("NP", Node("NNP", Node("X"))), match["univNP"][len(match["Root"]):])
            new_right.init_indexes()
            new_right = singularize_verbs(new_right)
            #words = " ".join(new_right.get_words())
            #new_right = Node("ROOT", Node("Not", Node.fromstring(spacyModel.get_tree(words))))
            new_right = Node("Not", new_right)

            new_root = Node("Universal", Node("Implies", new_left, new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def subj_plural_universal(tree):
    oldTree = Node("r")
    #NNS/NNPS as a subject, when there is no determiner before them in the NP
    #The VP should be in simple present tense
    SubjPluralUnivTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", "mainNP", Condition("<<", Tregex("NNS|NNPS", "pluralNN", Condition("!>>", Tregex("VP", None)))), Condition("$..", Tregex("VP", "mainVP")))))), Condition("!>>", Tregex("Implies|If|if", None)), Condition("!>>", Tregex("Universal", None)))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(SubjPluralUnivTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            mainVP = root.get_node(match["mainVP"][len(match["Root"]):])
            new_mainVP = copy.deepcopy(mainVP)
            new_mainVP.init_indexes()
            if not simple_present(new_mainVP):
                tree.init_indexes()
                continue

            mainNP = root.get_node(match["mainNP"][len(match["Root"]):])
            new_mainNP = copy.deepcopy(mainNP)
            new_mainNP.init_indexes()
            if not indefinite_plural_NP(new_mainNP, match["pluralNN"][len(match["mainNP"]):]):
                tree.init_indexes()
                continue
            
            #Rewrite the tree as an implication
            #Could add singularization here

            #Left side: "X is a" + mainNP
            new_mainNP = singularize_NP(new_mainNP)
            
            new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X"))), Node("VP", Node("VBZ", Node("is")), new_mainNP), Node(".", Node("."))))

            #Right side: start of sentence + "X" + end of sentence
            new_right = root.replace_subtree(Node("NP", Node("NNP", Node("X"))), match["mainNP"][len(match["Root"]):])
            new_right.init_indexes()
            new_right = singularize_verbs(new_right)
            new_right = Node("ROOT", new_right)

            new_root = Node("Universal", Node("Implies", new_left, new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def indef_singular_be_universal(tree):
    oldTree = Node("r")
    #"A mmmm is a nnnn" is transformed into "X is a mmmm" -> "X is a nnnn"
    IndefSingularBeUnivTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<", Tregex("NP", "mainNP1", Condition("<+NP", Tregex("DT", None, Condition("<", Tregex("a|an", None)), Condition("$..", Tregex("NN|NNP", None)), Condition("!$,,", Tregex("DT|PRP$|WDT|WP$|POS|CD", None)))))), Condition("<", Tregex("VP", None, Condition("<<", Tregex("is", None)), Condition("<", Tregex("NP", None, Condition("<+NP", Tregex("DT", None, Condition("<", Tregex("a|an", None)), Condition("$..", Tregex("NN|NNP", None)), Condition("!$,,", Tregex("DT|PRP$|WDT|WP$|POS|CD", None)))))))))), Condition("!>>,1", Tregex("Implies|If|if", None)), Condition("!>>", Tregex("Universal", None)))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        for match in findMatches(IndefSingularBeUnivTregex, tree, tree):
            root = tree.get_node(match["Root"])
            root.init_indexes()
            mainNP1 = root.get_node(match["mainNP1"][len(match["Root"]):])

            if not be_sentence(root):
                tree.init_indexes()
                continue

            #Rewrite the tree as an implication
            #Could add singularization here

            #Left side: "X is a" + mainNP1
            new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X")), Node("VP", Node("VBZ", Node("is")), mainNP1))))

            #Right side: start of sentence + "X" + end of sentence
            new_right = root.replace_subtree(Node("NP", Node("NNP", Node("X"))), match["mainNP1"][len(match["Root"]):])
            #words = " ".join(new_right.get_words())
            #new_right = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))
            new_right = Node("ROOT", new_right)

            new_root = Node("Universal", Node("Implies", new_left, new_right))
            tree = tree.replace_subtree(new_root, match["Root"])
            tree.init_indexes()
    return tree

def find_universal_quantification(tree, spacyModel):
    tree = universal_embedded_existential_marks(tree, spacyModel)
    tree = universal_embedded_existential_indef_singular_subj(tree, spacyModel)
    tree = universal_marks(tree)
    tree = neg_universal_marks(tree)
    tree = subj_plural_universal(tree)
    tree = indef_singular_be_universal(tree)

    return tree

def find_embedded_universal_quantification(tree):
    tree = universal_embedded_existential_marks(tree)
    tree = universal_embedded_existential_indef_singular_subj(tree)
    return tree

def universal_embedded_existential_marks(tree, spacyModel):
    oldTree = Node("r")
    existential_markers = {"some", "someone", "something", "somebody", "somewhere", "people"}
    UniversalEmbeddedExistMarkerTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", "univNP", Condition("<<", Tregex(None, "marker", Condition("<", Tregex("some|someone|something|somebody|somewhere|people", None)), Condition("!$,,", Tregex("RB", None)))))))), Condition(">>,1", Tregex("Implies|If|if", "Imply")), Condition("!>>", Tregex("Universal", None)))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        try:
            for match in findMatches(UniversalEmbeddedExistMarkerTregex, tree, tree):
                root = tree.get_node(match["Root"])
                root.init_indexes()
                univNP = root.get_node(match["univNP"][len(match["Root"]):])

                #Rewrite the tree as an implication
                #Could add singularization here
                
                #Left side: "X is " (+ marker) + univNP-marker
                marker = tree.get_node(match["marker"]).get_words()[0]
                if marker[-4:] == "body" or marker[-3:] == "one" or marker.lower() == "people":
                    property = " person "
                elif marker[-5:] == "thing":
                    property = " thing "
                elif marker[-5:] == "where":
                    property = " place "
                else:
                    property = " "
                
                new_univNP = univNP.remove_subtrees([match["marker"][len(match["univNP"]):]])
                if len(new_univNP) == 0:
                    words = "X is a" + property + "."
                else:
                    words = "X is a" + property + " ".join(new_univNP.get_words()).replace("people", "person")
                new_left = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))

                #Left side - 2: start of sentence + "X" + end of sentence
                new_left2 = root.replace_subtree(Node("NP", Node("NNP", Node("X"))), match["univNP"][len(match["Root"]):])
                #words = " ".join(new_left2.get_words())
                #new_left2 = Node("ROOT", Node.fromstring(spacyModel.get_tree(words)))

                if len(new_univNP) == 0:
                    new_left = new_left2
                else:
                    new_left = Node("And", new_left, new_left2)
                new_left = tree.get_node(match["Imply"]+[0]).replace_subtree(new_left, match["Root"][len(match["Imply"])+1:])

                #Right side: Replace coreferences with "X" on the right side of the implication
                right_side = tree.get_node(match["Imply"]+[1])
                right_side.init_indexes()
    
                #run coreference: d'abord prendre la phrase entière, run coref et récupérer le cluster pour l'élément quantifié (mainNP1). Pour tous les trucs qui sont pas la head, récupérer leur forme textuelle et les retrouver (tregex?) pour les remplacer par X.
                doc = spacyModel.process(" ".join(tree.get_node(match["Imply"]).get_words()))
                cluster = get_coref_mentions(doc, " ".join(univNP.get_words()))
                for elem in cluster:
                    elemTregex = Tregex(None, "elem", Condition("text=", Tregex(elem, None)))
                    replace_indexes = []
                    for cluster_match in findMatches(elemTregex, right_side, right_side):
                        right_side = right_side.replace_subtree(Node("NNP", Node("X")), cluster_match["elem"])
                new_right = copy.deepcopy(right_side)

                new_imply = Node("Universal", Node("Implies", new_left, new_right))

                tree = tree.replace_subtree(new_imply, match["Imply"])
                tree.init_indexes()
                break #not optimal, but better to break the loop to avoid errors due to reindexing of the tree

        except IndexError:
            tree.init_indexes()
            continue
    return tree

def universal_embedded_existential_indef_singular_subj(tree, spacyModel):
    oldTree = Node("r")
    #IF "a mmmm bbb" THEN "ccc" is transformed into IF ("X is a mmmm" AND "X bbb") THEN "ccc"(X)
    IndefSingularBeUnivTregex = Tregex("ROOT", "Root", Condition("<<:", Tregex("S", None, Condition("<<", Tregex("NP", "mainNP1", Condition("!>>", Tregex("VP", None)), Condition("<+NP", Tregex("DT", None, Condition("<", Tregex("a|an", None)), Condition("$..", Tregex("NN|NNP", None)), Condition("!$,,", Tregex("DT|PRP$|WDT|WP$|POS|CD", None)))))))), Condition(">>,1", Tregex("Implies|If|if", "Imply")), Condition("!>>", Tregex("Universal", None)))
    while oldTree != tree:
        oldTree = copy.deepcopy(tree)
        try:
            for match in findMatches(IndefSingularBeUnivTregex, tree, tree):
                root = tree.get_node(match["Root"])
                root.init_indexes()
                mainNP1 = root.get_node(match["mainNP1"][len(match["Root"]):])

                #Rewrite the tree as an implication
                #Could add singularization here

                #Left side - 1: "X is a" + mainNP1
                words = "X is " + " ".join(mainNP1.get_words())
                new_left = Node("ROOT", Node("S", Node("NP", Node("NNP", Node("X")), Node("VP", Node("VBZ", Node("is")), mainNP1))))

                #Left side - 2: start of sentence + "X" + end of sentence
                new_left2 = root.replace_subtree(Node("NP", Node("NNP", Node("X"))), match["mainNP1"][len(match["Root"]):])
                new_left = Node("And", new_left, new_left2)
                new_left = tree.get_node(match["Imply"]+[0]).replace_subtree(new_left, match["Root"][len(match["Imply"])+1:])

                #Right side: Replace coreferences with "X" on the right side of the implication
                right_side = tree.get_node(match["Imply"]+[1])
                right_side.init_indexes()

                #run coreference: d'abord prendre la phrase entière, run coref et récupérer le cluster pour l'élément quantifié (mainNP1). Pour tous les trucs qui sont pas la head, récupérer leur forme textuelle et les retrouver (tregex?) pour les remplacer par X.
                doc = spacyModel.process(" ".join(tree.get_node(match["Imply"]).get_words()))
                cluster = get_coref_mentions(doc, " ".join(mainNP1.get_words()))
                for elem in cluster:
                    elemTregex = Tregex("ROOT", "Root", Condition("<<", Tregex(None, "elem", Condition("text=", Tregex(elem, None)))))
                    for cluster_match in findMatches(elemTregex, right_side, right_side):
                        right_side = right_side.replace_subtree(Node("NNP", Node("X")), cluster_match["elem"])
                new_right = copy.deepcopy(right_side)

                new_imply = Node("Universal", Node("Implies", new_left, new_right))
                tree = tree.replace_subtree(new_imply, match["Imply"])
                tree.init_indexes()

                break #not optimal, but better to break the loop to avoid errors due to reindexing of the tree

        except IndexError:
            tree.init_indexes()
            continue
    return tree

def get_coref_mentions(doc, existentialNP):
    clusters = doc._.coref_clusters
    for cluster in clusters:
        spans = [doc.char_span(span[0],span[1]).text.lower() for span in cluster]
        if existentialNP.lower() in spans:
            return spans
    return [existentialNP]


#Main function
def process_tree(sentence, spacyModel, universal=True):
    tree = spacyModel.get_tree(sentence)
    tree = Node("ROOT", Node.fromstring(str(tree)))

    #tree = Node.fromstring(sentence)
    tree.init_indexes()

    # Apply processing functions iteratively until no change occurs
    while True:
        #print("starting loop")
        tree.clean_duplicates()
        tree.clean_tree()

        """print("-------")
        print("Let's start")
        print(tree)
        print("-------")"""

        old_tree = tree
        #Coordinate Clauses
        tree = CC_processing(tree, spacyModel)
        if tree != old_tree:
            continue

        tree = PunctSplit_processing(tree)
        if tree != old_tree:
            continue

        #Relative Clauses
        tree = NonResPrepClause_processing(tree)
        if tree != old_tree:
            continue

        tree = NonResWhereClause_processing(tree)
        if tree != old_tree:
            continue

        tree = NonResWhomClause_processing(tree)
        if tree != old_tree:
            continue

        tree = NonResWhoseClause_processing(tree)
        if tree != old_tree:
            continue

        tree = NonResWhoWhichClause_processing(tree)
        if tree != old_tree:
            continue

        #Subordinated Clauses
        tree = PrePurpAdvClauseIOT_processing(tree)
        if tree != old_tree:
            continue

        tree = PrePurpAdvClauseT_processing(tree)
        if tree != old_tree:
            continue

        tree = PreAdvClause_processing(tree)
        if tree != old_tree:
            continue

        #Universal Quantification function
        if universal:
            tree = find_universal_quantification(tree, spacyModel)
            if tree != old_tree:
                continue

        #Non-Restrictive Preposed Participial Phrases
        tree = PrePartPhrase_processing1(tree)
        if tree != old_tree:
            continue
        
        tree = PrePartPhrase_processing2(tree)
        if tree != old_tree:
            continue

        #Coordinate Verb Phrases
        tree = CoordVerbPhrase_cleaning(tree)
        if tree != old_tree:
            continue

        """tree = CoordVerbPhrase_cleaning2(tree)
        if tree != old_tree:
            continue"""
        

        tree = CoordVerbPhrase_processing(tree, spacyModel)
        if tree != old_tree:
            continue
        
        #Non-Restrictive Postposed Participial Phrases
        tree = NonResPostPartPhrase_processing1(tree)
        if tree != old_tree:
            continue
        tree = NonResPostPartPhrase_processing2(tree)
        if tree != old_tree:
            continue

        #Post-Posed Adverbial Clauses
        tree = PostPurpAdvClauseIOT_processing(tree)
        if tree != old_tree:
            continue
        tree = PostPurpAdvClauseT_processing(tree)
        if tree != old_tree:
            continue

        #Attribution Clauses
        tree = RepAttrPre_processing(tree)
        if tree != old_tree:
            continue

        tree = PostAttrClause_processing(tree)
        if tree != old_tree:
            continue

        tree = RepAttrPost_processing(tree)
        if tree != old_tree:
            continue
        
        #Embedded Participial Phrases
        tree = EmbPartPhrase_processing(tree)
        if tree != old_tree:
            continue

        #Restrictive Relative Clauses
        #For now we don't do anything. We might add this only within universal quantification if the subordinated NP is X

        #Restrictive Participial Phrases
        #For now we don't do anything. We might add this only within universal quantification if the subordinated NP is X


        #Coordinate Noun Phrases
        tree = CoordNounPhraseSubj_processing(tree, spacyModel)
        if tree != old_tree:
            continue

        tree = CoordNounPhraseSubj_cleaning(tree)
        if tree != old_tree:
            continue

        tree = CoordNounPhraseObj_processing(tree, spacyModel)
        if tree != old_tree:
            continue

        tree = CoordNounPhraseObj_cleaning(tree)
        if tree != old_tree:
            continue

        tree = CoordAdjPhrase_processing(tree)
        if tree != old_tree:
            continue

        tree = CoordAdjPhrase_cleaning(tree)
        if tree != old_tree:
            continue

        break
    return tree