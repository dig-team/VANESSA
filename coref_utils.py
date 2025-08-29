def resolve_coref_custom(doc):
    no_coref_trigger = {"who", "which", "that", "where", "when", "All", "all"}
    resolved = list(tok.text_with_ws for tok in doc)
    clusters = doc._.coref_clusters
    all_spans = [span for cluster in clusters for span in cluster]
    for cluster in clusters:
        #print("cluster", cluster)
        capital_trigger = False
        indices = get_span_noun_indices(doc,cluster)
        #print("indices", indices)
        if indices:
            mention_span, mention = get_cluster_head(doc, cluster, indices)
            #print("mention", mention_span)
            if mention[0] == 0 and mention_span[0].tag_ not in {"NNP", "NNPS"}:
                capital_trigger = True
            if any([trig in mention_span.text.split() for trig in no_coref_trigger]):
                #print("MECHANT", mention_span)
                continue
            for coref in cluster:
                #print("coref", coref, mention, all_spans)
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    #print("entered resolution")
                    resolved = core_logic_part(doc, coref, resolved, mention_span, capital_trigger)
    
    return "".join(resolved)

def get_span_noun_indices(doc, cluster):
    """
    > Get the indices of the spans in the cluster that contain at least one noun or proper noun
    :param doc: Doc
    :param cluster: List[Tuple]
    :return: A list of indices of spans that contain at least one noun or proper noun.
    """
    spans = [doc.char_span(span[0],span[1]) for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [
        i for i, span_pos in enumerate(spans_pos) if any(pos in span_pos for pos in ["NOUN", "PROPN"])
    ]
    if span_noun_indices == []:
        span_noun_indices = [
        i for i, span_pos in enumerate(spans_pos) if "PRON" in span_pos and any(pron in spans[i].text.lower() for pron in ["someone", "something", "something", "somebody", "somewhere"])
        ]
    return span_noun_indices

def get_cluster_head(doc, cluster, noun_indices):
    """
    > Given a spaCy Doc, a coreference cluster, and a list of noun indices, return the head span and its start and end
    indices
    :param doc: the spaCy Doc object
    :type doc: Doc
    :param cluster: a list of Tuples, where each tuple is a char indices of token in the document
    :type cluster: List[Tuple]
    :param noun_indices: a list of indices of the nouns in the cluster
    :type noun_indices: List[int]
    :return: The head span and the start and end indices of the head span.
    """
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    #if there is a comma in the head span, then we want to keep only the part of the span before the comma
    if "," in doc.char_span(head_start,head_end).text:
        head_end = head_start + doc.char_span(head_start,head_end).text.index(",")
    head_span = doc.char_span(head_start,head_end)
    return head_span, [head_start, head_end]

def is_containing_other_spans(span, all_spans):
    """
    It returns True if there is any span in all_spans that is contained within span and is not equal to span
    :param span: the span we're checking to see if it contains other spans
    :type span: List[int]
    :param all_spans: a list of all the spans in the document
    :type all_spans: List[List[int]]
    :return: A list of all spans that are not contained in any other span.
    """
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

def core_logic_part(document, coref, resolved, mention_span, capital_trigger):
    """
    If the last token of the mention is a possessive pronoun, then add an apostrophe and an s to the mention.
    Otherwise, just add the last token to the mention
    :param document: Doc object
    :type document: Doc
    :param coref: List[int]
    :param resolved: list of strings, where each string is a token in the sentence
    :param mention_span: The span of the mention that we want to replace
    :return: The resolved list is being returned.
    """
    char_span = document.char_span(coref[0],coref[1])
    final_token = char_span[-1]
    final_token_tag = str(final_token.tag_).lower()
    test_token_test = False
    if capital_trigger:
        mention_span_text = mention_span.text[0].lower() + mention_span.text[1:]
    else:
        mention_span_text = mention_span.text
    for option in ["PRP$", "POS", "BEZ"]:
        if option.lower() in final_token_tag:
            test_token_test = True
            break
    if test_token_test:
        resolved[char_span.start] = mention_span_text + "'s" + final_token.whitespace_
    else:
        if mention_span_text in ["someone", "something", "something", "somebody", "somewhere"] and char_span.text.lower() not in ["someone", "something", "something", "somebody", "somewhere"]:
            resolved[char_span.start] = "X" + final_token.whitespace_
        else:
            resolved[char_span.start] = mention_span_text + final_token.whitespace_
    for i in range(char_span.start + 1, char_span.end):
        resolved[i] = ""
    return resolved