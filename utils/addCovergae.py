import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import spacy
nlp = spacy.load('en_core_web_lg')

def computeCoverage(premise,hypothesis):
    doc_p = nlp(premise)
    doc_h = nlp(hypothesis)
    coverage = []

    for token1 in doc_h:
        max_sim = 0.001
        unique_items = set()
        for token2 in doc_p:
            if token2.text in unique_items:
                continue
            unique_items.add(token2.text)
            sim = abs(token1.similarity(token2))
            max_sim = max([max_sim,sim])
        coverage.append(float(max_sim))
    return coverage

def add_coverage(f_problems,f_out):

    problems = open(f_problems,'r').readlines()
    out = open("coverage_"+f_out,'w')

    for sample in problems:
        data = json.loads(sample)
        premises = data["premises"]
        choices = data["choices"]
        if isinstance(premises[0],str):
            premises = [premises]*len(choices)
        coverage = []
        for premise,choice in zip(premises,choices):
            per_choice_coverage =[]
            for sentence in premise:
                entity_coverage = computeCoverage(sentence,choice)
                per_choice_coverage.append(entity_coverage)
            coverage.append(per_choice_coverage)
        data["coverage"] = coverage

        out.write(json.dumps(data)+"\n")

add_coverage("../OBQA/mcq_abductive_train-7-test-obqa.jsonl","mcq_obqa_train.jsonl")
add_coverage("../OBQA/mcq_abductive_val-7-test-obqa.jsonl","mcq_obqa_dev.jsonl")