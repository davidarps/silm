# code for generating from CFG: https://stackoverflow.com/questions/603687/how-do-i-generate-sentences-from-a-formal-grammar
from random import choice
import traceback
from nltk import CFG, ChartParser
import sys
import tqdm
import argparse

recursion_limit = 256 # sys.getrecursionlimit() - 1
max_sent_len = 128

def produce(grammar, symbol, depth=0, cum_length=0):
    try:
        words = []
        productions = grammar.productions(lhs = symbol)
        if depth >= recursion_limit  or cum_length>(max_sent_len//2):
            productions = [pr for pr in productions if grammar.start() not in pr.rhs()]
            #productions = [pr for pr in productions if lex(pr.rhs())]
        production = choice(productions)
        for sym in production.rhs():
            if isinstance(sym, str):
                words.append(sym)
            else:
                words.extend(produce(grammar, sym, depth=depth+1, cum_length=len(words)))
        return words
    except Exception as e:
        print("depth: ", depth)
        print(traceback.format_exc())

def produce_multiple(grammar, k, min_length=0, max_length=1000):
    sents = set()
    generated_tokens = 0
    progressbar = tqdm.tqdm(total=k)

    while generated_tokens < k:
        sentlist = produce(gr, gr.start())
        if not(min_length <= len(sentlist) <= max_length):
            continue
        sent = " ".join(sentlist)
        if sent not in sents:
            sents.add(sent)
            new_toks = len(sent.split())
            progressbar.update(new_toks)
            generated_tokens += new_toks
    return list(sents)

def make_grammar():

    rules_basic = []
    rules_basic.append("S -> N V")
    rules_basic.append("S -> NSg V")
    rules_basic.append("S -> NSg VSg")
    rules_basic.append("S -> NPl V")
    rules_basic.append("S -> NPl VPl")
    rules_basic.append("S -> N VSg")
    rules_basic.append("S -> N VPl")
    rules_basic.append("S -> N S V")
    rules_basic.append("S -> NSg S VSg")
    rules_basic.append("S -> NPl S VPl")
    rules_basic.append("S -> NSg S V")
    rules_basic.append("S -> NPl S V")
    rules_basic.append("S -> N S VSg")
    rules_basic.append("S -> N S VPl")
    rules_basic.append("N -> 'n'")
    rules_basic.append("NSg -> 'nsg'")
    rules_basic.append("NPl -> 'npl'")
    rules_basic.append("V -> 'v'")
    rules_basic.append("VSg -> 'vsg'")
    rules_basic.append("VPl -> 'vpl'")

    rules_conj = [r + " S" for r in rules_basic if r.startswith("S")] # version with many rules but no structural ambiguity
    #rules_conj = ["S -> S S"] # less rules but many parses

    rules = rules_basic + rules_conj
    grammar = CFG.fromstring("\n".join(rules))
    parser = ChartParser(grammar)
    gr = parser.grammar() # grammar but slightly different. 
    return gr, parser


if __name__=="__main__":

    num_toks = int(sys.argv[1])
    out_file  = sys.argv[2]
    if len(sys.argv) > 3:
        lengths = sys.argv[3].split(",")
        min_length, max_length = int(lengths[0]), int(lengths[1])
    else:
        min_length = 6
        max_length = 84

    print(f"Generate {num_toks} tokens")
    print(f"Write output to {out_file}")

    gr, parser = make_grammar()
    sents = produce_multiple(gr, num_toks, min_length=min_length, max_length=max_length)
    print("With duplicates:    ", len(sents))
    sents = list(set(sents))
    print("Without duplicates: ", len(sents))

    print(f"Append to outfile {out_file}")
    with open(out_file, "w") as f:
        for sent in tqdm.tqdm(sents):
            f.write(sent)
            f.write("\n")
