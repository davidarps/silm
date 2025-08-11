import sys
import tqdm
from generate_underspecified_bracklang import make_grammar
import multiprocessing as mp
from multiprocessing import Pool

if __name__=="__main__":
    
    target_dataset_size = int(sys.argv[1])
    infilename = sys.argv[2]
    if len(sys.argv) > 3:
        lengths = sys.argv[3].split(",")
        min_length, max_length = int(lengths[0]), int(lengths[1])
    else:
        min_length = 6
        max_length = 84

    with open(infilename, "r") as f:
        lines = f.read().splitlines()
    lines = list(set(lines))
    print(lines[0])

    _, parser = make_grammar()

    def parallel_postproc_func(sent):
        
        sent = sent.split()
            
        if min_length <= len(sent) <= max_length:
            parses = list(parser.parse(sent))
            if len(parses) != 1: 
                print("Error, wrong number of parses: ")
                print(parses)
                return ""
            
            parse = parses[0]
            treestr = " ".join(str(parse).replace("\n","").split())
            sentstr = " ".join(sent)
            return '{"text": "'+ sentstr + '", "tree": "' + treestr + '"}'
        else:
            print("wrong length: ", sent)
            return ""

    outfilename = f"{infilename}.json"

    nproc = max(mp.cpu_count() - 3, 1)
    print(len(lines))
    
    #exit()
    with Pool(nproc) as p:
        postproc_lines = p.map(parallel_postproc_func, lines, chunksize=2000)

    print(len(postproc_lines))
    print(postproc_lines[0])
    with open(outfilename, "w") as f:
        for line in tqdm.tqdm(postproc_lines):
            f.write(line)
            f.write("\n")

   