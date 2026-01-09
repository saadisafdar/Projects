#!/usr/bin/env python3
import os, glob
def scan(root_dirs=("train","valid")):
    counts = {}
    found = set()
    for d in root_dirs:
        lbl_dir = os.path.join(d, "labels")
        if not os.path.isdir(lbl_dir):
            print(f"Missing: {lbl_dir}")
            continue
        for f in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(f) as fh:
                for line in fh:
                    line=line.strip()
                    if not line: continue
                    cls=int(line.split()[0])
                    counts[cls]=counts.get(cls,0)+1
                    found.add(cls)
    return counts, found
if __name__=="__main__":
    counts, found = scan()
    print("Class counts:", counts)
    print("Classes found:", sorted(found))
