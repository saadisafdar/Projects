#!/usr/bin/env python3
import os, glob, argparse

p = argparse.ArgumentParser()
p.add_argument("--map", required=True, help="mapping e.g. 0:2,1:4,2:1,3:5")
p.add_argument("--dirs", default="train,valid")
args = p.parse_args()

mapping = {}
for part in args.map.split(","):
    a, b = part.split(":")
    mapping[int(a)] = int(b)

dirs = args.dirs.split(",")

for d in dirs:
    lbl_dir = os.path.join(d, "labels")
    if not os.path.isdir(lbl_dir):
        continue

    for f in glob.glob(os.path.join(lbl_dir, "*.txt")):
        out_lines = []
        with open(f, "r") as fh:
            for L in fh:
                L = L.strip()
                if not L:
                    continue
                parts = L.split()
                old = int(parts[0])
                new = mapping.get(old, old)
                out_lines.append(" ".join([str(new)] + parts[1:]))

        with open(f, "w") as fh:
            fh.write("\n".join(out_lines) + "\n")

print("Remap finished.")
