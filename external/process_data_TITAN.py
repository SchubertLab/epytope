#!/usr/bin/python3
import sys
fasta = []
with open(sys.argv[1], "r") as f:
    fasta.extend(f.readlines())
del fasta[:67]
del fasta[-47:]
with open(sys.argv[2], "w") as f:
    f.writelines(fasta)
