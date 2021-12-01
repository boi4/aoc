#!/usr/bin/env python3
import sys


inputs = []
f = open("inputs/input02.txt", "r")
for line in f:
    if len(line)>1:
        inputs.append(int(line))
f.close()

l = []
acc = 0
i = 0
while True:
    print(i)
    print(acc)
    i+=1
    for val in inputs:
        l.append(acc)
        acc+=val
        if acc in l:
            print(acc)
            sys.exit(0)
