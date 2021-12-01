#!/usr/bin/env python3 
f = open("inputs/input03.txt", "r")

l = []

for line in f:
    l.append(line)

for line in l:
    for line2 in l:
        if line != line2 and len(line) == len(line2):
            numwr = 0
            r = ""
            for i in range(len(line)):
                if line[i] != line2[i]:
                    numwr += 1
                    r = line[:i]+line[i+1:]
                if numwr >1:
                    break
            if numwr == 1:
                print(r)

