#!/usr/bin/env python3

f = open("inputs/input03.txt", "r")

counttw = 0
counttr = 0

for line in f:
    twice = False
    thrice = False
    for i in range(len(line)):
        if line.count(line[i]) == 2:
            twice = True
        if line.count(line[i]) == 3:
            thrice = True
    if twice:
        counttw += 1
    if thrice:
        counttr += 1

print(counttw*counttr)
