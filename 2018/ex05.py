#!/usr/bin/env python3

import bisect

f = open("inputs/input05.txt", "r")
inp = [line for line in f]
f.close()

def binary_search(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        midval = a[mid]
        if midval < x:
            lo = mid+1
        elif midval > x:
            hi = mid
        else:
            return mid
    return -1

claimed = []

s = set()
k = 0
print("num lines: " + str(len(inp)))
for line in inp:
    if k % 10 == 0:
        print(k)

    i1 = line.find("@") + 1
    i2 = line.find(",")
    i3 = line.find(":")
    i4 = line.find("x")
    x0 = int(line[i1:i2])
    y0 = int(line[i2+1:i3])
    width = int(line[i3+1:i4])
    height = int(line[i4+1:])

    for i in range(width):
        for j in range(height):
            if binary_search(claimed, (x0+i,y0+j)) != -1:
                s.add((x0+i,y0+j))
            else:
                bisect.insort(claimed,(x0+i,y0+j))
    k+=1

print(len(s))
