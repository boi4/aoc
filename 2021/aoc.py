#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime


def day_1(input):
    k = [int(l) for l in input.split('\n') if l]
    k = [sum(l) for l in zip(k[:-2],k[1:-1],k[2:])] # uncomment for silver star
    return sum(b > a for a,b in zip(k[:-1],k[1:]))



def main():
    inputpath = "inputs"
    os.makedirs(inputpath, exist_ok=True)
    day = datetime.now().day
    fname = f"{inputpath}/{day}.txt"
    if not f"{day}.txt" in os.listdir(inputpath):
        subprocess.run(["zsh", "-ic", f"wgetcook https://adventofcode.com/2021/day/{day}/input -O {fname}"])

    with open(fname) as f:
        print(globals()[f"day_{day}"](f.read()))


if __name__ == "__main__":
    main()
