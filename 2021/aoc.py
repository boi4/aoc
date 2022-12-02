#!/usr/bin/env python3
import os
import re
import sqlite3
import sys
from copy import deepcopy
from functools import reduce
from datetime import datetime
from shutil import copyfile

import requests


def day_1(input):
    k = [int(l) for l in input.splitlines()]
    k = [sum(l) for l in zip(k[:-2],k[1:-1],k[2:])] # comment for silver star
    return sum(b > a for a,b in zip(k[:-1],k[1:]))


def day_2(input):
    #l = [({"f":1,"d":0,"u":0}[line[0]]*int(line.split()[1]),{"f":0,"d":1,"u":-1}[line[0]]*int(line.split()[1])) for line in input.splitlines()]
    #return [sum(x) for x in zip(*l)][0] * [sum(x) for x in zip(*l)][1]
    a,b,_ = reduce(lambda acc,line: (acc[0] + int(line.split()[1])*(line[0]=="f"),
                                     acc[1] + int(line.split()[1])*(line[0]=="f")*acc[2],
                                     acc[2] + int(line.split()[1])*(["u","f","d"].index(line[0])-1)),
                   input.splitlines(),
                   (0,0,0))
    return a * b


def day_3(input):
    #m = list(zip(*[[int(c) for c in line] for line in input.splitlines()]))
    #a = int("".join([str(int(sum(k) > len(k)/2)) for k in m]),2)
    #return a * (((1<<len(m))-1) ^ a)
    m = [[int(c) for c in line] for line in input.splitlines()]
    toconsider = deepcopy(m)
    bitpos = 0
    while len(toconsider) > 1:
        b = int(sum([l[bitpos] for l in toconsider])>=len(toconsider)/2)
        toconsider = [l for l in toconsider if l[bitpos] == b]
        bitpos += 1
    ox = int("".join(str(c) for c in toconsider[0]),2)
    toconsider = deepcopy(m)
    bitpos = 0
    while len(toconsider) > 1:
        b = int(sum([l[bitpos] for l in toconsider])>=len(toconsider)/2)
        toconsider = [l for l in toconsider if l[bitpos] != b]
        bitpos += 1
    co2 = int("".join(str(c) for c in toconsider[0]),2)
    return co2 * ox
    #return a * (((1<<len(m))-1) ^ a)


def day_4(input):
    blocks = [block for block in input.split('\n\n') if block.strip()]
    drawings = [int(k) for k in blocks[0].split(",")]
    blocks = [[[int(l) for l in line.split()]for line in block.splitlines()] for block in blocks[1:]]
    drawn = []
#    for drawing in drawings:
#        drawn.append(drawing)
#        for block in blocks:
#            for row in block+list(zip(*block)):
#                if all(num in drawn for num in row):
#                    s = sum(num for r in block for num in r if num not in drawn)
#                    return  s * drawing
    blocksdone = [False]*len(blocks)
    for drawing in drawings:
        drawn.append(drawing)
        for i,block in enumerate(blocks):
            for row in block+list(zip(*block)):
                if all(num in drawn for num in row):
                    blocksdone[i] = True
                    if all(blocksdone):
                        s = sum(num for r in block for num in r if num not in drawn)
                        return  s * drawing


def day_5(input):
    from collections import Counter
    l = [[(int(p.split(",")[0]),int(p.split(",")[1])) for p in l.split(" -> ")] for l in input.splitlines()]
    print(l)
    min_x = min(p[0] for a in l for p in a)
    max_x = max(p[0] for a in l for p in a)
    min_y = min(p[1] for a in l for p in a)
    max_y = max(p[1] for a in l for p in a)
    d = 0
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            c = 0
            for p1,p2 in l:
                if p1[0] == p2[0] or p1[1] == p2[1]:
                    if p1[0] <= x <= p2[0] and p1[1] <= y <= p2[1]:
                        c += 1
            print(c, end=" ")
            if c > 1:
                d += 1
        print("")
    print(d)

def day_8(input):
    pass
    return 8

    #c = Counter([(x,y) for (p1,p2) in l for x in range(p1[0],p2[0]+1) for y in range(p1[1],p2[1]+1)])
    #print(c)
    #print(len([1 for v in c.values() if v>1]))


def get_session_cookie():
    ffpath = os.path.expanduser("~/.mozilla/firefox")
    base,subs,_ = next(os.walk(ffpath))
    subdirs = [f"{base}/{sub}" for sub in subs]
    cookiefiles = [f"{s}/cookies.sqlite" for s in subdirs if os.path.isfile(f"{s}/cookies.sqlite")]
    sql = """
SELECT value
FROM moz_cookies
WHERE host='.adventofcode.com' AND name='session';
"""
    results = []
    for db in cookiefiles:
        copyfile(db, "/tmp/aoccookies.sqlite")
        with sqlite3.connect("/tmp/aoccookies.sqlite") as con:
            results += [row[0] for row in con.cursor().execute(sql)]
    results = list(set(results))
    if len(results) == 0:
        print("Please login in at adventofcode.com with one of your firefox profiles")
        sys.exit(1)
    if len(results) > 1:
        print("Warning: Found multiple session cookies")
    return results[0]


def submit_solution(session, day, level, answer):
    """
    return True if this solves the problem for first time
    """
    data = {"level": level, "answer": answer}
    url = f"https://adventofcode.com/2021/day/{day}/answer"

    print(f"Submitting solution for day {day}, level {level}...")
    print()
    r = session.post(url, data=data)

    pattern = r"<article><p>(.*?)</p></article>"
    m =re.search(pattern, r.text, re.DOTALL)
    if m is not None:
        result = m.group(1).strip()
    else:
        result = r.text.strip()

    completed = "You don't seem to be solving the right level.  Did you already complete it?"
    locked = "Please don't repeatedly request this endpoint before it unlocks!"
    wrong = "That's not the right answer"
    recently = "You gave an answer too recently;"
    correct = "That's the right answer!"

    print("adventofcode.com says:")

    if result.startswith(completed):
        print("\033[38;5;3m", end="") # yellow
        print(f"> {result}")
        print("\033[0;0m", end="") # reset color to normal
        return False
    elif result.startswith(locked):
        print("\033[38;5;3m", end="") # yellow
        print(f"> {result}")
        print("\033[0;0m", end="") # reset color to normal
        return False
    elif result.startswith(wrong):
        print("\033[38;5;1m", end="") # red
        i1 = len(wrong)
        i2 = result.find(".")
        if result[i1] == ";":
            i1 += 1
        print(result[:i1], end="")
        print("\033[38;5;3m", end="") # yellow
        print(result[i1:i2], end="")
        print("\033[38;5;1m", end="") # red
        print(result[i2:])
        print("\033[0;0m", end="") # reset color to normal
        return False
    elif result.startswith(recently):
        pattern = r"You have (?:(?P<minutes>\d+)m )?(?P<seconds>\d+)s left to wait"
        m =re.search(pattern, result)
        if m is None:
            print("\033[38;5;3m", end="") # yellow
            print(result)
            print("\033[0;0m", end="") # reset color to normal
        else:
            i1,i2 = m.span()
            i1 += len("You have ")
            i2 -= len(" left to wait")
            print("\033[38;5;3m", end="") # yellow
            print(result[:i1], end="")
            print("\033[38;5;1m", end="") # red
            print(result[i1:i2], end="")
            print("\033[38;5;3m", end="") # yellow
            print(result[i2:])
            print("\033[0;0m", end="") # reset color to normal
        return False
    elif result.startswith(correct):
        print("\033[38;5;2m", end="") # green
        print(f"> {result}")
        print("\033[0;0m", end="") # reset color to normal
        return True
    else:
        print(result)
        return False


def solve(day):
    s = requests.Session()
    s_cookie = get_session_cookie()
    s.cookies.set("session", s_cookie, domain=".adventofcode.com")
    
    inputpath = "inputs"
    os.makedirs(inputpath, exist_ok=True)
    fname = f"{inputpath}/{day}.txt"
    if not os.path.exists(fname):
        with open(fname, "wb+") as f:
            url = f"https://adventofcode.com/2021/day/{day}/input"
            f.write(s.get(url).content)

    with open(fname) as f:
        answer = globals()[f"day_{day}"](f.read())
        if answer is not None:
            print("Trying level 1")
            ok = submit_solution(s, day, 1, answer)
            if not ok:
                print()
                print("Trying level 2")
                submit_solution(s, day, 2, answer)


def main():
    now = datetime.now()
    if datetime(2021,12,1,0,0,0) <= now < datetime(2021,12,26,0,0,0):
        solve(now.day)
    else:
        print("Advent has ended you fool 🎅")

if __name__ == "__main__":
    main()
