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
import numpy as np


def product(it):
    return reduce(lambda acc,x: acc * x, it, 1)


def day_1(input):
    elves_strs =  input.split("\n\n")
    elves = [[int(line) for line in elve_str.split("\n") if line] for elve_str in elves_strs]
    elves_sums = [sum(elve) for elve in elves]
    elves_sums.sort(reverse=True)
    return elves_sums[0] + elves_sums[1] + elves_sums[2]
    #return max(elves_sums)


def day_2(input):
    games = [line.split() for line in input.split("\n") if line.strip()]
    f = lambda g: [6,3,0][(1 + "ABC".index(g[0])-"XYZ".index(g[1]))%3] \
            + 1+"XYZ".index(g[1])
    games = [(g[0],"XYZ"[("ABC".index(g[0]) + "XYZ".index(g[1]) - 1)%3]) for g in games]
    return sum([f(g) for g in games])


def day_3(input):
#    lines = [(line[:len(line)//2],line[len(line)//2:])
#             for line in input.split("\n") if line.strip()]
#    parts = [list(set(a).intersection(set(b)))[0] for (a,b) in lines]
    lines = [line for line in input.split("\n") if line.strip()]
    lines = [(lines[3*i],lines[3*i+1],lines[3*i+2]) for i in range(len(lines)//3)]
    parts = [list(set(a).intersection(set(b)).intersection(set(c)))[0] for (a,b,c) in lines]
    return sum(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".index(p) for p in parts)


def day_4(input):
    lines = [[[int(a) for a in k.split("-")] for k in l.split(",")]
             for l in input.strip().split("\n")]
    #return sum((a[0] <= b[0] and b[1] <= a[1])
    #            or (b[0] <= a[0] and a[1] <= b[1])
    #           for (a,b) in lines)
    return sum((a[0] <= b[0] and a[1] >= b[0])
                or (a[0] >= b[0] and a[0] <= b[1])
               for (a,b) in lines)


def day_5(input):
    stacks, commands = input.strip().split("\n\n")
    rows = reversed([l[1::4] for l in stacks.split("\n")[:-1]])
    stacks = list(map(lambda x: "".join(x).strip(), zip(*rows)))
    for h,f,t in re.findall(r"move (\d+) from (\d+) to (\d+)", commands):
        #stacks[int(t)-1] += "".join(reversed(stacks[int(f)-1][-int(h):]))
        stacks[int(t)-1] += "".join(stacks[int(f)-1][-int(h):])
        stacks[int(f)-1] = stacks[int(f)-1][:-int(h)]
    return "".join(s[-1] for s in stacks)


def day_6(input):
    input = input.strip()
    N = 14
    a = [i+N for i in range(len(input)-N) if len(set(input[i:i+N])) == N]
    return a[0]
    #return a[0]


def day_7(input):
    cmds = [(l.split("\n")[0],"\n".join(l.split("\n")[1:]))
     for l in input.strip().split("$ ")[1:]]

    # build file system tree
    pwd = []
    fs = {}
    for cmd,output in cmds:
        match cmd.split():
            case ("cd", "/"):
                pwd = []
            case ("cd", ".."):
                pwd.pop(-1)
            case ("cd", dir):
                pwd.append(dir)
            case ("ls",):
                d = fs
                for k in pwd:
                    d = d[k]
                for a,b in [l.split() for l in output.split("\n") if l]:
                    if a == "dir" and b not in d:
                        d[b] = {}
                    else:
                        d[b] = int(a)
            case _:
                print(f"Invalid cmd {cmd}")
                return

    def dir_size(d, path, sizes):
        print(d,path)
        s = 0
        for k,v in d.items():
            if isinstance(v, dict):
                s += dir_size(v, path + [k], sizes)
            else:
                s += v
        sizes[tuple(path)] = s
        return s

    sizes = {}
    s = dir_size(fs, [], sizes)
    #return sum(v for v in sizes.values() if v <= 100000)
    free_space = 70000000 - s
    space_needed = 30000000 - free_space
    return min(v for v in sizes.values() if v >= space_needed)


def day_8(input):
    treemap = np.array([[int(a) for a in l] for l in input.strip().split("\n")])
    masks = []
    for tm in [treemap, treemap.T, treemap[::-1], treemap.T[::-1]]:
        highest = np.maximum.accumulate(tm, axis=0)
        higher = np.diff(highest, prepend=-1, axis=0) > 0
        masks.append(higher)

    is_visible = np.logical_or(masks[0], masks[1].T)
    is_visible = np.logical_or(is_visible, masks[2][::-1])
    is_visible = np.logical_or(is_visible, masks[3][::-1].T)
    #return np.sum(is_visible) # solve 1

    ms = 0
    for idx,v in np.ndenumerate(treemap):
        i,j = idx
        tocheck = [treemap[:i,j][::-1],treemap[i,:j][::-1],treemap[i,j+1:],treemap[i+1:,j]]
        score = product(len(tc) if np.all((tc-v)<0) else np.argmax((tc-v)>=0) + 1 for tc in tocheck)
        if score > ms: ms = score
    return ms

def day_9(input):
    #T_pos, H_pos = np.array([0,0]),np.array([0,0])
    #visited = set(((0,0),))
    #for d,s in [(l.split()[0],int(l.split()[1])) for l in input.strip().split("\n")]:
    #    for i in range(s):
    #        H_pos += np.array([[1,0,-1, 0],
    #                             [0,1, 0,-1],])[:,"URDL".index(d)]
    #        if np.max(np.abs(H_pos-T_pos)) == 2:
    #            T_dir = (H_pos-T_pos) / 2
    #            T_pos += np.int64(np.where(T_dir < 0, np.floor(T_dir), np.ceil(T_dir)))
    #        visited.add(tuple(T_pos))
    #return len(visited)

    positions = [np.array([0,0]) for _ in range(10)]
    visited = set(((0,0),))
    for d,s in [(l.split()[0],int(l.split()[1])) for l in input.strip().split("\n")]:
        for i in range(s):
            positions[0] += np.array([[1,0,-1, 0],
                                      [0,1, 0,-1],])[:,"URDL".index(d)]
            for i in range(1,len(positions)):
                prev, next = positions[i-1],positions[i]
                if np.max(np.abs(prev-next)) == 2:
                    T_dir = (prev-next) / 2
                    positions[i] += np.int64(np.where(T_dir < 0, np.floor(T_dir), np.ceil(T_dir)))
            visited.add(tuple(positions[-1]))
    return len(visited)


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

def solve(day):
    s = requests.Session()
    s_cookie = get_session_cookie()
    s.cookies.set("session", s_cookie, domain=".adventofcode.com")
    
    inputpath = "inputs"
    os.makedirs(inputpath, exist_ok=True)
    fname = f"{inputpath}/{day}.txt"
    if not os.path.exists(fname):
        with open(fname, "wb+") as f:
            url = f"https://adventofcode.com/2022/day/{day}/input"
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


def submit_solution(session, day, level, answer):
    """
    return True if this solves the problem for first time
    """
    data = {"level": level, "answer": answer}
    url = f"https://adventofcode.com/2022/day/{day}/answer"

    print("Answer:", answer)
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

def main():
    now = datetime.now()
    if datetime(2022,12,1,0,0,0) <= now < datetime(2022,12,26,0,0,0):
        solve(now.day)
    else:
        print("Advent has ended you fool 🎅")

if __name__ == "__main__":
    #main()
    solve(9)

