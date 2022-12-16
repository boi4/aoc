#!/usr/bin/env python3
import functools
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


def day_10(input):
    orig = np.array([0 if l == "noop" else int(l[5:]) for l in 
    ("noop\n" + input.strip()).replace("addx","noop\naddx").split("\n")])
    a = 1 + np.cumsum(orig)
    b = a * np.arange(1,a.shape[-1]+1)
    #return sum(b[19::40])
    display = np.zeros(shape=(6,40), dtype=bool)
    for row in range(6):
        for col in range(40):
            spritepos = a[row*40+col]
            if col in [spritepos-1,spritepos,spritepos+1]:
                display[row,col] = True
                print("#",end="")
            else:
                print(" ",end="")
        print()
    return "RBPARAGF" # hardcoded


def day_11(input):
    start_pattern = r"Starting items: (.*)$"
    items = [[int(b) for b in l.split(",")]
                   for l in re.findall(start_pattern, input, re.MULTILINE)]
    operation_pattern = r"Operation: new = old ([+*]) (\d+|old)$"
    operations = re.findall(operation_pattern, input, re.MULTILINE)
    test_pattern = r"""Test: divisible by (\d+)
\s+If true: throw to monkey (\d+)
\s+If false: throw to monkey (\d+)
"""
    tests = [(int(a),int(b),int(c)) for (a,b,c) in re.findall(test_pattern, input, re.MULTILINE)]


    task1 = False
    modulus = product(a[0] for a in tests)
    num_inspected = [0] * len(items)
    for round in range(20 if task1 else 10000):
        #print(f"Round {round}\n==================")
        for monkey in range(len(items)):
            #print(f"Monkey {monkey}:")
            op,num = operations[monkey]
            divisor,truemonkey,falsemonkey = tests[monkey]
            for item in items[monkey]:
                #print(f"  Monkey inspects an item with a worry level of {item}")
                # inspect item
                if num == "old":
                    arg = item
                else:
                    arg = int(num)
                if op == "*":
                    item *= arg
                else:
                    item += arg

                # get bored of item
                if task1:
                    item //= 3
                else:
                    item = item % modulus
                #print(f"  Now it's {item}")

                # check where to throw item
                if item % divisor == 0:
                    items[truemonkey].append(item)
                else:
                    items[falsemonkey].append(item)
                num_inspected[monkey] += 1 
            items[monkey] = []
        #from pprint import pprint
        #pprint(items)
    #print(num_inspected)
    return product(sorted(num_inspected, reverse=True)[:2])


def day_12(input):
    heightmap = np.array([[ord(c) for c in line] for line in input.strip().split("\n")])

    rx,cx = np.where(heightmap == ord('S'))
    heightmap[rx,cx] = ord('a')
    start = (rx[0],cx[0])

    rx,cx = np.where(heightmap == ord('E'))
    heightmap[rx,cx] = ord('z')
    end = (rx[0],cx[0])

    heightmap -= ord('a')
    maxint = np.iinfo(np.int64).max
    costgrid = np.ones(shape=heightmap.shape, dtype=np.int64) * maxint

    def get_neighbors(row,col):
        return list(zip(*[(row+dr,col+dc) for (dr,dc) in [(-1,0),(1,0),(0,-1),(0,1)]
                          if 0 <= (row+dr) < costgrid.shape[0] and 0 <= (col+dc) < costgrid.shape[1]
                             and heightmap[row,col] >= heightmap[row+dr,col+dc] - 1]))


    # create dijkstra costgrid, can be obviously made more efficient by spiraling around end
    costgrid[end[0],end[1]] = 0
    for _ in range(product(costgrid.shape)):
        for row in range(costgrid.shape[0]):
            for col in range(costgrid.shape[1]):
                neighbors = get_neighbors(row,col)
                if len(neighbors):
                    vals = costgrid[neighbors[0],neighbors[1]]
                    mval = min(vals)
                    if mval < maxint and mval < costgrid[row,col]:
                        costgrid[row,col] = mval + 1

    def walk(pos):
        history = [pos]
        while pos != end:
            all_neighbors = get_neighbors(*pos)
            if len(all_neighbors) == 0:
                return None
            neighbors = [[],[]]
            for i in range(len(all_neighbors[0])):
                n = (all_neighbors[0][i],all_neighbors[1][i])
                if n not in history:
                    neighbors[0].append(n[0])
                    neighbors[1].append(n[1])
            if len(neighbors[0]) == 0:
                return None
            costs = costgrid[neighbors[0],neighbors[1]]
            i = np.argmin(costs)
            if costs[i] > costgrid[pos[0],pos[1]]:
                return None
            pos = (neighbors[0][i],neighbors[1][i])
            history.append(pos)
        return history
    

    #return len(walk(start)) - 1

    positions = [(row,col) for row in range(costgrid.shape[0]) for col in range(costgrid.shape[1]) if heightmap[row,col] == 0]
    walks = []
    for pos in positions:
        w = walk(pos)
        if w is not None:
            walks.append(walk(pos))
    return min([len(walk) - 1 for walk in walks])


def day_13(input):
    pairs = input.strip().split("\n\n")

    def mysplit(line, tok):
        lvl = 0
        parts = []
        i = 0
        part = ""
        while i < len(line):
            if lvl == 0 and line[i:].startswith(tok):
                parts.append(part)
                part = ""
                i += len(tok)
            else:
                c = line[i]
                if c == "[":
                    lvl += 1
                elif c == "]":
                    lvl -= 1
                part += c
                i += 1
        parts.append(part)
        return parts

    def parse(line):
        res = []
        if line[1:-1] == "":
            return res
        for tok in mysplit(line[1:-1], ","):
            res.append(parse(tok) if tok[0] == "[]"[0] else int(tok)) # [] bc pyright bug
        return res

    def compare(l,r):
        if type(l) == type(r) == type([]):
            l = l.copy()
            r = r.copy()
            while True:
                if len(l) == len(r) == 0:
                    return 0
                if len(l) == 0:
                    return 1
                if len(r) == 0:
                    return -1
                litem = l.pop(0)
                ritem = r.pop(0)
                c = compare(litem,ritem)
                if c != 0:
                    return c
        if type(l) == type(r) == type(1):
            if l == r: return 0
            if l < r: return 1
            if l > r: return -1
        if type(l) == type([]) and type(r) == type(1):
            return compare(l,[r])
        if type(r) == type([]) and type(l) == type(1):
            return compare([l],r)


    #indices = []
    #for i,pair in enumerate(pairs):
    #    left,right = pair.split("\n")
    #    left,right = parse(left),parse(right)
    #    if compare(left,right) == 1:
    #        indices.append(i+1)
    #return sum(indices)
    lines = [parse(line) for pair in pairs for line in pair.split("\n")]
    lines.append(parse("[[2]]"))
    lines.append(parse("[[6]]"))
    lines.sort(key=functools.cmp_to_key(compare), reverse=True)
    lines = [str(l) for l in lines]
    return (lines.index("[[2]]") + 1) * (lines.index("[[6]]") + 1)


def day_14(input):
    task1 = False
    structures = [[(int(a.split(",")[0]),int(a.split(",")[1])) for a in l.split(" -> ")]
                  for l in input.strip().split("\n")]


    # turn structures into list of blocked coordinates
    structures_np = []
    for s in structures:
        for a,b in zip(s[:-1],s[1:]):
            if a > b:
                a,b = b,a
            if a[0] == b[0]:
                l = np.stack((
                        np.arange(a[1],b[1]+1),
                        np.ones((b[1]-a[1]+1,),dtype=np.int64) * a[0],
                 ))
            else:
                l = np.stack((
                        np.ones((b[0]-a[0]+1,),dtype=np.int64) * a[1],
                        np.arange(a[0],b[0]+1),
                ))
            structures_np.append(l)
    blocks = np.concatenate(structures_np, axis=1)

    sand = np.array([0,500])
    if task1:
        # set make origin to 0,0
        dx = np.min(blocks[1])
        blocks[1] -= dx
        sand[1] -= dx

        # determine cave dimensions
        height = np.max(blocks[0]) + 1
        width  = np.max(blocks[1]) + 1

        cave = np.zeros(shape=(height,width), dtype=np.int64)
        cave[blocks[0],blocks[1]] = 1
    else:
        height = np.max(blocks[0]) + 1 + 2

        # make enough space for floor
        dx = np.min(blocks[1]) - height
        blocks[1] -= dx
        sand[1] -= dx

        width  = np.max(blocks[1]) + 1 + height
        cave = np.zeros(shape=(height,width), dtype=np.int64)

        cave[blocks[0],blocks[1]] = 1

        # add floor
        cave[-1] = 1

    cnt = 0
    pos = sand.copy()
    while True:
        if pos[0] + 1 >= height:
            break
        elif cave[pos[0]+1,pos[1]] == 0:
            pos[0] += 1
        elif pos[1] - 1 < 0:
            break
        elif cave[pos[0]+1,pos[1]-1] == 0:
            pos += np.array((1,-1))
        elif pos[1] + 1  >= width:
            break
        elif cave[pos[0]+1,pos[1]+1] == 0:
            pos += np.array((1,1))
        elif np.all(pos == sand): # task 2, blocking start
            cnt += 1
            break
        else: # bottom of cave
            cnt += 1
            cave[pos[0],pos[1]] = 2
            pos = sand.copy()
    return cnt





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
    solve(14)

