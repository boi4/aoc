#!/usr/bin/env python3
import functools
import os
import re
import sqlite3
import sys
from collections import defaultdict
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


def day_15(input):
    l = re.findall(r"Sensor at x=([-\d]+), y=([-\d]+): closest beacon is at x=([-\d]+), y=([-\d]+)$", input, re.MULTILINE)
    sensor_data = np.array([[int(a) for a in b] for b in l]).T
    distances = np.abs(sensor_data[:2]-sensor_data[2:]).sum(axis=0)


    # task 1
    y = 2000000
    undetected_ranges = set()
    for i in range(sensor_data.shape[-1]):
        distance = distances[i]
        x = sensor_data[0,i]
        xdistance = distance - np.abs(sensor_data[1,i] - y)
        if xdistance < 0:
            continue
        undetected_ranges |= set(range(x-xdistance, x+xdistance+1))
    for i in range(sensor_data.shape[-1]):
        if sensor_data[3,i] == y and sensor_data[2,i] in undetected_ranges:
            undetected_ranges.remove(sensor_data[2,i])
    #return len(undetected_ranges)


    # task 2
    # walk along edges
    for i in range(len(distances)):
        distance = distances[i]
        sensor_x = sensor_data[0,i]
        sensor_y = sensor_data[1,i]

        xvals = np.arange(sensor_x - distance - 1, sensor_x + distance + 2)
        xvals = np.concatenate((xvals, xvals[-2:0:-1]))

        yvals = np.arange(sensor_y - distance - 1, sensor_y + distance + 2)
        ly = len(yvals)
        yvals = np.concatenate((yvals[ly//2:], yvals[-2:0:-1], yvals[:ly//2]))
        for x,y in zip(xvals,yvals):
            if not (0 <= x < 4000000 and 0 <= y < 4000000): continue
            check = distances - (np.abs(sensor_data[1] - y) + np.abs(sensor_data[0] - x))
            if np.all(check < 0): return 4000000*x + y


def day_16(input):
    pattern = r"Valve (.*) has flow rate=(\d+); tunnels? leads? to valves? (.*)$"
    data = re.findall(pattern, input, re.MULTILINE)

    tunnels = {}
    frs = {}
    for name,fr,ts in data:
        tunnels[name] = [n.strip() for n in ts.split(",")]
        frs[name] = int(fr)

    relevant_tunnels = [k for (k,v) in frs.items() if v > 0]
    relevant_tunnels.insert(0, "AA")
    D = np.ones((len(relevant_tunnels),len(relevant_tunnels)), dtype=np.int64) * -1
    reachable = {}

    for t in tunnels.keys():
        reachable[t] = set((t,))

    for nsteps in range(0,len(tunnels)+1):
        for i in range(len(relevant_tunnels)):
            t = relevant_tunnels[i]
            new_reachable = set()
            for conn in reachable[t]:
                if conn in relevant_tunnels:
                    j = relevant_tunnels.index(conn)
                    if D[i,j] == -1:
                        D[i,j] = nsteps
                new_reachable |= set(tunnels[conn])
            reachable[t] = new_reachable

    relevant_frs = np.empty(shape=(D.shape[-1],), dtype=np.int64)
    for i in range(len(relevant_tunnels)):
        relevant_frs[i] = frs[relevant_tunnels[i]]




    def compute_water(time_left, cur_path):
        # no need to compute 16!, we abort once time is out
        # roughly ~1 million paths we check this way
        cur_max = 0
        for next in range(D.shape[0]):
            if next in cur_path:
                continue
            tl = time_left - D[cur_path[-1],next] - 1
            if tl >= 1:
                m = compute_water(tl, cur_path + [next])
                m += tl * relevant_frs[next]
                if m > cur_max:
                    cur_max = m
        return cur_max

    #return compute_water(30, [0])

    def get_paths(time_left, cur_path, blacklist=[]):
        # we also allow shorter_paths
        res = [cur_path]
        for next in range(D.shape[0]):
            if next in cur_path or next in blacklist:
                continue
            tl = time_left - D[cur_path[-1],next] - 1
            if tl >= 1:
                res += get_paths(tl, cur_path + [next], blacklist)
        return res

    def compute_path(path, tl=30):
        res = 0
        tl = tl
        for prev,next in zip(path[:-1],path[1:]):
            tl -= D[prev,next]
            tl -= 1
            res += relevant_frs[next]*tl
        return res

    ps = get_paths(26, [0])
    max_seen = 0
    for p in ps:
        ele_ps = get_paths(26, [0], blacklist=p)
        press = compute_path(p, 26)
        for ele_p in ele_ps:
            ele_press = compute_path(ele_p, 26)
            if press+ele_press > max_seen:
                max_seen = press+ele_press

    return max_seen


def day_17(input):
    # setup shapes
    shapes = """
####

.#.
###
.#.

..#
..#
###

#
#
#
#

##
##
""".strip().split("\n\n")
    shapes = [np.array([[".#".index(c) for c in line] for line in reversed(shape.split("\n"))]) for shape in shapes]


    # setup wind
    wind_directions = input.strip()

    # setup chamber
    num_rocks = 1000000000000
    max_shape_height = max(s.shape[0] for s in shapes)
    chamber_width = 7
    #chamber_height = num_rocks * max_shape_height
    chamber_height = 10000 * max_shape_height

    chamber = np.zeros(shape=(chamber_height,chamber_width), dtype=np.int64)
    chamber[0] = 1

    pile_height = 0
    cur_rock = 0
    cur_wind = 0
    new_rock = True
    pos = (0,0)

    def print_chamber():
        lines = []
        for row in range(pile_height+10):
            s = "".join([".#"[l] for l in chamber[row]])
            s = "|" + s + "|"
            lines.append(list(s))
        if not new_rock:
            r = shapes[cur_rock]
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    if r[i,j] == 1:
                        if pos[0]+i < len(lines):
                            lines[pos[0]+i][pos[1]+j+1] = "@"
        lines = list(reversed(lines))
        print("\n".join(["".join(l) for l in lines]))
        print()

    c = 0
    iter = 0
    observed = {}
    additional_height = 0
    cycle_found = False
    while c < num_rocks:
        if new_rock:
            # new rock gets placed
            pos = (pile_height + 4,2)
            new_rock = False
            

            if not cycle_found:
                # record floor state
                reached_rock = np.ones((chamber.shape[1],)) * -1
                i = 0
                while np.any(reached_rock == -1):
                    for j,t in enumerate(chamber[pile_height-i]==1):
                        if t and reached_rock[j] == -1:
                            reached_rock[j] = i
                    i += 1

                # this encodes the complete current state
                key = (cur_wind,cur_rock,tuple(reached_rock))

                # check in our lookup dict
                if key in observed:
                    cycle_found = True
                    c_old,pile_old = observed[key]

                    cycle_len = c - c_old
                    cycle_start = c_old

                    print(f"{c}: Found cycle: start: {cycle_start}, length: {cycle_len}")
                    blocks_left = (num_rocks - c)
                    print(f"{c=},{c_old=}")
                    print(f"{blocks_left=}")
                    print(f"{cycle_len=}")
                    cycles_skipped = blocks_left // cycle_len
                    print(f"{cycles_skipped=}")
                    additional_height = cycles_skipped * (pile_height - pile_old)
                    print(f"{additional_height=}")
                    num_rocks = blocks_left % cycle_len
                    c = 0
                else:
                    observed[key] = (c,pile_height)
        elif iter % 2 == 0:
            # check for collision
            collision_detected = False
            r = shapes[cur_rock]
            for row in range(r.shape[0]):
                for col in range(r.shape[1]):
                    if r[row,col]:
                        if pos[0] + row - 1 < 0:
                            collision_detected = True
                            break
                        if chamber[pos[0]+row-1,pos[1]+col]:
                            collision_detected = True
                            break

            if collision_detected:
                #print("Collision detected")
                # update chamber
                r = shapes[cur_rock]
                chamber[pos[0]:pos[0]+r.shape[0],pos[1]:pos[1]+r.shape[1]] += r
                new_rock = True
                cur_rock = (cur_rock + 1) % len(shapes)
                pile_height = np.argmin(np.any(chamber==1, axis=1)) - 1
                c += 1
                #print(f"New {pile_height=}")
                iter -= 1 # don't count collision as a round
            else:
                pos = (pos[0]-1,pos[1])
        else:
            dx = [-1,1]["<>".index(wind_directions[cur_wind])]
            #print("wind:", dx)
            r = shapes[cur_rock]
            collision_detected = False
            for row in range(r.shape[0]):
                for col in range(r.shape[1]):
                    if r[row,col]:
                        if pos[1] + col + dx < 0 or pos[1] + col + dx >= chamber.shape[1]:
                            collision_detected = True
                            break
                        if chamber[pos[0]+row,pos[1]+col+dx]:
                            collision_detected = True
                            break
            if not collision_detected:
                #print("wind applied")
                pos = (pos[0],pos[1]+dx)
            cur_wind = (cur_wind + 1) % len(wind_directions)

        #print(iter)
        iter = (iter + 1) % 2
        #print_chamber()
    return pile_height + additional_height



def day_18(input):
    cubes = [[int(a) for a in l.split(",")] for l in input.strip().split("\n")]
    cubes = np.array(cubes)
    cubes += 1 # add buffer around all
    space = np.zeros(shape=np.max(cubes, axis=0)+2, dtype=np.int64)
    space[cubes[:,0],cubes[:,1],cubes[:,2]] = 1

    count = 0
    count += np.sum(((space[1:] + space[:-1]) ==  2))
    count += np.sum((space[:,1:] + space[:,:-1]) ==  2)
    count += np.sum((space[:,:,1:] + space[:,:,:-1]) ==  2)
    count = 6*len(cubes)-2*count
    #return count


    # create dijkstra costgrid

    maxint = np.iinfo(np.int64).max
    costgrid = np.ones(shape=space.shape, dtype=np.int64) * maxint
    costgrid[0] = 0
    costgrid[:,0] = 0
    costgrid[:,:,0] = 0
    costgrid[-1] = 0
    costgrid[:,-1] = 0
    costgrid[:,:,-1] = 0

    def get_neighbors(row,col,depth):
        return list(zip(*[(row+dr,col+dc,depth+dd)
                          for (dr,dc,dd) in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
                          if 0 <= (row+dr) < costgrid.shape[0] and 0 <= (col+dc) < costgrid.shape[1]
                          and 0 <= (depth+dd) < costgrid.shape[2]]))

    #for _ in range(np.max(costgrid.shape) + cubes.shape[0]+10):
    deltas = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
    from tqdm import tqdm
    for _ in tqdm(range(product(costgrid.shape))):
        for row in range(1,costgrid.shape[0]-1):
            for col in range(1,costgrid.shape[1]-1):
                for depth in range(1,costgrid.shape[2]-1):
                    if space[row,col,depth] == 0:
                        neighbors = deltas + np.array([row,col,depth])
                        vals = costgrid[neighbors[:,0],neighbors[:,1],neighbors[:,2]]
                        mval = np.min(vals)
                        if mval < maxint and mval < costgrid[row,col,depth]:
                            costgrid[row,col,depth] = mval + 1

    costgrid[cubes[:,0],cubes[:,1],cubes[:,2]] = 1

    inner_space = np.array((costgrid == maxint), dtype=np.int64)
    inner_count = 0
    inner_count += np.sum(((inner_space[1:] + inner_space[:-1]) ==  2))
    inner_count += np.sum((inner_space[:,1:] + inner_space[:,:-1]) ==  2)
    inner_count += np.sum((inner_space[:,:,1:] + inner_space[:,:,:-1]) ==  2)
    inner_count = 6*np.sum(inner_space)-2*inner_count
    print(inner_count)

    count -= inner_count
    return count



def day_19(input):
#    input = """
#    Blueprint 1: Each ore robot costs 4 ore. Each clay robot costs 2 ore. Each obsidian robot costs 3 ore and 14 clay. Each geode robot costs 2 ore and 7 obsidian.
#Blueprint 2: Each ore robot costs 2 ore. Each clay robot costs 3 ore. Each obsidian robot costs 3 ore and 8 clay. Each geode robot costs 3 ore and 12 obsidian.
#    """
    # start with one ore-collecting robot
    bps = re.findall(r"Blueprint \d+: Each ore robot costs (\d+) ore. Each clay robot costs (\d+) ore. Each obsidian robot costs (\d+) ore and (\d+) clay. Each geode robot costs (\d+) ore and (\d+) obsidian.$", input, re.MULTILINE)
    assert(len(bps) == len(input.strip().split("\n")))

    # 0: ore, 1: clay, 2: obsidian, 3: geode
    bps = np.array([[[int(bp[0]),0,0,0],
                     [int(bp[1]),0,0,0],
                     [int(bp[2]),int(bp[3]),0,0],
                     [int(bp[4]),0,int(bp[5]),0]] for bp in bps], dtype=np.int32)


    def optimal_geode(bp, duration):
        # TODO: iterate over lengths and create sets
        # purge after every length
        maxes = np.max(bp, axis=0)
        l = 0
        queue = [(np.array([0,0,0,0], dtype=np.int32), np.array([1,0,0,0], dtype=np.int32), [])]
        #observed = set((((0,0,0,0),(1,0,0,0)),))
        while l < duration:
            print()
            print("======================")
            print("Minute", l+1)
            print("======================")
            new_queue = []
            for inventory, robots, path in queue:
                buyable = np.all((inventory - bp)>=0, axis=1)

                # if enough saved up, TODO
                #buyable[:3] &= ((inventory[:3] / (duration - l)) < (maxes[:3]-robots[:3]))

                for i in range(4):
                    if buyable[i] and (i == 3 or robots[i] < maxes[i]): # can build robot
                        rarg = robots.copy()
                        rarg[i] += 1
                        new_conf = (inventory-bp[i]+robots, rarg,  path + [i])
                        new_queue.append(new_conf)

                # not building a robot makes only sense if we are saving up for something
                addempty = False
                for i in range(4):
                    if not buyable[i] and np.all(robots[bp[i] > 0] > 0):
                        addempty = True
                        break
                if addempty:
                    new_conf = (inventory+robots, robots, path + [-1])
                    new_queue.append(new_conf)

            next = new_queue
            if l == 20:
                ii = np.random.permutation(len(next))
                for i in range(10):
                    print(next[ii[i]])

            # remove path
            for i in range(len(next)):
                next[i] = (next[i][0],next[i][1],[])


            if l == duration-1:
                # skip filtering on last run
                queue = new_queue
                break

            

            print("Same inventory filter")
            # if two solutions have the same inventory config but one has strictly better robots
            # remove the worse one
            m = np.array([[a for c in config for a in c] for config in next], dtype=np.int32)
            m = m[m[:, 3].argsort()]
            m = m[m[:, 2].argsort(kind='mergesort')] # keep order
            m = m[m[:, 1].argsort(kind='mergesort')]
            m = m[m[:, 0].argsort(kind='mergesort')]

            if m.shape[0] > 1:
                ixs = np.where(np.any(np.diff(m, axis=0)[:,0:4]!=0, axis=1))[0]
                ixs = [0] + list(ixs + 1) + [m.shape[0]]
            else:
                ixs = [0,1]
            print("max same:", np.max(np.diff(np.array(ixs))))

            def determine_next1(indices):
                next = []
                istart,iend = indices
                for i in range(istart,iend):
                    found_better = False
                    for j in range(istart,iend):
                        if i == j: continue
                        #assert(np.all(m[i,:4] == m[j,:4]))
                        if i < j and np.all(m[i,:8] == m[j,:8]):
                            found_better = True
                            break
                        if np.all(m[i,:8] <= m[j,:8]):
                            found_better = True
                            break
                    if not found_better:
                        next.append((m[i,:4],m[i,4:8],list(m[i,8:])))
                return next

            import time
            from pathos.multiprocessing import Pool
            c1 = time.monotonic()
            with Pool() as pool:
                nexts = pool.map(determine_next1, zip(ixs[:-1],ixs[1:]))
            next = [elem for nx in nexts for elem in nx]
            c2 = time.monotonic()
            print(f"took {c2-c1} seconds")



            print("Same robots filter")
            # if two solutions have the same robot config but one has strictly better
            # inventory, remove the worse one
            m = np.array([[a for c in config for a in c] for config in next], dtype=np.int32)
            m = m[m[:, 7].argsort(kind='mergesort')]
            m = m[m[:, 6].argsort(kind='mergesort')] # keep order
            m = m[m[:, 5].argsort(kind='mergesort')]
            m = m[m[:, 4].argsort(kind='mergesort')]

            if m.shape[0] > 1:
                ixs = np.where(np.any(np.diff(m, axis=0)[:,4:8]!=0, axis=1))[0]
                ixs = [0] + list(ixs + 1) + [m.shape[0]]
            else:
                ixs = [0,1]
            print("max same:", np.max(np.diff(np.array(ixs))))

            LIMIT = 1000
            def determine_next2(indices):
                istart,iend = indices
                tmp = m[istart:iend]

                if tmp.shape[0] == 1:
                    return [(tmp[0,:4],tmp[0,4:8],list(tmp[0,8:]))]

                if tmp.shape[0] > LIMIT:
                    # too expensive, just add all of them
                    next = []
                    for i in range(tmp.shape[0]):
                        next.append((tmp[i,:4],tmp[i,4:8],list(tmp[i,8:])))
                    return next

                # we group them by the ore number
                tmp = tmp[tmp[:, 0].argsort(kind='mergesort')]

                ixs = np.where(np.diff(tmp[:,0])!=0)[0]
                ixs = [0] + list(ixs + 1) + [tmp.shape[0]]

                next = []
                for istart,iend in zip(ixs[:-1],ixs[1:]):
                    for i in range(istart, iend):
                        found_better = False
                        for j in range(istart, tmp.shape[0]):
                            if i == j: continue
                            if i < j and np.all(tmp[i,:8] == tmp[j,:8]):
                                found_better = True
                                break
                            if np.all(tmp[i,:8] <= tmp[j,:8]):
                                found_better = True
                                break
                        if not found_better:
                            next.append((tmp[i,:4],tmp[i,4:8],list(tmp[i,8:])))

                return next

            import time
            from pathos.multiprocessing import Pool
            c1 = time.monotonic()
            with Pool() as pool:
                nexts = pool.map(determine_next2, zip(ixs[:-1],ixs[1:]))
            next = [elem for nx in nexts for elem in nx]
            c2 = time.monotonic()
            print(f"took {c2-c1} seconds")
            #assert(len(set([(a,b) for (a,b,_) in next])) == len(next))

            l += 1
            queue = next
            print("number of paths before filtering:", len(new_queue))
            print("number of paths after filtering:", len(queue))



        return queue
   


#    q = 0
#    for i,bp in enumerate(bps):
#        best_configs = optimal_geode(bp, 24)
#        m = np.array([invent[3] for invent,_,_ in best_configs])
#        print(bp)
#        print("#############################################", np.max(m))
#        print(i, np.max(m), (i+1)*np.max(m))
#        q += (i+1)*np.max(m)
#    #return q

    best_ones = []
    for i,bp in enumerate(bps[:3]):
        best_configs = optimal_geode(bp, 32)
        m = np.array([invent[3] for invent,_,_ in best_configs])
        best_ones.append(np.max(m))
        print("#############################################", i, np.max(m))
    return product(best_ones)


def day_20(input):
    task1 = False
    arr = np.array([int(line) for line in input.strip().split("\n")])
    N = len(arr)

    if not task1:
        arr *= 811589153

    pos = list(np.arange(N))

    for _ in range(1 if task1 else 10):
        for i in range(N):
            mov = arr[i]
            j = pos.index(i)
            if mov > 0:
                pos = pos[j+1:] + pos[:j]
                pos.insert(mov%(N-1), i)
            elif mov < 0:
                pos = pos[j+1:] + pos[:j]
                pos.insert(mov%(N-1), i)
    res = arr[pos]
    i = list(res).index(0)
    return res[(i+1000) % N] + res[(i+2000) % N] + res[(i+3000) % N]


def day_21(input):
#    input = """
#root: pppw + sjmn
#dbpl: 5
#cczh: sllz + lgvd
#zczc: 2
#ptdq: humn - dvpt
#dvpt: 3
#lfqf: 4
#humn: 5
#ljgn: 2
#sjmn: drzm * dbpl
#sllz: 4
#pppw: cczh / lfqf
#lgvd: ljgn * ptdq
#drzm: hmdt - zczc
#hmdt: 32
#    """
    pattern = r"^(....): ((....) (.) (....)|(\d+))$"
    l = [re.match(pattern, line).groups() for line in input.strip().split("\n")]
    monkeys = {}
    for monkey in l:
        name, _, arg1, op, arg2, num = monkey
        if num is not None:
            num = int(num)
            monkeys[name] = num
        else:
            monkeys[name] = (op, arg1, arg2)

    def eval_monkey(name):
        v = monkeys[name]
        if type(v) == type(0): return v
        op, arg1, arg2 = v
        match op:
            case "+": return eval_monkey(arg1) + eval_monkey(arg2)
            case "-": return eval_monkey(arg1) - eval_monkey(arg2)
            case "*": return eval_monkey(arg1) * eval_monkey(arg2)
            case "/": return eval_monkey(arg1) // eval_monkey(arg2)

    #return eval_monkey('root')

    def has_humn(name):
        if name == "humn": return True
        v = monkeys[name]
        if type(v) == int: return False
        _, arg1, arg2 = v
        return has_humn(arg1) or has_humn(arg2)

    def resolve(name):
        if name == "humn":
            # don't resolve human
            return

        v = monkeys[name]

        # already resolved
        if type(v) == int: return

        op, arg1, arg2 = v

        # resolve subtrees
        resolve(arg1)
        resolve(arg2)

        v1 = monkeys[arg1]
        v2 = monkeys[arg2]
        if type(v1) == type(0) and type(v2) == type(0) and "humn" not in [arg1,arg2]:
            match op:
                case "+": v = v1 + v2
                case "-": v = v1 - v2
                case "*": v = v1 * v2
                case "/": v = v1 // v2
            monkeys[name] = v


    side1 = monkeys['root'][1]
    side2 = monkeys['root'][2]
    if not has_humn(side2):
        side2,side1 = side1,side2

    resolve(side1)
    resolve(side2)

    def solve(lhs, rhs_eq):
        op,arg1,arg2 = rhs_eq
        if type(monkeys[arg1]) == type(0):
            # LNUM = RNUM op x
            arg1 = monkeys[arg1]
            match op:
                case "+": x = lhs - arg1
                case "-": x = arg1 - lhs
                case "*": x = lhs // arg1
                case "/": x = arg1 // lhs
            if arg2 == "humn":
                return x
            else:
                res = solve(x, monkeys[arg2])
                return res
        else:
            # LNUM = x op RNUM
            arg2 = monkeys[arg2]
            match op:
                case "+": x = lhs - arg2
                case "-": x = lhs + arg2
                case "*": x = lhs // arg2
                case "/": x = lhs * arg2
            if arg1 == "humn":
                return x
            else:
                res = solve(x, monkeys[arg1])
                return res

    return solve(monkeys[side1], monkeys[side2])

    #target = eval_monkey(side2)




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
    solve(21)

