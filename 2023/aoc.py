#!/usr/bin/env python3
import functools
import os
import re
import sqlite3
import sys
from collections import defaultdict,Counter
from copy import deepcopy
from functools import reduce
from datetime import datetime
from shutil import copyfile
import itertools
from functools import lru_cache

import requests
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

YEAR=2023


def product(it):
    return reduce(lambda acc,x: acc * x, it, 1)

def flatten(l):
    return [item for sublist in l for item in sublist]

def day_1(inp):
    lines = inp.splitlines()
    digits = [[c for c in line if c.isdigit()] for line in lines]
    values = [int(d[0])*10+int(d[-1]) for d in digits]
    #return sum(values) # part 1

    names = [None, "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    digits = []
    for line in lines:
        d = []
        for i in range(len(line)):
            if line[i].isdigit():
                d.append(int(line[i]))
            elif any(line[i:].startswith(name) for name in names[1:]):
                d.append(names.index(next(n for n in names[1:] if line[i:].startswith(n))))
        digits.append(d)
    values = [int(d[0])*10+int(d[-1]) for d in digits]
    return sum(values)





def day_2(inp):
    lines = inp.splitlines()
    ms = [re.match(r"^Game (\d+): (.*)$", line) for line in lines]
    game_nos = [int(m.groups()[0]) for m in ms]
    games = [m.groups()[1].split(";") for m in ms]
    games = [[draw.split(", ") for draw in game] for game in games]
    games = [[{color.strip().split(" ")[1]: int(color.strip().split(" ")[0]) for color in draw} for draw in game] for game in games]

    color_caps = {'red': 12, 'green': 13, 'blue': 14}
    games_valid = [all(all(draw.get(color,0)<=cap for color,cap in color_caps.items()) for draw in game) for game in games]
    answer1 = sum(int(game_no) for game_no,is_valid in zip(game_nos,games_valid) if is_valid)

    min_needed_cubes = [[max(draw.get(color,0) for draw in game) for color in ['red', 'green', 'blue']] for game in games]
    answer2 = sum(product(game) for game in min_needed_cubes)

    return answer2


def day_3(inp):
    chars = list(str(c) for c in range(10)) + ["."]
    chars += list(sorted(set(c for c in inp.replace("\n","") if c not in chars)))
    lines = inp.splitlines()
    cmap = np.array([[chars.index(c) for c in line] for line in lines])
    rows,cols = cmap.shape

    is_symbol_map = (cmap > 10)
    is_symbol_map_padded = np.pad(is_symbol_map, 1, mode="constant", constant_values=False)
    is_next_to_symbol_map = (np.logical_or.reduce(
        np.stack([
            is_symbol_map_padded[2:,2:],
            is_symbol_map_padded[2:,1:-1],
            is_symbol_map_padded[2:,:-2],
            is_symbol_map_padded[1:-1,2:],
            is_symbol_map_padded[1:-1,1:-1],
            is_symbol_map_padded[1:-1,:-2],
            is_symbol_map_padded[:-2,2:],
            is_symbol_map_padded[:-2,1:-1],
            is_symbol_map_padded[:-2,:-2],
        ]),
    ) > 0)
    is_digit_map = (cmap < 10)

    numbers = []
    numbers_positions = []
    for row in range(0,rows):
        #l = re.finditer(r"(^|[^\d])(\d+)($|[^\d])", lines[row])
        l = re.finditer(r"\d+", lines[row])
        num_pos = []
        for m in l:
            number_start,number_end = m.span()
            number = int(m.group())
            if any(is_next_to_symbol_map[row,number_start:number_end]):
                numbers.append(number)
                num_pos.append((number_start,number_end))
        numbers_positions.append(num_pos)

    answer1 = sum(numbers)


    star_pos = np.argwhere((cmap == chars.index("*")))
    gear_ratios = []
    for star_row,star_col in star_pos:
        matches = set()
        for dy in range(-1,2):
            numbers_to_check = numbers_positions[star_row+dy]
            for number_start,number_end in numbers_to_check:
                if number_start <= star_col + 1 and star_col <= number_end:
                    matches.add((star_row+dy,number_start,number_end))
        matches = list(matches)
        if len(matches) == 2:
            gear_ratio = product(int(lines[m[0]][m[1]:m[2]]) for m in matches)
            gear_ratios.append(gear_ratio)


    answer2 = sum(gear_ratios)
    return answer2



def day_4(inp):
    lines = [line.split(":")[1].strip() for line in inp.splitlines()]
    wins = [set(int(a) for a in line.split("|")[0].split()) & set(int(a) for a in line.split("|")[1].split()) for line in lines]
    num_wins = [len(w) for w in wins]
    answer1 = sum(pow(2,n-1) if n else 0 for n in num_wins)

    num_cards = [1 for _ in range(len(lines))]

    for i in range(len(lines)):
        nw =  num_wins[i]
        nc = num_cards[i]
        for j in range(i+1,i+nw+1):
            num_cards[j] += nc

    answer2 = sum(num_cards)
    return answer2


def day_5(inp):
    chunks = inp.split("\n\n")
    info = {
        chunk.split(":")[0].strip(): np.array([int(a.strip()) for a in chunk.split(":")[1].strip().split()])
        for chunk in chunks
    }
    seeds = info["seeds"]
    maps = {map_name: ar.reshape(-1,3) for map_name,ar in info.items() if map_name != "seeds"}

    # sort all maps based on column 1
    for map_name in maps.keys():
        maps[map_name] = maps[map_name][np.argsort(maps[map_name][:,1])]

    conversions = defaultdict(list)
    for map_name in maps.keys():
        source,destination = map_name.replace(" map","").split("-to-")
        conversions[source].append(destination)

    # fill inner gaps in mapping with identity
    for map_name in maps.keys():
        new_map = []
        prev_end = 0
        for row in maps[map_name]:
            dest_start,src_start,range_len = row
            if src_start > prev_end:
                new_map.append([prev_end,prev_end,src_start-prev_end])
            new_map.append(row)
            prev_end = src_start + range_len
        maps[map_name] = np.array(new_map)

    # deterministic conversion path
    assert(len(l) == 1 for l in conversions.values())

    def transform(value):
        currency = "seed"
        while currency != "location":
            new_currency = conversions[currency][0]
            mapping = maps[f"{currency}-to-{new_currency} map"]
            indices = np.where((mapping[:,1] <= value) & (value < (mapping[:,1]+mapping[:,2])))[0]
            assert(len(indices) <= 1)
            if len(indices) != 0:
                value = mapping[indices[0],0] + value - mapping[indices[0],1]
            currency = new_currency
        return value


    def transform_range_rec(start, end, currency):
        """
        return a list of (start, end) tuples
        """
        if currency == "location":
            return [(start,end)]
        new_currency = conversions[currency][0]
        mapping = maps[f"{currency}-to-{new_currency} map"]

        # print(start,end,currency)


        # create new list of ranges
        ranges = []
        if start < mapping[0,1]:
            # first range
            ranges += [(start, min(end,mapping[0,1]))]
        if end > mapping[-1,1]+mapping[-1,2]:
            # last range
            ranges += [(max(start,mapping[-1,1]+mapping[-1,2]), end)]

        # could be optimized as mapping is sorted
        indices = np.where((mapping[:,1] < end) & ((start <= mapping[:,1]+mapping[:,2])))[0]

        for dest_start,src_start,range_len in mapping[indices]:
            relative_start = max(0,start-src_start)
            relative_end = min(end - src_start,range_len)
            ranges.append((dest_start+relative_start,dest_start+relative_end))

        # could merge ranges

        return flatten([transform_range_rec(s,e,new_currency) for s,e in ranges])


    def transform_range(start, end):
        return transform_range_rec(start, end, "seed")


    seed_locations = list(map(transform, seeds))
    answer1 = min(seed_locations)

    seeds = seeds.reshape(-1,2)
    seeds[:,1] = seeds[:,0] + seeds[:,1]

    #print(min(a[0] for a in transform_range(seeds[0,0], seeds[0,1])))
    seed_ranges = flatten([transform_range(s,e) for s,e in seeds])
    answer2 = min(a[0] for a in seed_ranges)
    return answer2



def day_6(inp):
    times = np.array([int(a) for a in inp.split("\n")[0].split(":")[1].strip().split()])
    distances = np.array([int(a) for a in inp.split("\n")[1].split(":")[1].strip().split()])

    cnts = []
    for i in range(len(times)):
        duration = times[i]
        record = distances[i]

        cnt = 0
        for possible_waittime in range(duration):
            new_distance = (duration - possible_waittime)*possible_waittime
            if new_distance > record: cnt += 1
        cnts.append(cnt)
    answer1 = product(cnts)


    duration = int(inp.split("\n")[0].split(":")[1].replace(" ", ""))
    record = int(inp.split("\n")[1].split(":")[1].replace(" ", ""))

    # formula: (duration-t)*t-record > 0 <> t^2 - duration*t + record < 0

    # solve for roots
    t0 = (duration + np.sqrt(duration**2 - 4*record))/2
    t1 = (duration - np.sqrt(duration**2 - 4*record))/2

    t0,t1 = min(t0,t1),max(t0,t1)

    answer2 = int(t1) - int(t0) # works only if t0 and t1 are not exactly integers
    return answer2



def day_7(inp):
    cards = "23456789TJQKA"
    hands = np.array([([cards.index(c) for c in a.split()[0]] + [int(a.split()[1])]) for a in inp.strip().splitlines()], dtype=np.float64)
    N = hands.shape[0]
    # add new column at front for value of hand
    hands = np.concatenate((np.zeros((hands.shape[0],1),dtype=hands.dtype),hands),axis=1)

    ctrs = [Counter(hands[i, 1:6]) for i in range(N)]
    # five of a kind -> 5, four of a kind -> 4, and so on
    hands[:,0] = [ctr.most_common()[0][1] for ctr in ctrs]
    # full house check
    hands[:,0] += [0.5*(len(ctr.most_common()) == 2 and ctr.most_common()[0][1]==3) for ctr in ctrs]
    # two pair check
    hands[:,0] += [0.5*(len(ctr.most_common()) == 3 and ctr.most_common()[0][1]==2) for ctr in ctrs]

    # sorty np array by rows, omg so complicated
    keys = tuple(hands[:, col] for col in range(hands.shape[1] - 1, -1, -1))
    idx = np.lexsort(keys)
    hands = hands[idx]

    answer1 = int(np.sum(hands[:,-1] * np.arange(1,N+1)))

    cards = "J23456789TQKA"
    hands = np.array([([cards.index(c) for c in a.split()[0]] + [int(a.split()[1])]) for a in inp.strip().splitlines()], dtype=np.float64)
    hands = np.concatenate((np.zeros((hands.shape[0],1),dtype=hands.dtype),hands),axis=1)
    ctrs = [Counter([a for a in hands[i, 1:6] if a != 0]) for i in range(N)] # exlucde jokers
    for ctr,hand in zip(ctrs, hands[:,1:6]):
        if len(ctr) == 0:
            best_val = cards.index("A")
        else:
            best_val = sorted([(b,a) for a,b in ctr.most_common()], reverse=True)[0][1]
        ctr.update([best_val]*np.sum(hand==0))
    # five of a kind -> 5, four of a kind -> 4, and so on
    hands[:,0] = [ctr.most_common()[0][1] for ctr in ctrs]
    # full house check
    hands[:,0] += [0.5*(len(ctr.most_common()) == 2 and ctr.most_common()[0][1]==3) for ctr in ctrs]
    # two pair check
    hands[:,0] += [0.5*(len(ctr.most_common()) == 3 and ctr.most_common()[0][1]==2) for ctr in ctrs]
    keys = tuple(hands[:, col] for col in range(hands.shape[1] - 1, -1, -1))
    idx = np.lexsort(keys)
    hands = hands[idx]

    answer2 = int(np.sum(hands[:,-1] * np.arange(1,N+1)))
    return answer2




def day_8(inp):
    instrs,rules = inp.strip().split("\n\n")
    instrs = ["LR".index(c) for c in instrs]
    rules = {line.split(" = (")[0]: line[:-1].split(" = (")[1].split(", ") for line in rules.splitlines()}

    # position = "AAA"
    # cnt = 0
    # for instr in itertools.cycle(instrs):
    #     position = rules[position][instr]
    #     cnt += 1
    #     if position == "ZZZ":
    #         break
    # answer1 = cnt

    def compute_trajectory(position):
        # return start_path + cycle
        i = 0
        visited = set()
        track = [(position,i)]
        visited.add((position,i))
        while True:
            position = rules[position][instrs[i]]
            i = (i+1) % len(instrs)
            if (position,i) in visited:
                break
            visited.add((position,i))
            track.append((position,i))

        offset = track.index((position,i))
        start_path = track[:offset]
        cycle = track[offset:]
        return start_path,cycle

    positions = [p for p in rules.keys() if p.endswith("A")]
    trajectories = [compute_trajectory(p) for p in positions]
    max_start_path_len = max(len(p) for p,_ in trajectories)

    # move for max_start_path_len steps
    i = 0
    cnt = 0
    for _ in range(max_start_path_len):
        cnt += 1
        positions = [rules[p][instrs[i]] for p in positions]
        i = (i+1) % len(instrs)
        if all(p.endswith("Z") for p in positions):
            break

    if all(p.endswith("Z") for p in positions):
        answer2 = cnt
        return answer2


    cycles = [cycle for _,cycle in trajectories]
    # move cycle so that current position aligns
    bin_cycles = []
    for cycle,position in zip(cycles,positions):
        offset = cycle.index((position,i))
        cycle = cycle[offset:] + cycle[:offset]
        bin_cycles.append([p.endswith("Z") for p,_ in cycle])

    # weirdly they all only have 1 z in their cycle
    # indices = [np.where(np.array(cycle))[0][0] for cycle in bin_cycles]
    cycle_lens = [len(cycle) for cycle in bin_cycles]
    answer2 = np.lcm.reduce(cycle_lens)

    return answer2


def day_9(inp):
    sequences = np.array([list(map(int, line.split())) for line in inp.strip().splitlines()])

    def extrapolate(seq):
        if np.all(seq == 0):
            return 0
        toreduce = seq[1:] - seq[:-1]
        return seq[-1] + extrapolate(toreduce)

    answer1 = np.sum([extrapolate(seq) for seq in sequences])

    def extrapolate2(seq):
        if np.all(seq == 0):
            return 0
        toreduce = seq[1:] - seq[:-1]
        return seq[0] - extrapolate2(toreduce)
    answer2 = np.sum([extrapolate2(seq) for seq in sequences])
    return answer2


def day_10(inp):
    inp = """
FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L
       """
    cmap = np.array([list(line) for line in inp.strip().splitlines()])

    # origin is top left
    connections = {
        '|': [(0,1),(0,-1)],
        '-': [(1,0),(-1,0)],
        'L': [(0,-1),(1,0)],
        'J': [(0,-1,),(-1,0)],
        '7': [(0,1),(-1,0)],
        'F': [(0,1),(1,0)],
    }

    def follow_pipe(startx,starty):
        x,y = startx,starty
        tile = cmap[y,x]
        # start in arbitrary connection
        dx,dy = connections[tile][0]
        visited = []
        while len(visited) == 0 or (x,y) != (startx,starty):
            visited.append((x,y))
            x,y = x+dx,y+dy
            if x < 0 or x >= cmap.shape[1] or y < 0 or y >= cmap.shape[0]:
                return None # failed, out of bounds
            tile = cmap[y,x]
            if tile not in connections.keys():
                return None # failed
            if (-dx,-dy) not in connections[tile]:
                return None # failed, don't connect
            dx,dy = [d for d in connections[tile] if d != (-dx,-dy)][0]

        return visited


    starty,startx = np.argwhere(cmap == "S")[0]

    for symbol in connections.keys():
        cmap[starty,startx] = symbol
        pipe = follow_pipe(startx,starty)
        if pipe is not None: break

    answer1 = int(np.ceil(len(pipe)/2))


    # # set all tiles not in pipe to .
    # for y in range(cmap.shape[0]):
    #     for x in range(cmap.shape[1]):
    #         if (x,y) not in pipe:
    #             cmap[y,x] = "."

    # add . padding around cmap
    cmap = np.pad(cmap, 1, mode="constant", constant_values=".")

    # build "bridges" where one can squeeze
    bridges = defaultdict(list)

    # horizontal bridges
    for y in range(cmap.shape[0]):
        x = 0
        while x < cmap.shape[1]-2:
            # pipe 'entrance'
            if cmap[y,x] == "." and cmap[y,x+1] in ["F", "L"]:
                c = cmap[y,x+1]
                startx = x
                x += 2
                # squeeze along pipe
                while x < cmap.shape[1] and cmap[y,x] == "-":
                    x += 1
                # pipe 'exit'
                if x < cmap.shape[1]-1 and cmap[y,x] == {"F": "7","L":"J"}[c] and cmap[y,x+1] == ".":
                    bridges[(startx,y)].append((x+1,y))
                    bridges[(x+1,y)].append((startx,y))
            x += 1

    # vertical bridges
    for x in range(cmap.shape[1]):
        y = 0
        while y < cmap.shape[0]-1:
            # pipe 'entrance'
            if cmap[y,x] == "." and cmap[y+1,x] in ["F", "7"]:
                c = cmap[y+1,x]
                starty = y
                y += 2
                # squeeze along pipe
                while y < cmap.shape[0] and cmap[y,x] == "|":
                    y += 1
                # pipe 'exit'
                if y < cmap.shape[0]-1 and cmap[y,x] == {"F": "L","7":"J"}[c] and cmap[y+1,x] == ".":
                    bridges[(x,starty)].append((x,y+1))
                    bridges[(x,y+1)].append((x,starty))
            y += 1

    # find all outside points and change them to "O"
    tovisit = set([(0,0)])
    while len(tovisit) > 0:
        x,y = tovisit.pop()
        if cmap[y,x] == "O":
            continue
        cmap[y,x] = "O"

        # add all bridge ends to tovisit
        for x2,y2 in bridges[(x,y)]:
            if cmap[y2,x2] == ".":
                tovisit.add((x2,y2))

        # add top neighbor
        if y > 0 and cmap[y-1,x] == ".":
            tovisit.add((x,y-1))
        # add bottom neighbor
        if y < cmap.shape[0]-1 and cmap[y+1,x] == ".":
            tovisit.add((x,y+1))
        # add left neighbor
        if x > 0 and cmap[y,x-1] == ".":
            tovisit.add((x-1,y))
        # add right neighbor
        if x < cmap.shape[1]-1 and cmap[y,x+1] == ".":
            tovisit.add((x+1,y))

    answer2 = np.sum(cmap == ".")
    print("\n".join("".join(row) for row in cmap))
    print(answer2)



def day_11(inp):
    orig_map = [list(line) for line in inp.strip().splitlines()]
    new_map = []
    for line in orig_map:
        new_map.append(line)
        if all(c == "." for c in line):
            new_map.append(line)
    flipped_map = list(zip(*new_map))
    new_map = []
    for line in flipped_map:
        new_map.append(line)
        if all(c == "." for c in line):
            new_map.append(line)
    new_map = list(zip(*new_map))
    cmap = np.array(new_map)

    galaxies = np.argwhere(cmap == "#")

    s = 0
    for i in range(len(galaxies)):
        for j in range(i+1,len(galaxies)):
            dist = np.sum(np.abs(galaxies[i]-galaxies[j]))
            s += dist
    answer1 = s

    galaxies2 = np.argwhere(np.array(orig_map) == "#")

    s = 0
    for i in range(len(galaxies)):
        for j in range(i+1,len(galaxies)):
            dist_shifted = np.sum(np.abs(galaxies[i]-galaxies[j]))
            dist_orig = np.sum(np.abs(galaxies2[i]-galaxies2[j]))
            s += dist_orig + (dist_shifted - dist_orig)*(1000000-1)
    answer2 = s
    return answer2






def day_12(inp):
    lines = inp.strip().splitlines()
    springs = [line.split(" ")[0] for line in lines]
    springs_nums = [tuple([int(a) for a in line.split(" ")[1].split(",")]) for line in lines]

    @lru_cache(maxsize=None)
    def num_pos(spring, nums, inspring):
        if len(nums) == 0:
            return 1 if all(c in ".?" for c in spring) else 0
        if (spring.count("#") + spring.count("?")) < sum(nums):
            return 0
        if spring == "":
            return 1 if sum(nums) == 0 else 0

        match spring[0]:
            case "#":
                if nums[0] == 0: return 0
                return num_pos(spring[1:], (nums[0]-1, *nums[1:]), True)
            case ".":
                if inspring:
                    if nums[0] != 0: return 0
                    else: return num_pos(spring[1:], nums[1:], False)
                return num_pos(spring[1:], nums, False)
            case "?":
                answer = 0
                if not inspring or nums[0] == 0:
                    answer += num_pos("." + spring[1:], nums, inspring)
                if nums[0] != 0:
                    answer += num_pos("#" + spring[1:], nums, inspring)
                return answer

    springs2 = ["?".join([s for _ in range(5)]) for s in springs]
    springs_nums2 = [5*n for n in springs_nums]

    answer2 = 0
    for spring,nums in tqdm(list(zip(springs2, springs_nums2))):
        answer2 += num_pos(spring, nums, False)

    return answer2





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
            url = f"https://adventofcode.com/{YEAR}/day/{day}/input"
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
    url = f"https://adventofcode.com/{YEAR}/day/{day}/answer"

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
    if datetime(YEAR,12,1,0,0,0) <= now < datetime(YEAR,12,26,0,0,0):
       #solve(now.day)
       solve(12)
    else:
       print("Advent has ended you fool 🎅")

if __name__ == "__main__":
    main()

