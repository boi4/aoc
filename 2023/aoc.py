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

import requests
import numpy as np

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
       solve(now.day)
    else:
       print("Advent has ended you fool 🎅")

if __name__ == "__main__":
    main()

