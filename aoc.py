#!/usr/bin/env python3
import collections
import functools
import itertools
import operator
import os
import re
import math
import subprocess
from pprint import pprint
ppritn = pprint
ppirnt = pprint
from datetime import datetime
from copy import deepcopy


def product(it):
    return functools.reduce(lambda acc,x: acc * x, it, 1)


def day_1():
    numbers = [int(a.strip()) for a in open("1.txt").read().split("\n") if a.strip()]
    print([a * b * c for a in numbers for b in numbers for c in numbers if a + b + c == 2020])


def day_2():
    lines = [re.match(r"(\d+)-(\d+) (.): (.*)", a.strip()).groups() for a in open("2.txt").read().split("\n") if a.strip()]
    #print(sum([1 for a in lines if (int(a[0]) <= a[3].count(a[2]) <= int(a[1]))]))
    print(sum([1 for a in lines if (a[3][int(a[0]) - 1] == a[2]) ^ (a[3][int(a[1]) -1] == a[2])]))


def day_3():
    lines = [line.strip() for line in open("3.txt")]
    print(product([functools.reduce(lambda acc, t: acc + int((t[1][(t[0] * right) % len(lines[0])] == "#")), enumerate(lines[::down]), 0) for (right,down) in [(1,1),(3,1),(5,1),(7,1),(1,2)]]))


def day_4():
    #passports = [[k.split(":")[0] for k in pp.split()] for pp in open("4.txt").read().split("\n\n")]
    #print(len([passport for passport in passports if all(k in passport for k in ["byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"])]))
    passports = [{k.split(":")[0]:k.split(":")[1] for k in pp.split()} for pp in open("4.txt").read().split("\n\n")]
    pprint(len([k for k in passports if all([
        "byr" in k and re.match(r"^\d{4}$", k["byr"]) and 1920 <= int(k["byr"]) <= 2002,
        "iyr" in k and re.match(r"^\d{4}$", k["iyr"]) and 2010 <= int(k["iyr"]) <= 2020,
        "eyr" in k and re.match(r"^\d{4}$", k["eyr"]) and 2020 <= int(k["eyr"]) <= 2030,
        "hgt" in k and re.match(r"^\d+(in|cm)$", k["hgt"]) and int(re.match(r"^(\d+)(in|cm)$", k["hgt"]).group(1)) in [range(150,194),range(59,77)][k["hgt"].endswith("in")],
        "hcl" in k and re.match(r"^#[0-9a-f]{6}$", k["hcl"]),
        "ecl" in k and k["ecl"] in ["amb", "blu", "brn", "gry", "grn", "hzl", "oth"],
        "pid" in k and re.match(r"^\d{9}$", k["pid"])])]))


def day_5():
    vals = sorted([int("".join({"F":"0","B":"1","L":"0","R":"1"}[c] for c in line.strip()),2) for line in open("5.txt")])
    vals = sorted([int("".join(["0","1"]["FLBR".index(c)//2] for c in line.strip()),2) for line in open("5.txt")])
    vals = sorted([int("".join(str("FLBR".index(c)//2) for c in line.strip()),2) for line in open("5.txt")])
    #print(max(vals))
    print([val+1 for (i,val) in enumerate(vals[:-1]) if vals[i+1] == val + 2][0])
    print([a-1 for (a,b) in zip(vals,range(vals[0],vals[-1]-1)) if a != b][0])
    print({(val1-val2): val1-1 for (val1,val2) in zip(vals[1:],vals[:-1])}[2])
    print([val for val in range(vals[0],vals[-1]-1) if val not in vals][0])


def day_6():
    #answers = [set(c for line in group for c in line.strip()) for group in  open("6.txt").read().split("\n\n")]
    #pprint(sum(len(answer) for answer in answers))

    answers = [[set(line.strip()) for line in group.split("\n") if line.strip()] for group in  open("6.txt").read().split("\n\n")]
    pprint(sum([len(functools.reduce(operator.and_, [set(c for c in answer) for answer in group],set(c for answer in group for c in answer))) for group in answers]))


def day_7():
    rules = [re.match(r"^(.*?) bags contain (no other bags|.*)\.$", line).groups() for line in open("7.txt").read().split('\n')[:-1]]
    rules = {r[0]: [] if r[1] == "no other bags" else [re.match(r"^(\d+) (.*?) bags?$", l).groups() for l in r[1].split(", ")] for r in rules}

    def get_contains(color): return functools.reduce(operator.or_, (get_contains(k) for (k,vs) in rules.items() if color in [a for (_,a) in vs]),set([color]))
    def get_num_bags(color): return sum(int(c)*(get_num_bags(r)+1) for (c,r) in rules[color])

    print(len(get_contains("shiny gold")) - 1)
    print(get_num_bags("shiny gold"))


def day_8():
    orig_code = [(line.strip().split()[0],int(line.strip().split()[1])) for line in open("8.txt")]

    def sim(code):
        ip,visited = 0,set()
        while ip not in visited and ip < len(code):
            visited.add(ip)
            ip += code[ip][1] if code[ip][0] == "jmp" else 1
        return sum(orig_code[i][1] for i in visited if orig_code[i][0] == "acc"),(ip == len(code))

    print(sim(orig_code)[0])

    for i in range(len(orig_code)):
        code = orig_code[:i] + [({"jmp":"nop","nop":"jmp","acc":"acc"}[orig_code[i][0]],orig_code[i][1])] + orig_code[i+1:]
        a,b = sim(code)
        b and print(a)


def day_9():
    ns = [int(line.strip()) for line in open("9.txt")]
    a = [a for i,a in enumerate(ns) if i >= 25 and not any(b+c == a for b,c in itertools.product(ns[i-25:i],repeat=2))][0]
    #print(a)
    print([max(ns[s:e])+min(ns[s:e]) for s,e in itertools.combinations(range(len(ns)+1),2) if s != e-1 and sum(ns[s:e]) == a][0])


def day_10():
    ns = sorted([int(line.strip()) for line in open("10.txt")])
    ns.insert(0,0)
    ns.append(ns[-1] + 3)

    print(product(collections.Counter(a-b for a,b in zip(ns[1:], ns[:-1])).values()))

    counts = [len(list(it)) for val,it in itertools.groupby([a-b for a,b in zip(ns[1:], ns[:-1])]) if val == 1]

    @functools.cache
    def combs(n): return 1 if n < 2 else 2 if n == 2 else combs(n-1) + combs(n-2) + combs(n-3)

    print(product(combs(count) for count in counts))


def day_11():

    directions = [(drow,dcol) for dcol in range(-1,2) for drow in range(-1,2) if (drow,dcol) != (0,0)]

    def round1(seats):
        new_seats = [row.copy() for row in seats]
        for row in range(1,len(seats)-1):
            for col in range(1,len(seats[row])-1):
                count = sum(seats[row+drow][col+dcol] == "#" for drow,dcol in directions)
                if seats[row][col] == "L" and count == 0:
                        new_seats[row][col] = "#"
                elif seats[row][col] == "#" and count >= 4:
                        new_seats[row][col] = "L"
        return new_seats

    def round2(seats):
        new_seats = [row.copy() for row in seats]

        for row in range(1,len(seats)-1):
            for col in range(1,len(seats[row])-1):
                val = seats[row][col]
                if val == ".":
                    continue

                count = 0
                for drow,dcol in directions:
                    tmp = ((row + drow*s,col + dcol*s) for s in range(1,min(len(seats),len(seats[0]))))
                    tmp = itertools.takewhile(lambda x: 0 <= x[0] < len(seats) and 0 <= x[1] < len(seats[0]), tmp)
                    trace = (seats[row2][col2] for row2,col2 in tmp)
                    trace = list(trace)

                    a = list(itertools.dropwhile(lambda c: c not in ["#","L"], trace))
                    if a and a[0] == "#":
                        count += 1
                        #if val == "L":
                            #break
                        #elif count >= 5:
                            #break


                if val == "L" and count == 0:
                    new_seats[row][col] = "#"
                elif val == "#" and count >= 5:
                    new_seats[row][col] = "L"
                    
        return new_seats

    seats = [list(line.strip()) for line in open("11.txt")]
    #seats = [list(line.strip()) for line in open("a")]
    seats.append( len(seats[0]) * ["."])
    seats.insert(0,len(seats[0]) * ["."])
    seats = [["."] + row + ["."] for row in seats]

    #pprint(seats)

    c = 0
    while True:
        print(c)
        c += 1
        new_seats = round2(seats)
        #pprint(new_seats)
        eq = True
        for row in range(1,len(seats)-1):
            for col in range(1,len(seats[row])-1):
                if new_seats[row][col] != seats[row][col]:
                    eq = False
        if eq:
            break
        seats = new_seats

    count = 0
    for row in range(1,len(seats)-1):
        for col in range(1,len(seats[row])-1):
            if seats[row][col] == "#":
                count += 1
    print(count)


def day_12():
    dirs = {"E" : (1,0), "S": (0,1), "W": (-1,0), "N": (0,-1)}
    dirs2 = ["E", "S", "W", "N"]
    ins = [(line[0:1],int(line[1:].strip())) for line in open("12.txt")]
    #ins = [(line[0:1],int(line[1:].strip())) for line in open("a")]

    dir = "E"
    pos = (0,0)
    for instr in ins:
        if instr[0] == "F":
            pos = (pos[0] + instr[1] * dirs[dir][0], pos[1] + instr[1] * dirs[dir][1])
        elif instr[0] == "R":
            dir = dirs2[(dirs2.index(dir) + instr[1]//90) % 4]
        elif instr[0] == "L":
            dir = dirs2[(dirs2.index(dir) - instr[1]//90) % 4]
        else:
            pos = (pos[0] + instr[1] * dirs[instr[0]][0], pos[1] + instr[1] * dirs[instr[0]][1])

    print(abs(pos[0]) + abs(pos[1]))


    wp = (10,-1)
    pos = (0,0)
    for instr in ins:
        if instr[0] == "F":
            pos = (pos[0] + instr[1] * wp[0], pos[1] + instr[1] * wp[1])
        elif instr[0] == "R":
            for i in range(instr[1]//90):
                wp = (-wp[1],wp[0])
        elif instr[0] == "L":
            for i in range(instr[1]//90):
                wp = (wp[1],-wp[0])
        else:
            wp = (wp[0] + instr[1] * dirs[instr[0]][0], wp[1] + instr[1] * dirs[instr[0]][1])

    print(abs(pos[0]) + abs(pos[1]))


def day_13():
    lines = [line.strip() for line in open("13.txt")]
    #lines = [line.strip() for line in open("a")]
    earliest = int(lines[0])

    busses = [(int(l),i) for i,l in enumerate(lines[1].split(',')) if l != "x"]

    def crt(modulun, values):
        M = product(modulun)
        xs = []
        for m_i,a_i in zip(modulun, values):
            M_i = M//m_i
            s_i = pow(M_i, -1, m_i)
            xs.append(a_i * M_i * s_i)
        return sum(xs) % M

    waits = [bus - (earliest % bus) for bus,_ in busses]
    print(busses[waits.index(min(waits))][0] * min(waits))


    vs = [v for v,_ in busses]
    v2s = [v-i for v,i in busses]
    print(crt(vs, v2s))


def day_14():
    lines = [line.strip() for line in open("14.txt")]
    #lines = [line.strip() for line in open("a")]

    memory = collections.defaultdict(int)
    mask = 36*"X"
    for line in lines:
        comm = line.split("=")[0].strip()
        if comm == "mask":
            mask = line.split("=")[1].strip()
        else:
            addr = int(re.match(r"^mem\[(\d+)\]$", comm).groups()[0])
            val = bin(int(line.split("=")[1].strip()))[2:].rjust(36, "0")
            memory[addr] = int("".join(m if m != "X" else c for c,m in zip(val, mask)),2)

    #print("\n".join(f"{k:3}: {v}" for k,v in memory.items()))
    print(sum(memory.values()))

    memory = collections.defaultdict(int)
    mask = 36*"X"
    for line in lines:
        comm = line.split("=")[0].strip()
        if comm == "mask":
            mask = line.split("=")[1].strip()
        else:
            addr = bin(int(re.match(r"^mem\[(\d+)\]$", comm).groups()[0]))[2:].rjust(36, "0")
            val = int(line.split("=")[1].strip())
            addr_floating = "".join(c if m == "0" else "1" if m == "1" else "X" for c,m in zip(addr, mask))
            indizes = [i for i,c in enumerate(addr_floating) if c == "X"]
            for i in range(2**len(indizes)):
                i2 = bin(i)[2:].rjust(len(indizes), "0")
                addr2 = list(addr_floating)

                for j,bit in zip(indizes, i2):
                    addr2[j] = bit

                memory[int("".join(addr2),2)] = val
    print(sum(memory.values()))


def day_15():
    last_index = collections.defaultdict(int)
    vals = [int(i) for i in next(open("15.txt")).strip().split(",")]

    for i,v in enumerate(vals[:-1]):
        last_index[v] = i + 1

    i = len(vals)
    val = vals[-1]
    while True:
        if val in last_index:
            tmp = last_index[val]
            last_index[val] = i
            val = i - tmp
        else:
            last_index[val] = i
            val = 0
        if i+1 == 30000000:
            print(val)
            break
        i += 1



def day_16():
    fields,myticket,tickets = [[line.strip() for line in f.split("\n") if line.strip()] for f in open("16.txt").read().split("\n\n")]
    fields = {field.split(":")[0].strip():[range(int(a.strip()),int(b.strip())+1) for a,b in [f.split("-") for f in field.split(":")[1].split("or")]] for field in fields}

    myticket = [int(val) for val in myticket[1].split(",")]

    tickets = [[int(val) for val in ticket.split(",")] for ticket in tickets[1:]]

    s = 0
    valid_tickets = []
    for ticket in tickets:
        valid = True
        for val in ticket:
            if not any(any(val in r for r in field) for field in fields.values()):
                s += val
                valid = False
        if valid:
            valid_tickets.append(ticket)
    print(s)

    ok_indizes = collections.defaultdict(list)
    for fieldname,fieldranges in fields.items():
        for i in range(len(myticket)):
            if all(any(ticket[i] in r for r in fieldranges) for ticket in valid_tickets):
                ok_indizes[fieldname].append(i)

    sol = {}
    while len(ok_indizes):
        val = -1
        kk = -1
        for k,v in ok_indizes.items():
            if len(v) == 1:
                sol[k] = v[0]
                val = v[0]
                kk = k
        if val == -1:
            print("Oh no")
            return
        else:
            del ok_indizes[kk]
            for k,v in ok_indizes.items():
                if val in v:
                    v.remove(val)

    print(product([myticket[i] for f,i in sol.items() if
        f.startswith("departure")]))


def day_17():
    lines = [list(line.strip()) for line in open("17.txt")]

    field = collections.defaultdict(lambda :".")
    field2 = collections.defaultdict(lambda :".")
    for y in range(len(lines)):
        for x in range(len(lines[y])):
            field[(x,y,0)] = lines[y][x]
            field2[(x,y,0,0)] = lines[y][x]

    def round(field, ndims, switchrule):
        res = deepcopy(field)
        boundaries = list(zip(*[[(f(key[i] for key in field.keys()))-1+3*j for i in range(ndims)] for j,f in enumerate((min,max))]))
        for coords in itertools.product(*(range(*b) for b in boundaries)):
            c = sum((field[(*(a+b for a,b in zip(coords,deltas)),)] == "#") for deltas in itertools.product((-1,0,1), repeat=ndims) if not all(delta == 0 for delta in deltas))
            res[(*coords,)] = "#."[("#.".index(field[(*coords,)]) + switchrule(field[(*coords,)],c)) % 2]
        return res

    for i in range(6):
        field = round(field, 3, lambda v,c: c not in range(2,3+1) if v == "#" else c in range(3,3+1))
        field2 = round(field2, 4, lambda v,c: c not in range(2,3+1) if v == "#" else c in range(3,3+1))

    print(sum(v == "#" for v in field.values()))
    print(sum(v == "#" for v in field2.values()))



def day_18():
    lines = [line.strip() for line in open("18.txt")]

    def tokenize(line):
        tokens_tmp = line.split(' ')
        tokens = []
        for token in tokens_tmp:
            while token.startswith('('):
                tokens.append("(")
                token = token[1:]
            c = 0
            while token.endswith(')'):
                c += 1
                token = token[:-1]
            tokens.append(token)
            tokens += c * [')']

        return tokens

    def gen_tree(tokens):
        if len(tokens) == 1:
            return int(tokens[0])
        elif tokens[-1] == ")":
            d = 1
            opening_index = -1
            for i in range(len(tokens)-2,-1,-1):
                if tokens[i] == '(':
                    d -= 1
                elif tokens[i] == ')':
                    d += 1
                if d == 0:
                    opening_index = i
                    break
            if opening_index == 0:
                return gen_tree(tokens[1:-1])
            return (gen_tree(tokens[:opening_index-1]), tokens[opening_index-1], gen_tree(tokens[opening_index+1:-1]))
        else:
            return (gen_tree(tokens[:-2]), tokens[-2], gen_tree(tokens[-1]))

    def gen_tree2(tokens):
        if len(tokens) == 1:
            return int(tokens[0])
        starpos = -1
        d = 0
        for i in range(len(tokens)-1,-1,-1):
            if tokens[i] == ')':
                d += 1
            elif tokens[i] == '(':
                d -= 1
            if d == 0 and tokens[i] == '*':
                starpos = i
                break
        if starpos != -1:
            return (gen_tree2(tokens[:starpos]), '*', gen_tree2(tokens[starpos+1:]))

        if tokens[-1] == ")":
            d = 1
            opening_index = -1
            for i in range(len(tokens)-2,-1,-1):
                if tokens[i] == '(':
                    d -= 1
                elif tokens[i] == ')':
                    d += 1
                if d == 0:
                    opening_index = i
                    break
            if opening_index == 0:
                return gen_tree2(tokens[1:-1])
            return (gen_tree2(tokens[:opening_index-1]), tokens[opening_index-1], gen_tree2(tokens[opening_index+1:-1]))
        else:
            return (gen_tree2(tokens[:-2]), tokens[-2], gen_tree2(tokens[-1]))


    def eval_tree(tree):
        if type(tree) == type(1):
            return tree
        elif tree[1] == "+":
            return eval_tree(tree[0]) + eval_tree(tree[2])
        elif tree[1] == "*":
            return eval_tree(tree[0]) * eval_tree(tree[2])

    def eval_line(line):
        tokens = tokenize(line)
        tree = gen_tree(tokens)
        return eval_tree(tree)

    def eval_line2(line):
        tokens = tokenize(line)
        tree = gen_tree2(tokens)
        return eval_tree(tree)

    print(sum(eval_line(line) for line in lines))
    print(sum(eval_line2(line) for line in lines))


def day_19():
    parts = open("19.txt").read().split("\n\n")
    rules = {int(line.split(":")[0].strip()): line.split(":")[1].strip() for line in parts[0].split("\n")}
    rules = {n: [[tok[1] if tok[0] == '"' else int(tok) for tok in r.strip().split(" ")] for r in rule.split("|")] for n,rule in rules.items()}
    rules2 = deepcopy(rules)
    rules2[8] = [[42],[42, 8]]
    rules2[11] = [[42, 31],[42, 11, 31]]

    # easy solution of part 1 using pythons re modules
    #def create_pattern(rule):
    #    if type(rule) == type("a"):
    #        return rule
    #    else:
    #        return "(" + "|".join("".join(create_pattern(rule) for rule in and_rules) for and_rules in rules[rule]) + ")"

    #p = "^" + create_pattern(0) + "$"
    #print(len([line for line in parts[1].split("\n") if re.match(p,line)]))

    def match_rule(line, rule, i, rules):
        if i >= len(line):
            return -1
        if type(rule) == type("a"):
            return i+1 if line[i] == rule else -1
        matching = []
        for and_rules in rules[rule]:
            matches = True
            j = i
            for and_rule in and_rules:
                j = match_rule(line, and_rule, j, rules)
                if j == -1:
                    matches = False
                    break
            if matches:
                matching.append(j)
        if len(matching) == 0:
            return -1
        return matching[0]

    def match_rule2(line, rule, i, rules):
        if i >= len(line):
            return set()
        if type(rule) == type("a"):
            return set((i+1,)) if line[i] == rule else set()
        matching = set()
        for and_rules in rules[rule]:
            positions = set((i,))
            for and_rule in and_rules:
                new_positions = set()
                for position in positions:
                    ks = match_rule2(line, and_rule, position, rules)
                    for k in ks:
                        new_positions.add(k)
                positions = new_positions
            for p in positions:
                matching.add(p)
        return matching

    print(len([line for line in parts[1].split("\n") if match_rule(line, 0, 0, rules) == len(line)]))
    print(len([line for line in parts[1].split("\n") if len(line) in match_rule2(line, 0, 0, rules2)]))


def day_20():
    tiles = open("20.txt").read().strip().split("\n\n")
    tiles = {re.match(r'Tile (\d+):', tile.split('\n')[0]).groups()[0]: [list(line) for line in tile.split("\n")[1:]] for tile in tiles}

    num_tiles_borders = round(math.sqrt(len(tiles)))
    len_tile = len(list(tiles.values())[0])

    def get_borders(tile):
        a = ["".join(tile[0]), "".join(l[-1] for l in tile), "".join(tile[-1][::-1]), "".join([l[0] for l in tile][::-1])]
        return a + ["".join(reversed(k)) for k in a]

    def share_border(tile1, tile2):
        """
        the first part of result should never be flipped (so < 4)
        borders are clockwise with 0 at top
        """
        borders1 = get_borders(tile1)
        borders2 = get_borders(tile2)
        for i,border1 in enumerate(borders1):
            for j,border2 in enumerate(borders2):
                if border1 == "".join(reversed(border2)):
                    return (i,j)
        return None

    def rotate_counter_clockwise(tile):
        res = [[' ' for i in range(len(tile))] for j in range(len(tile))]
        for i in range(len(tile)):
            for j in range(len(tile)):
                res[-j-1][i] = tile[i][j]
        return res

    def flip(tile):
        res = [[' ' for i in range(len(tile))] for j in range(len(tile))]
        for i in range(len(tile)):
            for j in range(len(tile)):
                res[i][-j-1] = tile[i][j]
        return res
    

    ts = list(tiles.items())
    borders = collections.defaultdict(list)
    for name1,tile1 in ts:
        for name2,tile2 in ts:
            if name1 == name2:
                continue
            a = share_border(tile1,tile2)
            if a is not None:
                borders[name1].append((name2, a))

    corners = [name for name,l in borders.items() if len(l) == 2]

    print(product(int(corner) for corner in corners))


    # orient and setup upper left corner
    upper_left = corners[0]
    upper_left_borders = borders[upper_left]

    neighbour1 = upper_left_borders[0]
    neighbour2 = upper_left_borders[1]

    if (neighbour1[1][0] - neighbour2[1][0]) % 4 == 1:
        neighbour1,neighbour2 = neighbour2,neighbour1
    

    gridnrs = [[0 for i in range(num_tiles_borders)] for j in range(num_tiles_borders)]
    gridcontents = [[[[' ' for k in range(len_tile)] for l in range(len_tile)] for i in range(num_tiles_borders)] for j in range(num_tiles_borders)]
    #print(len(gridcontents), len(gridcontents[0]))

    tt = deepcopy(tiles[upper_left])
    n_rots = (neighbour1[1][0] - 1) % 4

    for i in range(n_rots):
        tt = rotate_counter_clockwise(tt) 

    gridnrs[0][0] = upper_left
    gridcontents[0][0] = tt

    next_nr = neighbour1[0]
    next_nr_down = neighbour2[0]
    has_to_flip = neighbour1[1][1] >= 4
    has_to_flip_down = neighbour2[1][1] >= 4

    for row in range(num_tiles_borders):
        if row != 0:
            # setup col 0 of row
            nr = next_nr_down
            t = tiles[nr]
            if has_to_flip_down:
                t = flip(t)
            a,b = share_border(t, gridcontents[row-1][0])
            if b != 2:
                print("Oh no2 :(")
            n_rots = a
            for i in range(n_rots):
                t = rotate_counter_clockwise(t)

            gridnrs[row][0] = nr
            gridcontents[row][0] = t

            neighbours = [b[0] for b in borders[nr]]
            for neighbour in neighbours:
                t2 = tiles[neighbour]
                a,b = share_border(t,t2)
                if a == 2:
                    next_nr_down = neighbour
                    has_to_flip_down = b >= 4
                if a == 1:
                    next_nr = neighbour
                    has_to_flip = b >= 4

        # setup rest of row
        for col in range(1,num_tiles_borders):
            nr = next_nr
            t = tiles[nr]

            # flip and rotate tile
            if has_to_flip:
                t = flip(t)
            a,b = share_border(t, gridcontents[row][col-1])
            if b != 1:
                print("Oh no :(")
            n_rots = (a + 1) % 4
            for i in range(n_rots):
                t = rotate_counter_clockwise(t)

            gridnrs[row][col] = nr
            gridcontents[row][col] = t

            # get next tile
            if col != num_tiles_borders - 1:
                neighbours = [b[0] for b in borders[nr]]
                for neighbour in neighbours:
                    t2 = tiles[neighbour]
                    a,b = share_border(t,t2)
                    if a == 1:
                        next_nr = neighbour
                        has_to_flip = b >= 4

    # remove joined borders and assemble image
    stripped = [[[row2[1:-1] for row2 in t[1:-1]] for t in row] for row in gridcontents]
    picture = [[stripped[row // (len_tile-2)][col][row % (len_tile-2)][col2] for col in range(num_tiles_borders) for col2 in range(len_tile-2)]
                        for row in range(num_tiles_borders * (len_tile-2))]


    monster = """
                  # 
#    ##    ##    ###
 #  #  #  #  #  #   
"""[1:-1].split('\n')
    
    sums = []
    # each possible combination of rotaions + flip
    for toflip in (True, False):
        for n_rots in range(4):
            # replace monsters with "0"
            picture2 = deepcopy(picture)
            for i in range(n_rots):
                picture2 = rotate_counter_clockwise(picture2)
            if toflip:
                picture2 = flip(picture2)

            for row in range(len(picture2)-len(monster)):
                for col in range(len(picture2[0])-len(monster[0])):
                    ok = True
                    for drow in range(len(monster)):
                        for dcol in range(len(monster[0])):
                            if monster[drow][dcol] == "#" and picture2[row+drow][col+dcol] != "#":
                                ok = False
                    if ok:
                        # if it's a monster, change tiles to "O" (we don't
                        # expect overlaps)
                        for drow in range(len(monster)):
                            for dcol in range(len(monster[0])):
                                if monster[drow][dcol] == "#":
                                    picture2[row+drow][col+dcol] = "O"

            sums.append(("\n".join("".join(row) for row in picture2)).count("#"))
    print(min(sums))




def day_21():
    lines = [line.strip() for line in open("21.txt")]
    food_list = []
    all_allergens = set()
    all_ingredients = set()
    for line in lines:
        a,b = line.split(" (contains ")
        ingredients = set(a.split())
        all_ingredients.update(ingredients)
        allergens = set(b[:-1].split(", "))
        all_allergens.update(allergens)
        food_list.append((ingredients,allergens))

    d = collections.defaultdict(lambda :deepcopy(all_ingredients))
    for allergen in all_allergens:
        for ings,alls in food_list:
            if allergen in alls:
                d[allergen] = d[allergen].intersection(ings)

    impossible_ingredients = [ing for ing in all_ingredients if not any(ing in ings for ings in d.values())]
    c = 0
    for a,_ in food_list:
        for b in a:
            if b in impossible_ingredients:
                c+=1
    print(c)

    final_d = {}
    while len(d):
        for k,v in d.items():
            if len(v) == 1:
                elem = list(v)[0]
                final_d[k] = elem
                del d[k]
                for k2,v2 in d.items():
                    if elem in v2:
                        v2.remove(elem)
                break

    l = list(final_d.items())
    l.sort(key=lambda b: b[0])
    print(",".join(ing for _,ing in l))


def day_22():
    decks = [[int(line) for line in split.split('\n')[1:]] for split in open("22.txt").read().strip().split('\n\n')]
    #decks = [[int(line) for line in split.split('\n')[1:]] for split in open("a").read().strip().split('\n\n')]
    decks2 = deepcopy(decks)

    while not any(len(deck) == 0 for deck in decks):
        p0 = decks[0].pop(0)
        p1 = decks[1].pop(0)
        winner = int(p1 > p0)
        decks[winner] = decks[winner] + [[p0, p1][winner],[p1, p0][winner]]

    winning_deck = decks[winner]
    #print(winning_deck)
    #print([(len(winning_deck) - i,card) for i,card in enumerate(winning_deck)])
    print(sum((len(winning_deck) - i) * card for i,card in enumerate(winning_deck)))

    
    def recursive_combat(deck0, deck1):
        deck0,deck1 = deepcopy(deck0),deepcopy(deck1)
        d = set()
        while len(deck0) > 0 and len(deck1) > 0:
            if (tuple(deck0),tuple(deck1)) in d:
                return 0,None,None
            else:
                d.add((tuple(deck0),tuple(deck1)))

                p0 = deck0.pop(0)
                p1 = deck1.pop(0)
                if len(deck0) >= p0 and len(deck1) >= p1:
                    winner,_,_ = recursive_combat(deck0[:p0], deck1[:p1])
                else:
                    winner = int(p1 > p0)
            if winner == 0:
                deck0 = deck0 + [p0, p1]
            else:
                deck1 = deck1 + [p1, p0]
        return winner,deck0,deck1

    winner,d0,d1 = recursive_combat(decks2[0], decks2[1])

    if winner == 0:
        winning_deck = d0
    else:
        winning_deck = d1

    print(sum((len(winning_deck) - i) * card for i,card in enumerate(winning_deck)))


def day_23():
    cups = [int(c) for c in open("23.txt").read().strip()]
    cups2 = deepcopy(cups)

    num_cups = len(cups)
    max_cup = max(cups)

    current_index = 0
    for i in range(100):
        current_cup = cups[current_index]
        #print(cups, current_cup)
        to_the_side = []
        for j in range(3):
            if  current_index + 1 >= len(cups):
                to_the_side.append(cups.pop(0))
            else:
                to_the_side.append(cups.pop((current_index + 1) % len(cups)))
        n = (current_cup - 1) % (max_cup + 1)
        l = -1
        while True:
            if n in cups:
                l = cups.index(n)
                break
            n = (n - 1) % (max_cup + 1)
        #print(to_the_side, n)
        for j in range(3):
            cups.insert(l+1, to_the_side.pop())

        current_index = (cups.index(current_cup) + 1) % len(cups)

    i = cups.index(1)
    print("".join(str(c) for c in cups[i+1:]) + "".join(str(c) for c in cups[:i]))


    cups = cups2
    current_cup = cups[0]
    print(cups)

    for i in range(10000000):
        to_the_side = []
        for j in range(3):
            if cups.index(current_cup) + 1 == len(cups):
                to_the_side.append(max((max(cups),max(to_the_side))))
            else:
                to_the_side.append(cups.pop(cups.index(current_cup) + 1))

        l = -1
        n = current_cup - 1
        while n > 0:
            if n in cups:
                l = cups.index(n)
                break
            n = n - 1
        #print(n)
        if n == 0:
            for j in range(3):
                cups.append(to_the_side.pop(0))
        else:
            for j in range(3):
                cups.insert(l+1, to_the_side.pop())

        if i % 100000 == 0:
            print(i, len(cups))
    print(cups)







day = datetime.now().day
if not f"{day}.txt" in os.listdir():
    subprocess.run(["zsh", "-ic", f"wgetcook https://adventofcode.com/2020/day/{day}/input -O {day}.txt"])

locals()[f"day_{day}"]()
