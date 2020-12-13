#!/usr/bin/env python3
import collections
import functools
import itertools
import operator
import re
import math
from pprint import pprint


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

    def eea(a,b):
        """
        kgv, alpha, beta
        """
        if b == 0:
            return (a,1,0)

        (ds,ss,ts) = eea(b, a % b)
        return (ds, ts, ss - ((a // b) * ts))

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


day_13()
