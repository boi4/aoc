#!/usr/bin/env python3

edges = set()
vertices = set()

f = open("inputs/input07.txt","r")
for line in f:
    ind = line.find(" ") + 1
    ind2 = line.find("step") + 5
    a = line[ind:ind+1]
    b = line[ind2:ind2+1]
    edges.add((a,b))
    vertices.add(a)
    vertices.add(b)
f.close()

print("found", len(edges), "edges")
print("found", len(vertices), "vertices")

pred = {}

for v in vertices:
    preds = set()
    for (a,b) in edges:
        if b == v:
            preds.add(a)
    pred[v] = preds

done = set()
path = []

workers = 5*[0]
workerchars = 5*[" "]
print(workerchars)

todo = vertices.copy()
free = vertices.copy()
for (a,b) in edges:
   if b in free:
       free.remove(b) 

count = 0
while done != vertices:
    #print(count, ": remaining:", workers, "chars:", workerchars)
    print("".join(workerchars), "done:", "".join(path))
    #print("open:", free, "todo:", todo, "done:", "".join(path))


    changed = True
    tochange = [0,1,2,3,4]
    while changed:
        changed = False
        for i in range(len(workers)):
            if i not in tochange:
                continue
            if workers[i] <= 0:
                if workerchars[i] != " ":
                    done.add(workerchars[i])
                    path.append(workerchars[i])
              #      print(workerchars[i], "done")
                for v in todo:
                    if pred[v].issubset(done):
                        free.add(v)
                workerchars[i] = " "
                if not free:
                    continue
                tochange.remove(i)
                workerchars[i] = min(free)
                free.remove(workerchars[i])
                #print(todo)
                todo.remove(workerchars[i])
                workers[i] = ord(workerchars[i]) - ord("A") + 1 + 60

    for i in range(len(workers)):
        workers[i] -= 1
    count += 1

count -= 1
print(count)


#while todo:
#    current = min(free)
#    free.remove(current)
#    todo.remove(current)
#    done.add(current)
#
#    print("current:", current)
#    print("free:", free)
#    print("todo:", todo)
#    print("done:", done)
#    path.append(current)
#
#    for v in todo:
#        if pred[v].issubset(done):
#            free.add(v)

#print("".join(path))
