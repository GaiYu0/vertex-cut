import sys

f = 'data/' + sys.argv[1]
n_nodes = int(sys.argv[2])
lines = open(f).readlines()
for i, line in enumerate(lines):
    if line[0] != '#':
        break
lines = lines[i:]
reduced = []
for line in lines:
    m, n = tuple(line[:-1].split('\t'))
    if int(m) >= n_nodes:
        break
    if int(n) < n_nodes:
        reduced.append(line)
open('data/' + f + '-reduced', 'w').write(''.join(reduced))
