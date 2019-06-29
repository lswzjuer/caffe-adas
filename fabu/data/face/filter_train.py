import os

os.system('find train -iname "*.jpg" > train.txt')

f = open('train.txt', 'r')
lines = f.readlines()
res = []
cnt = 0
phone_cnt = 0
for line in lines:
    if 'ava' not in line and 'hmdb' not in line and 'normal' not in line and '2018' in line and 'charade' not in line and 'drink' not in line:
        cnt += 1
        if 'smoke' in line and cnt % 3 == 0:
            continue
        if 'smoke' not in line and cnt % 3:
            continue
    #if 'ava' not in line and 'hmdb' not in line and 'charade' not in line and '2018' in line:
    #    continue
    #if 'charade' in line:
    #    continue
    #if 'ava' in line:
    #    continue
    if '0606' in line:
        continue
    res.append(line)
f.close()
with open('train.txt', 'w+') as f:
    for line in res:
        f.write(line)

os.system('find test -iname "*.jpg" > test.txt')

f = open('test.txt', 'r')
lines = f.readlines()
res = []
cnt = 0
phone_cnt = 0
for line in lines:
    cnt += 1
    if '0606' in line:
        continue
    #if 'ava' in line:
    #    continue
    #if 'ava' not in line and 'hmdb' not in line and 'charade' not in line and '2018' in line:
    #    continue
    if 'ava' not in line and 'hmdb' not in line and 'normal' not in line and '2018' in line and 'charade' not in line and 'drink' not in line:
        cnt += 1
        if 'smoke' in line and cnt % 3 == 0:
            continue
        if 'smoke' not in line and cnt % 3:
            continue
    #if 'ava' in line:
    #    continue
    #if 'charade' in line:
    #    continue
    #elif 'ava' not in line and 'phone' in line:
    #    if cnt % 3 == 0:
    #        continue
    res.append(line)
f.close()
with open('test.txt', 'w+') as f:
    for line in res:
        f.write(line)
