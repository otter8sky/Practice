from operations import *
from main import n

file = open('DATA.txt', 'r')
s = 0
names = []
Bodies = []
colors = []
mas = []

cnt_body = 0

for item in file:
    if cnt_body > n:
        break
    item = item.strip()
    if s < 5:
        if s == 1 or s == 2:
            item = list(map(float, item.split()))
        elif s == 3:
            item = float(item)
        mas.append(item)
        s += 1
    else:
        names.append(mas[0])
        colors.append(mas[4])
        Bodies.append(Body(mas[1], mas[2], mas[3], Acs))
        mas = []
        s = 0
        cnt_body += 1

file.close()

