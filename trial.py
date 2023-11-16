import operations

mas = list(map(int, input().split()))

x1 = mas[0]
y1 = mas[1]
R1 = mas[2]
x2 = mas[3]
y2 = mas[4]
R2 = mas[5]

def find_circle_intersection(x1, y1, r1, x2, y2, r2):
    dist = operations.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    a = (r1**2 - r2**2 + dist ** 2) / (2 * dist)
    h = operations.sqrt(abs(r1 ** 2 - a ** 2))

    x3 = x1 + a * (x2 - x1) / dist
    y3 = y1 + a * (y2 - y1) / dist
    x4 = x3 + h * (y2 - y1) / dist
    y4 = y3 - h * (x2 - x1) / dist
    x5 = x3 - h * (y2 - y1) / dist
    y5 = y3 + h * (x2 - x1) / dist
    A = [x4, y4]
    B = [x5, y5]
    return A, B

r = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

if x1 == x2 and y1 == y2 and R1 == R2:
    print(3)
elif x1 == x2 and y1 == y2:
    print(0)
elif R1 + R2 < r or R1 + r < R2 or R2 + r < R1:
    print(0)
elif R1 + r == R2:
    print(1)
    x = x2 + (x1 - x2) * (r + R1) / r
    y = y2 + (y1 - y2) * (r + R1) / r
    print(x, y)
elif R2 + r == R1:
    print(1)
    x = x1 + (x2 - x1) * (r + R2) / r
    y = y1 + (y2 - y1) * (r + R2) / r
    print(x, y) 
elif R1 + R2 == r:
    print(1)
    x = R1 * (x2 - x1) / (R1 + R2)
    y = R1 * (y2 - y1) / (R1 + R2)
    print(x, y)
else:
    print(2)
    A, B = find_circle_intersection(x1, y1, R1, x2, y2, R2)
    print(*A)
    print(*B)
