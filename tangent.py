import operations

x_0, y_0, R, a, b = map(int, input().split())

def find_circle_intersection(x1, y1, r1, x2, y2, r2):

    dist = operations.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    a = (r1**2 - r2**2 + dist ** 2) / (2 * dist)

    h = operations.sqrt(r1 ** 2 - a ** 2)

    x3 = x1 + a * (x2 - x1) / dist
    y3 = y1 + a * (y2 - y1) / dist

    x4 = x3 + h * (y2 - y1) / dist
    y4 = y3 - h * (x2 - x1) / dist

    x5 = x3 - h * (y2 - y1) / dist
    y5 = y3 + h * (x2 - x1) / dist

    return x4, y4, x5, y5

d = ((a - x_0)**2 + (b - y_0)**2)**0.5

if d < R:
    print(0)
elif d == R:
    print(1)
    print(a, b)
else:
    print(2)
    print(*find_circle_intersection(x_0, y_0, R, a, b, (d**2 - R**2) ** 0.5))