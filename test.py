def rand(r):
    base = 256.0
    a = 17.0
    b = 139.0
    tmp0 = a * r + b
    tmp1 = tmp0 // base
    tmp = tmp0 - tmp1 * base
    p = tmp / base
    return p

if __name__ == '__main__':
    for i in range(10):
        print(rand(i))