import math

def f(params, *args): 
    x, y, z = params
    k1, k2  = args
    return (x-1)**2 + (y-2)**2 + (z-3)**2 + k1 + k2

def gradDesc(f, init, args):

    hval = 0.001
    step = 0.01
    curX = init
    norm = 9999
    thld = 1e-8

    i = 0
    while (norm > thld):

        gradient = [None for x in curX]
        for x in range(len(gradient)):
            stepXP = curX[:x] + [(curX[x] + hval)] + curX[x+1:]
            stepXN = curX[:x] + [(curX[x] - hval)] + curX[x+1:]
            gradient[x] = (f(stepXP, *args) - f(stepXN, *args))/(2*hval)

        oldX = curX
        curX = [curX[x] - step*gradient[x] for x in range(len(gradient))]
        norm = math.sqrt(sum([(curX[x]-oldX[x])**2 for x in range(len(gradient))]))
        i += 1

    return i, curX, f(curX, *args)


init = [0, 0, 0]
print(gradDesc(f, init, (5, 10)))
