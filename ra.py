import numpy as np
def averline_p(x, y):
    sumx = 0
    sumxy = 0
    sumy = 0
    sum2x = 0

    for i in range(512):
        sumx = sumx + x[i]
        sumy = sumy + y[i]
        sum2x = sum2x + pow(x[i], 2)
        sumxy = sumxy + (x[i] * y[i])

    delta = sum2x * 512 - sumx * sumx
    dela = sumxy * 512 - sumy * sumx
    vala = dela / delta
    delb = sum2x * sumy - sumx * sumxy
    valb = delb / delta
    i = 0
    averL = np.zeros(512)
    for i in range(512):
        averL[i] = round(vala * x[i] + valb, 4)
    return averL


def calculation_Ra(x, y):
    averageln = averline_p(x, y)
    par_ra = 0
    for i in range(512):
        par_ra += abs(averageln[i] - y[i])

    parRa = par_ra / 512
    return round(parRa, 4)