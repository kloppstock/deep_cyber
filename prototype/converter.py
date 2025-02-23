import numpy as np
import sys

# convert any index to a 4 tuple
def unpackIndex(i, default):
    a = b = c = d = default
    if type(i) == int:
        d = i
    elif len(i) == 1:
        d = i[0]
    elif len(i) == 2:
        c = i[0]
        d = i[1]
    elif len(i) == 3:
        b = i[0]
        c = i[1]
        d = i[2]
    else:
        a = i[0]
        b = i[1]
        c = i[2]
        d = i[3]
    return (a, b, c, d)

def convert(path):

    # load the file
    arr = np.load(path + ".npy")
    
    # open the output file
    with open(("../cifar10/" + path + ".c").lower(), "w") as f:
        # get dimensions
        (a, b, c, d) = unpackIndex(arr.shape, 1)
        arr = arr.reshape((a, b, c, d))
        
        # write head
        f.write('#include "../include/deep_cyber.h"\n')
        f.write('\n')
        f.write('const uint8_t ' + path.upper() + '_DATA[' + str(arr.view(np.uint8).flatten().shape[0]) + '] = {\n')
        
        # write data
        for ai in range(a):
            for bi in range(b):
                for ci in range(c):
                    for di in range(d):
                        elem_arr = np.zeros((1), dtype=np.float32)
                        elem_arr[0] = arr[ai, bi, ci, di]
                        elem = elem_arr.view(np.uint8).flatten()
                        e = elem.shape[0]
                        for ei in range(e):
                            if ai == a - 1 and bi == b - 1 and ci == c - 1 and di == d - 1 and ei == e - 1:
                                break
                            f.write('\t' + hex(elem[ei]) + ',\n')
                    
        # write tail        
        elem_arr = np.zeros((1), dtype=np.float32)
        elem_arr[0] = arr.flatten()[-1]
        elem = elem_arr.view(np.uint8).flatten()
        e = elem.shape[0]
        f.write('\t' + hex(elem[-1]) + '};\n')
        f.write('\n')
        f.write('Tensor ' + path.upper() + ' = {' + str(a) + ', ' + str(b) + ', ' + str(c) + ', ' + str(d) + ', (float*)' + path.upper() + '_DATA};\n')
        
convert("c1b")
convert("c1w")
convert("c2b")
convert("c2w")
convert("c3b")
convert("c3w")
convert("c4b")
convert("c4w")

convert("d1b")
convert("d1w")
convert("d2b")
convert("d2w")
