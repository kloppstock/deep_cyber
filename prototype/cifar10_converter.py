from tensorflow.keras.datasets import cifar10
import numpy as np

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

def saveData(arr, reference): 
    # open the output file
    with open("../cifar10_data.c", "w") as f:
        # get dimensions
        (a, b, c, d) = unpackIndex(arr.shape, 1)
        arr = arr.reshape((a, b, c, d))
        reference = reference.flatten()
        
        # write head
        f.write('#include "cifar10_data.h"\n')
        f.write('\n')
        f.write('const uint8_t CIFAR10_DATA[' + str(arr.view(np.uint8).flatten().shape[0]) + '] = {\n')
        
        # write data
        for ai in range(a):
            for bi in range(b):
                for ci in range(c):
                    for di in range(d):
                        elem_arr = np.zeros((1), dtype=np.float32, order='C')
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
        f.write('Tensor CIFAR10 = {' + str(a) + ', ' + str(b) + ', ' + str(c) + ', ' + str(d) + ', (float*)CIFAR10_DATA};\n')
        f.write('\n')
        f.write('const uint8_t CIFAR10_REFERENCE[' + str(reference.shape[0]) + '] = {')
        for i in reference[:-1]:
            f.write('\t' + str(int(i)) + ',\n')
        f.write('\t' + str(int(reference[-1])) + '};\n')

# load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# conversion and preprocessing
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255.
x_test /= 255.

data = np.array(x_test[0:10], dtype=np.float32, order='C')
saveData(data, y_test[0:10])
