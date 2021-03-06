from ctypes import CDLL, Structure, c_uint, c_float, c_ubyte, POINTER, pointer
import numpy as np

# custom interface functions
lib = CDLL("./libdeep_cyber.so")

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

class Tensor(Structure):
    _fields_ = [("a", c_uint),
               ("b", c_uint),
               ("c", c_uint),
               ("d", c_uint),
               ("data", POINTER(c_float))]

    def __init__(self, i):
        if type(i) == tuple:
            (a, b, c, d) = unpackIndex(i, 1)
            lib.create_tensor.argtypes = [c_uint, c_uint, c_uint, c_uint]
            lib.create_tensor.restype = Tensor
            t = lib.create_tensor(a, b, c, d)
            self.a = t.a
            self.b = t.b
            self.c = t.c
            self.d = t.d
            self.data = t.data
            t.data = None
        elif type(i) == np.ndarray:
            # create new tensor
            (a, b, c, d) = unpackIndex(i.shape, 1)
            lib.create_tensor.argtypes = [c_uint, c_uint, c_uint, c_uint]
            lib.create_tensor.restype = Tensor
            
            # copy data from numpy array
            t = lib.create_tensor(1, 1, 1, a*b*c*d)
            i = i.flatten()
            for x in range(t.d):
                t[x] = float(i[x])
                
            # move data 
            self.data = t.data
            t.data = None
            self.a = a
            self.b = b
            self.c = c
            self.d = d
        else:
            raise("Illegal input type!")

    def __del__(self):
        lib.free_tensor.argtypes = [Tensor]
        lib.free_tensor(self)

    def __getitem__(self, i):
        (a, b, c, d) = unpackIndex(i, 0)
        lib.at.argtypes = [POINTER(Tensor), c_uint, c_uint, c_uint, c_uint]
        lib.at.restype = POINTER(c_float)
        return lib.at(pointer(self), a, b, c, d)[0]

    def __setitem__(self, i, v):
        (a, b, c, d) = unpackIndex(i, 0)
        lib.at.argtypes = [POINTER(Tensor), c_uint, c_uint, c_uint, c_uint]
        lib.at.restype = POINTER(c_float)
        lib.at(pointer(self), a, b, c, d)[0] = float(v)

    @property
    def shape(self):
        if self.a != 1 and self.b != 1 and self.c != 1 and self.d != 1:
            return (self.a, self.b, self.c, self.d)
        elif self.b != 1 and self.c != 1 and self.d != 1:
            return (self.b, self.c, self.d)
        elif self.c != 1 and self.d != 1:
            return (self.c, self.d)
        else:
            return (self.d, )
        
    def reshape(self, i):
        (self.a, self.b, self.c, self.d) = unpackIndex(i, 1)
        
    def numpy(self):
        t = np.zeros((self.a, self.b, self.c, self.d))
        for ai in range(self.a):
            for bi in range(self.b):
                for ci in range(self.c):
                    for di in range(self.d):
                        t[ai, bi, ci, di] = self[ai, bi, ci, di]
        return t
    
def conv2d(X, w, b, kernel_size, strides, padding, groups):
    X = Tensor(X)
    w = Tensor(w)
    b = Tensor(b)
    lib.conv2d.argtypes = [Tensor, Tensor, Tensor, c_uint, c_uint, c_ubyte, c_uint]
    lib.conv2d.restype = Tensor
    return lib.conv2d(X, w, b, int(strides[0]), int(strides[1]), int(padding == "same"), int(groups)).numpy()

def dense(X, w, b):
    X = Tensor(X)
    w = Tensor(w)
    b = Tensor(b)
    lib.dense.argtypes = [Tensor, Tensor, Tensor]
    lib.dense.restype = Tensor
    return lib.dense(X, w, b).numpy()

def relu(X):
    X = Tensor(X)
    lib.relu.argtypes = [Tensor]
    lib.relu.restype = Tensor
    return lib.relu(X).numpy()

def sigmoid(X):
    X = Tensor(X)
    lib.sigmoid.argtypes = [Tensor]
    lib.sigmoid.restype = Tensor
    return lib.sigmoid(X).numpy()

def softmax(X):
    X = Tensor(X)
    lib.softmax.argtypes = [Tensor]
    lib.softmax.restype = Tensor
    return lib.softmax(X).numpy()

def maxpool2d(X, pool_sizes, strides, padding):
    X = Tensor(X)
    lib.maxpool2d.argtypes = [Tensor, c_uint, c_uint, c_uint, c_uint, c_ubyte]
    lib.maxpool2d.restype = Tensor
    return lib.maxpool2d(X, int(pool_sizes[0]), int(pool_sizes[1]), int(strides[0]), int(strides[1]), int(padding == "same")).numpy()

def avgpool2d(X, pool_sizes, strides, padding):
    X = Tensor(X)
    lib.avgpool2d.argtypes = [Tensor, c_uint, c_uint, c_uint, c_uint, c_ubyte]
    lib.avgpool2d.restype = Tensor
    return lib.avgpool2d(X, int(pool_sizes[0]), int(pool_sizes[1]), int(strides[0]), int(strides[1]), int(padding == "same")).numpy()

