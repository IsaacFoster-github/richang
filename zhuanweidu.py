import numpy as np
import scipy
from scipy import interpolate
import os
def resizeFeature(inputData, newSize):
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    # print originalSize
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = scipy.interpolate.interp1d(x, inputData, axis=0)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new
# for i in range(1,7,1):
#     path = str(i) + ".npy"
#     dim = np.load(path)
#     dim_new = resizeFeature(dim, 100)
#     print(dim_new.shape)
path = "/data/zqs/ssj/datasets/tianchi/feature/audio_test/"
for root,dirs,files in os.walk(path):
    # for file in files:
        # path1 = str(file# )
    for file in files:
        try:
            dim = np.load("/data/zqs/ssj/datasets/tianchi/feature/audio_test/"+file)
        except:
            print(file)
        try:
            dim_new = resizeFeature(dim, 100)
        except:
            print(file)
        try:
            np.save('/data/zqs/ssj/datasets/tianchi/feature/audio_test_new/'+file, dim_new)
        except:
            print(file)