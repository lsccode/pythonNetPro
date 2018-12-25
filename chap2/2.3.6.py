import numpy as np
import pylab
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt

a = np.zeros([3,2])
print(a)
a[0,0] = 1
a[0,1] = 5.0
a[1,0] = 2
a[1,1] = 3.0
a[2,0] = 4.0
a[2,1] = 7.0
print(a)
plt.imshow(a,interpolation="nearest")
plt.show()