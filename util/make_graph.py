import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D


x1 = [10, 20, 30, 40, 50]

y1 = [0.941998,    0.953399,    0.954899,    0.955199,    0.955199]
y2 = [0.956198,    0.965198,    0.964998,    0.965598,    0.965698]
y3 = [0.959197,    0.968796,    0.970196,    0.969896,    0.969796]
y4 = [0.959198,    0.971497,    0.972797,    0.972997,    0.972897]
y5 = [0.951398,    0.970698,    0.971098,    0.970399,    0.970398]
y6 = [0.9439,      0.965398,    0.967198,    0.966998,    0.966998]
y7 = [0.9372,      0.959299,    0.960799,    0.961098,    0.961198]

s16,   = plt.plot(x1, y1, 'o-', color = 'b', label='16')
s32,   = plt.plot(x1, y2, 'o-', color = 'g', label='32')
s64,   = plt.plot(x1, y3, 'o-', color = 'r', label='64')
s128,  = plt.plot(x1, y4, 'o-', color = 'c', label='128')
s256,  = plt.plot(x1, y5, 'o-', color = 'm', label='256')
s512,  = plt.plot(x1, y6, 'o-', color = 'y', label='512')
s1024, = plt.plot(x1, y7, 'o-', color = 'k', label='1024')

size_16   = mpatches.Patch(color='b', label='16')
size_32   = mpatches.Patch(color='g', label='32')
size_64   = mpatches.Patch(color='r', label='64')
size_128  = mpatches.Patch(color='c', label='128')
size_256  = mpatches.Patch(color='m', label='256')
size_512  = mpatches.Patch(color='y', label='512')
size_1024 = mpatches.Patch(color='k', label='1024')

plt.legend([size_16,size_32,size_64,size_128,size_256,size_512,size_1024],["16","32","64","128","256","512","1024"])
# plt.legend(handles=[size_16])
# plt.legend(handles=[size_32])
# plt.legend(handles=[size_64])
# plt.legend(handles=[size_128])
# plt.legend(handles=[size_256])
# plt.legend(handles=[size_512])
# plt.legend(handles=[size_1024])


plt.title('Epocs')
plt.ylabel('Accuracy')

plt.show()


# 16      0.941998    0.953399    0.954899    0.955199    0.955199
# 32      0.956198    0.965198    0.964998    0.965598    0.965698
# 64      0.959197    0.968796    0.970196    0.969896    0.969796
# 128     0.959198    0.971497    0.972797    0.972997    0.972897
# 256     0.951398    0.970698    0.971098    0.970399    0.970398
# 512     0.9439      0.965398    0.967198    0.966998    0.966998
# 1024    0.9372      0.959299    0.960799    0.961098    0.961198