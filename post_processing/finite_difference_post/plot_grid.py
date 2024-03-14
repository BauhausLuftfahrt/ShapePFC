"""Plot grids..

Author:  A. Habermann

"""

# Built-in/Generic Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plotGrid(x, y, flags):
    """
        x:  x-coordinates
        y:  y-coordinates
        flags:  node flags 0: inner point; 1: surface; 2: inlet; 3: outlet; 4: farfield; 5: center axis
    """

    plt.close()
    plt.scatter(x[np.where(flags==0)],y[np.where(flags==0)],color="k",marker=".")
    plt.scatter(x[np.where(flags==1)],y[np.where(flags==1)],color="r",marker="o")
    plt.scatter(x[np.where(flags==12)],y[np.where(flags==12)],color="r",marker="v")
    plt.scatter(x[np.where(flags==11)],y[np.where(flags==11)],color="r",marker="^")
    plt.scatter(x[np.where(flags==2)],y[np.where(flags==2)],color="g",marker=">")
    plt.scatter(x[np.where(flags==3)],y[np.where(flags==3)],color="y",marker=">")
    plt.scatter(x[np.where(flags==4)],y[np.where(flags==4)],color="b",marker="^")
    plt.scatter(x[np.where(flags==5)],y[np.where(flags==5)],color="c",marker="x")
    plt.scatter(x, y, s=0.05)
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    plt.gca().add_collection(LineCollection(segs1,linewidth=0.2))
    plt.gca().add_collection(LineCollection(segs2,linewidth=0.2))


def plotBoundaries(boundaries):
    # plot boundaries of grid
    for i in range(0,len(boundaries)):
        plt.plot(boundaries[i][0],boundaries[i][1],linewidth=0.2)
        plt.scatter(boundaries[i][0],boundaries[i][1],s=0.05)
    #plt.show()
