"""
Calculate unit normals or tangents of points for given x,y-coordinates of line

Author:  A. Habermann
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_normals(x: list, y: list):
    if len(x) != len(x):
        Warning('X and y must have same length for normal calculation.')
    elif len(x) <= 2:
        Warning('More than two values required for normal calculation.')

    else:
        dx = [0]*len(x)
        dy = [0]*len(x)
        normals_dim = [[0, 0]]*len(x)
        normals = [[0, 0]]*len(x)
        for i in range(0, len(x)):
            # exception for last coordinate
            if i == len(x)-1:
                dx[i] = (x[i]-x[i-1])
                dy[i] = (y[i]-y[i-1])
            # all other coordinates
            else:
                dx[i] = (x[i+1]-x[i])
                dy[i] = (y[i+1]-y[i])

            normals_dim[i] = [-dy[i], 0, dx[i]]
            # scale normal vector to unit vector
            normals[i] = np.dot(normals_dim[i],1/np.linalg.norm(normals_dim[i]))

    return normals


def calculate_tangents(x: list, y: list):
    if len(x) != len(x):
        Warning('X and y must have same length for tangent calculation.')
    elif len(x) <= 2:
        Warning('More than two values required for tangent calculation.')

    else:
        dx = [0]*len(x)
        dy = [0]*len(x)
        tangents_dim = [[0, 0]]*len(x)
        tangents = [[0, 0]]*len(x)
        for i in range(0, len(x)):
            # exception for last coordinate
            if i == len(x)-1:
                dx[i] = (x[i]-x[i-1])
                dy[i] = (y[i]-y[i-1])
            # all other coordinates
            else:
                dx[i] = (x[i+1]-x[i])
                dy[i] = (y[i+1]-y[i])

            tangents_dim[i] = [dx[i], 0, dy[i]]
            # scale normal vector to unit vector
            tangents[i] = np.dot(tangents_dim[i],1/np.linalg.norm(tangents_dim[i]))

    return tangents


if __name__ == "__main__":
    x1 = [0,         0.00264651, 0.0053018,  0.00796585, 0.01063867,0.01332025,
     0.01601061, 0.01870973 ,0.02141763 ,0.02413429 ,0.02685972, 0.02959391,
     0.03233688 ,0.03508861 ,0.03784912 ,0.04061839 ,0.04339643, 0.04618324,
     0.04897881, 0.05178316 ,0.05459627 ,0.05741815 ,0.0602488 , 0.06308822,
     0.06593641 ,0.06879336 ,0.07165908 ,0.07453358 ,0.07741684, 0.08030887,
     0.08320966 ,0.08611923 ,0.08903756 ,0.09196466, 0.09490053, 0.09784517,
     0.10079858 ,0.10376076 ,0.1067317 , 0.10971141, 0.11269989, 0.11569714,
     0.11870316 ,0.12171795 ,0.1247415,  0.12777382, 0.13081492, 0.13386478,
     0.1369234  ,0.1399908  ,0.14306696, 0.1461519 , 0.1492456 , 0.15234807,
     0.15545931 ,0.15857931 ,0.16170809 ,0.16484563, 0.16799194, 0.17114702,
     0.17431087 ,0.17748349 ,0.18066488, 0.18385503, 0.18705395, 0.19026164,
     0.1934781  ,0.19670333 ,0.19993732 ,0.20318009 ,0.20643162, 0.20969192,
     0.21296099 ,0.21623883 ,0.21952544, 0.22282081 ,0.22612495, 0.22943786,
     0.23275954 ,0.23608999 ,0.23942921 ,0.24277719 ,0.24613395, 0.24949947,
     0.25287376 ,0.25625682 ,0.25964864 ,0.26304924 ,0.2664586 , 0.26987673,
     0.27330363 ,0.2767393  ,0.28018374, 0.28363694 ,0.28709892 ,0.29056966,
     0.29404917 ,0.29753745 ,0.3010345 , 0.30454031]

    y1 = [0., - 0.00232913 ,- 0.0046307 ,- 0.00690471,- 0.00915117 , -0.01137006,
      - 0.01356139, - 0.01572517 ,- 0.01786139, - 0.01997004 ,- 0.02205114, - 0.02410468,
      - 0.02613066, - 0.02812908,- 0.03009994 ,- 0.03204324 ,- 0.03395898,-       0.03584717,
      - 0.03770779 ,- 0.03954085 ,- 0.04134636, - 0.04312431, - 0.04487469, -       0.04659752,
      - 0.04829279 ,- 0.0499605 ,- 0.05160065 ,- 0.05321324, - 0.05479827, -       0.05635574,
      - 0.05788566 ,- 0.05938801, - 0.06086281, - 0.06231004, - 0.06372972, -       0.06512184,
      - 0.0664864, - 0.06782339 ,- 0.06913283 ,- 0.07041471, - 0.07166904 ,-       0.0728958,
      - 0.074095, - 0.07526664, - 0.07641073 ,- 0.07752725, - 0.07861622 ,-       0.07967763,
      - 0.08071147, - 0.08171776 ,- 0.08269649 ,- 0.08364766, - 0.08457127 ,-       0.08546733,
      - 0.08633582, - 0.08717675, - 0.08799012 ,- 0.08877594,- 0.0895342 ,-       0.09026489,
      - 0.09096803, - 0.09164361, - 0.09229163, - 0.09291208, - 0.09350499,-       0.09407033,
      - 0.09460811, - 0.09511833 ,- 0.09560099 ,- 0.0960561, - 0.09648364, -       0.09688363,
      - 0.09725606, - 0.09760092 ,- 0.09791823, - 0.09820798, - 0.09847017,-       0.0987048,
      - 0.09891187, - 0.09909139 ,- 0.09924334 ,- 0.09936773, - 0.09946457, -       0.09953384,
      - 0.09957556, - 0.09958972 ,- 0.09957631, - 0.09953535 ,- 0.09946683, -       0.09937075,
      - 0.09924711, - 0.09909591, - 0.09891716, - 0.09871084, - 0.09847696, -       0.09821553,
      - 0.09792653, - 0.09760998 ,- 0.09726587 ,- 0.0968942]

    normals_10 = calculate_normals(x1, y1)
    tangents_10 = calculate_tangents(x1, y1)

    tangents_2 = np.cross(normals_10[0], [0,-1,0])

    plt.plot(x1, y1)
    plt.scatter(x1, y1)
    plt.plot((np.array(x1), np.array(x1)+np.array([item[0] for item in normals_10])),
             (np.array(y1), np.array(y1)+np.array([item[2] for item in normals_10])))
    plt.axis('equal')
    plt.show()

    plt.plot(x1, y1)
    plt.scatter(x1, y1)
    plt.plot((np.array(x1), np.array(x1)+np.array([item[0] for item in tangents_10])),
             (np.array(y1), np.array(y1)+np.array([item[2] for item in tangents_10])))
    plt.axis('equal')
    plt.show()
