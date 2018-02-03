import numpy as np


def dbmoon(N=1000, d=1, r=10, w=6):
    N1 = 10*N
    w2 = w/2
    done = True
    data = np.empty(0)
    while done:
        tmp_x = 2*(r+w2)*(np.random.random([N1,1])-0.5)
        tmp_y = (r+w2)*np.random.random([N1,1])
        tmp = np.concatenate((tmp_x, tmp_y), 1)
        tmp_ds = np.sqrt(tmp_x*tmp_x + tmp_y*tmp_y)

        idx = np.logical_and(tmp_ds>(r-w2), tmp_ds<(r+w2))
        idx = (idx.nonzero())[0]

        if data.shape[0] == 0:
            data = tmp.take(idx,axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx,axis=0)),axis=0)
        if data.shape[0] >= N:
            done = False

    db_moon1 = np.concatenate((data[0:N,:], np.ones((N,1))), 1)
    data_t = np.empty([N,2])
    data_t[:,0] = data[0:N,0] + r
    data_t[:,1] = -data[0:N,1] - d
    db_moon2 = np.concatenate((data_t[0:N,:], -1*np.ones((N,1))), 1)
    db_moon = np.concatenate((db_moon1, db_moon2), 0)

#    db_moon = data[0:N,:]

#    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon
