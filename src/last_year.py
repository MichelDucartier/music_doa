import numpy as np


def generate_data(M, N, P, d, wavelen, w, doa, var):
    E = np.exp(-1j*2*np.pi*d/wavelen* np.kron(np.arange(M), np.sin(doa)).reshape((M, P)))
    x0 = 2*np.exp(1j*(np.kron(w, np.arange(N)).reshape((P, N)))) 
    X = np.dot(E, x0)
    Y = X + var*np.random.randn(M, N) 
    return Y


def music(X, f, P, M, locations):
    wavelen = 343 / f
    locations = locations - locations[0,:]
    
    Xmean = np.mean(X, axis=1)
    X0 = X - np.tile(Xmean, (np.shape(X)[1],1)).T
    R = np.dot(X0, X0.conj().T)
    # if M==8:
    J = np.flip(np.eye(M), axis=1)
    R = R + np.dot(J, np.dot(R.conj(), J))
    w, v = np.linalg.eig(R)
    ids = np.abs(w).argsort()[:(M-P)] # find the smallest eignvalues
    En = v[:,ids]
    Ren = np.dot(En, En.conj().T)
    
    if M==8:
        theta = np.arange(-90, 90, 1)
        d = np.array([0, 0.04, 0.08, 0.098, 0.102, 0.12, 0.16, 0.2])
        atheta = np.exp(-1j*2*np.pi/wavelen*np.kron(d, np.sin(theta/180*np.pi)).reshape(8, np.size(theta)))
    else:
        theta = np.arange(0, 360, 1)
        a = np.array([np.cos(theta/180*np.pi), np.sin(theta/180*np.pi)])
        atheta = np.exp(-1j*2*np.pi/wavelen*np.dot(locations, a))

    l = np.size(theta)
    Pmusic = np.zeros(l)
    for j in range(l):
        Pmusic[j] = 1/abs(np.dot(np.dot(atheta[:,j].conj().T, Ren), atheta[:,j]))

    return Pmusic, theta


