import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn import datasets

def make_ellipses(gmm, ax):
    color = 'navy'
    for n in range(gmm.n_components):
        covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')

def plot_gmms(data, gmm=None, n_comp = None):
    if gmm==None:
        gmm = GaussianMixture(n_components=n_comp, random_state=0)
        gmm.fit(data)
    ax = plt.subplot(1, 1, 1)
    make_ellipses(gmm, ax)
    plt.scatter(data[:, 0], data[:, 1], s=0.8, color='red')
    # x = [[1500, 1080]]
    # plt.plot(x[0][0], x[0][1], 'bo')
    # comp = gmm.predict(x)[0]
    # plt.plot(gmm.means_[comp][0], gmm.means_[comp][1], 'go')
    for i in range(gmm.n_components):
        plt.text(gmm.means_[i][0],gmm.means_[i][1],str(i+1),color='blue')

    plt.show()

if __name__ == '__main__':
    data = []
    data_addr = '../../training_data/TP-HSMM/'
    for i in range(35):
        data.append(np.loadtxt(open(data_addr + str(i + 1) + ''), delimiter=","))
    positions = np.array(data[0])
    for i in range(1, len(data)):
        positions = np.append(positions, data[i], axis=0)
    positions = np.delete(positions,[2,3,4,5],1)
    gmm = GaussianMixture(n_components=10, random_state=0)
    gmm.fit(positions)

    plot_gmms(positions, gmm)
    # gmm = GaussianMixture(n_components=10,random_state=0)
    # gmm.fit(positions)
    # ax = plt.subplot(1,1,1)
    # make_ellipses(gmm,ax)
    # plt.scatter(positions[:, 0], positions[:, 1], s=0.8, color='red')
    # x = [[1500,1080]]
    # plt.plot(x[0][0],x[0][1],'bo')
    # comp = gmm.predict(x)[0]
    # plt.plot(gmm.means_[comp][0],gmm.means_[comp][1], 'go')
    #
    # plt.show()
