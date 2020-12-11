from __future__ import division
from __future__ import absolute_import
import sys, time, os
import numpy as np
import statistics
import scipy.stats
import matplotlib.pyplot as plt
import math
import os
from scipy import linalg
from io import open


class ProMP(object):
    def __init__(self, ndemos=50, hum_dofs=2, robot_dofs=2, dt=0.1, training_address=u"Data\Human A Robot B 1"):
        # assert sys.version_info[0] == 3
        self.ndemos = ndemos
        self.human_dofs = hum_dofs
        self.robot_dofs = robot_dofs
        self.nBasis = 30 # No. of basis functions per time step
        self.noise_stdev = 0.0009
        self.dt = dt

        # data is a list, the ith element of which represents an Nx4 array containing the data of the ith demo
        self.data = self.loadData(training_address)
        print u"Data Loaded"
        # Time normalization and encoding alpha
        self.alpha, self.alphaM, self.alphaV, self.mean_time_steps = self.PhaseNormalization(self.data)
        print u"Alphas Obtained"
        # Normalize the data by writing it in terms of phase variable z. dz = alpha*dt
        self.ndata = self.normalizeData(self.alpha, self.dt, self.data, (self.robot_dofs+self.human_dofs),self.mean_time_steps)
        print u"Data Normalized"
        # Now generate a gaussian basis and find the weights using this basis
        self.promp = self.pmpRegression(self.ndata)
        # self.param = {"nTraj": self.p_data["size"], "nTotalJoints": self.promp["nJoints"],
        #               "observedJointPos": np.array(range(obs_dofs)), "observedJointVel": np.array([])}
        self.alpha_samples = np.linspace(min(self.alpha),max(self.alpha),100)

        # The following 3 variables are used in querying phase only
        self.mu_new = self.promp[u"w"][u"mean_full"]
        self.cov_new = self.promp[u"w"][u"cov_full"]
        self.obsdata = np.empty((0,(self.promp[u"nJoints"])))
        # self.obsndata = np.empty((0, (self.promp["nJoints"])))
        print u"Initialization Complete"

    def loadData(self, addr):
        data = []
        for i in xrange(self.ndemos):
            data.append(np.loadtxt(open(addr + unicode(i + 1) + u""), delimiter=u","))
        print data
        return data

    def PhaseNormalization(self, data):

        sum = 0
        for i in xrange(len(data)):
            sum += len(data[i])
        mean = sum / len(data)
        alpha = []
        for i in xrange(len(data)):
            alpha.append(len(data[i]) / mean)

        alpha_mean = statistics.mean(alpha)
        alpha_var = statistics.variance(alpha)

        return alpha, alpha_mean, alpha_var, int(mean)

    def normalizeData(self, alpha, dt, data, dofs, mean_t):

        # normalize the data to contain same number of data points
        ndata = []
        for i in xrange(len(alpha)):
            demo_ndata = np.empty((0, dofs))
            for j in xrange(mean_t):  # Number of phase steps is same as number of time steps in nominal trajectory, because for nominal traj alpha is 1
                z = j * alpha[i] * dt
                corr_timestep = z / dt
                whole = int(corr_timestep)
                frac = corr_timestep - whole
                row = []
                for k in xrange(dofs):
                    if whole == (data[i].shape[0] - 1):
                        row.append(data[i][whole][k])
                    else:
                        row.append(data[i][whole][k] + frac * (data[i][whole + 1][k] - data[i][whole][k]))
                demo_ndata = np.append(demo_ndata, [row], axis=0)
            ndata.append(demo_ndata)

        return ndata

    def pmpRegression(self, data):
        nJoints = data[0].shape[1]
        nDemo = len(data)
        nTraj = data[0].shape[0]
        nBasis = self.nBasis

        weight = {u"nBasis": nBasis, u"nJoints": nJoints, u"nTraj": nTraj, u"nDemo": nDemo}
        weight[u"my_linRegRidgeFactor"] = 1e-08 * np.identity(nBasis)

        basis = self.generateGaussianBasis(self.dt, nTraj,nBasis)
        weight = self.leastSquareOnWeights(weight, basis, data)

        pmp = {u"w": weight, u"basis": basis, u"nBasis": nBasis, u"nJoints": nJoints, u"nDemo": nDemo,
               u"nTraj": nTraj}

        return pmp

    def generateGaussianBasis(self, dt, nTraj, nBasis):
        basisCenter = np.linspace(0, nTraj * dt, nBasis) # Assuming the average for our basis functions
        sigma = 0.5 * np.ones((1, nBasis))
        z = np.linspace(0, dt * (nTraj-1), nTraj)
        z_minus_center = np.matrix(z).T - np.matrix(basisCenter)
        at = np.multiply(z_minus_center, (1.0 / sigma)) # (z-mu)/sigma for each z, each sigma and each mu.     np.multiply = elementwise multiplication

        basis = np.multiply(np.exp(-0.5 * np.power(at, 2)), 1. / sigma / np.sqrt(2 * np.pi)) #root(2pi)/sigma*e^(-((z-mu)/sigma)^2/2)
        basis_sum = np.sum(basis, axis=1)
        basis_n = np.multiply(basis, 1.0 / basis_sum) # Normalized basis? (Tnom)x(nBasis)

        return basis_n

    def leastSquareOnWeights(self, weight, Gn, data):
        weight[u"demo_q"] = data
        nDemo = weight[u"nDemo"]
        nJoints = weight[u"nJoints"]
        nBasis = weight[u"nBasis"]
        my_linRegRidgeFactor = weight[u"my_linRegRidgeFactor"]

        MPPI = np.linalg.solve(Gn.T * Gn + my_linRegRidgeFactor, Gn.T)

        w, ind = [], []
        for i in xrange(nJoints):
            w_j = np.empty((0, nBasis), float)
            for j in xrange(nDemo):
                w_ = MPPI * np.matrix(data[j][:, i]).T # weights for the jth demo's ith dof (30x1)
                w_j = np.append(w_j, w_.T, axis=0)
            w.append(w_j)
            ind.append(np.matrix(xrange(i * nBasis, (i + 1) * nBasis)))

        weight[u"index"] = ind
        weight[u"w_full"] = np.empty((nDemo, 0), float)
        for i in xrange(nJoints):
            weight[u"w_full"] = np.append(weight[u"w_full"], w[i], axis=1) # nDemos x (nBasis*nJoints)

        weight[u"cov_full"] = np.cov(weight[u"w_full"].T)  # (num of bases*num of dofs) x (num of bases*num of dofs)
        weight[u"mean_full"] = np.mean(weight[u"w_full"], axis=0).T  # (num of bases*num of dofs) x 1

        return weight

    def predict(self, data):
        promp = self.promp
        mu_new = self.mu_new
        cov_new = self.cov_new
        # self.obsdata = np.append(self.obsdata,data,axis=0) # accumulate all the data from previous steps
        # alpha = self.findAlpha(self.obsdata)
        alpha = 1       #self.alpha[42]
        obsndata = data

        mu_new, cov_new = self.conditionNormDist(obsndata, alpha, mu_new, cov_new)
        prdct_data_z = self.weightsToTrajs(promp, mu_new)
        # prdct_data_t = self.z2t(prdct_data_z, alpha)
        return prdct_data_z

    def findAlpha(self, obs):
        # Find log probability of observation given alpha and log probability of alpha and add the two
        # This gives log probability of alpha given observation. The alpha with max of this prob is the winner
        alpha_samples = self.alpha_samples
        alphaM = self.alphaM
        alphaV = self.alphaV
        alpha_dist = scipy.stats.norm(alphaM,alphaV)
        lprob_alphas = []
        inv_obs_noise = 0.00001
        mu_w = self.promp[u"w"][u"mean_full"]
        sig_w = self.promp[u"w"][u"cov_full"]
        dt = self.dt

        for i in xrange(alpha_samples.shape[0]):
            # log probability (actually likelihood) alpha
            lp_alpha = math.log(alpha_dist.pdf(alpha_samples[i]))

            # log probability (actually likelihood) of observation given alpha
            nTraj = int(obs.shape[0]/alpha_samples[i])
            nBasis = self.nBasis
            basis = self.generateGaussianBasis(alpha_samples[i]*dt,nTraj,nBasis)
            lp_obs_alpha = self.computeLogProbObs_alpha(obs,basis,mu_w,sig_w,alpha_samples[i],inv_obs_noise)
            lprob_alphas.append(lp_alpha+lp_obs_alpha)

        best_alpha_index = lprob_alphas.index(max(lprob_alphas))
        best_alpha = alpha_samples[best_alpha_index]

        return best_alpha



    def computeLogProbObs_alpha(self, obs,basis,mu_w,sig_w,alpha_sample,inv_obs_noise):

        length_obs_t = obs.shape[0]
        length_obs_z = int(length_obs_t/alpha_sample)
        if(length_obs_z <= self.mean_time_steps):
            obs_dofs = self.human_dofs
            obs = self.normalizeObservation(obs,alpha_sample)
            obs = obs[:, 0:obs_dofs]
            nTraj = basis.shape[0]
            nBasis = basis.shape[1]
            mu_w = mu_w[np.linspace(0,obs_dofs*nBasis-1,obs_dofs*nBasis,dtype=int),:]
            sig_w = sig_w[0:obs_dofs*nBasis,0:obs_dofs*nBasis]

            # Define A matrix:
            A = np.zeros((obs_dofs*nTraj,obs_dofs*nBasis))
            for i in xrange(obs_dofs):
                A[i*nTraj:(i+1)*nTraj,i*nBasis:(i+1)*nBasis] = basis
            # end Define A matrix:
            a = inv_obs_noise
            u = A*mu_w
            # Writing observation data in a single column
            obs_new = np.zeros((obs.shape[0] * obs.shape[1], 1))
            for i in xrange(obs.shape[1]):
                obs_new[i*obs.shape[0]:(i+1)*obs.shape[0],:] = np.reshape(obs[:,i],(obs[:,i].shape[0],1))
            sigma = (1/a)*np.identity(A.shape[0]) + np.matmul(A,np.matmul(sig_w,A.T))
            sigma_chol = np.linalg.cholesky(2*np.pi*sigma).T # Transposed because in MATLAB, chol() returns upper triang matrix
            diag_sig = np.diag(sigma_chol)
            sum_log_diag = 0
            for i in xrange(diag_sig.shape[0]):
                sum_log_diag = sum_log_diag + math.log(diag_sig[i])
            log_p = -sum_log_diag - (1/2)*np.matmul((obs_new.T-u.T),np.matmul(np.linalg.inv(sigma),(obs_new.T - u.T).T))
            log_p = float(log_p)
        else:
            log_p = -np.inf
        return log_p

    # nBasis = basis.shape[1]
    # obs_dofs = self.human_dofs
    # noise = (self.noise_stdev ** 2) * np.identity(self.promp["nJoints"])
    # mu_w = mu_w[0:obs_dofs*nBasis, :]
    # sig_w = sig_w[0:obs_dofs * nBasis, 0:obs_dofs * nBasis]
    # H = self.observationMatrix(int(obs.shape[0]/alpha_sample),True)
    # A = H[0:obs_dofs,0:nBasis*obs_dofs]
    # mu_obs = np.matmul(A,mu_w)
    # sig_obs = np.matmul(A,np.matmul(sig_w,A.T)) + noise
    # obs_alpha_pdf = scipy.stats.multivariate_normal(mu_obs,sig_obs)

    def observationMatrix(self, k, only_obs=False):  # k = phase step, p = promp
        # Gn_d = p["basis"]["Gndot"]
        p = self.promp
        phaseStep = k
        nJoints = p[u"nJoints"]
        nTraj = p[u"nTraj"]
        Gn = p[u"basis"]
        nBasis = self.nBasis
        H = np.zeros((nJoints, nJoints * nBasis))
        if only_obs:
            for i in xrange(self.human_dofs):
                H[i, np.linspace(i, (i + nBasis - 1), (nBasis), dtype=int)] = Gn[(phaseStep - 1), :]
        else:
            for i in xrange(H.shape[0]):
                H[i, np.linspace(i, (i + nBasis - 1), (nBasis), dtype=int)] = Gn[(phaseStep - 1), :]
        return H


    def normalizeObservation(self, obs_data, alpha):
        human_dofs = self.human_dofs
        dt = self.dt
        ndata = np.empty((0, human_dofs))
        max_z = int(obs_data.shape[0] / alpha)
        for j in xrange(max_z):
            zj_time = j * alpha * dt
            corr_timestep = zj_time / dt
            whole = int(corr_timestep)
            frac = corr_timestep - whole
            data = []
            for k in xrange(human_dofs):
                if whole == (obs_data.shape[0] - 1):
                    data.append(obs_data[whole][k])
                else:
                    data.append(obs_data[whole][k] + frac * (obs_data[whole + 1][k] - obs_data[whole][k]))
            ndata = np.append(ndata, [data], axis=0)
        ndata = np.append(ndata, np.zeros((ndata.shape[0], 2)), axis=1)
        return ndata

    def conditionNormDist(self, obs_data, alpha, mu_new, cov_new):
        promp = self.promp
        sigma_obs = self.noise_stdev
        R_obs = (sigma_obs ** 2) * np.identity(promp[u"nJoints"])
        obs_index = xrange(obs_data.shape[0])
        for k in obs_index:
            H = self.observationMatrix(k,True)
            y0 = np.matrix(obs_data[k,:]).T

            # Conditioning
            tmp = np.matmul(H, np.matmul(cov_new, H.T)) + R_obs
            K = np.matmul(np.matmul(cov_new, H.T), np.linalg.inv(tmp))
            cov_new = cov_new - np.matmul(K, np.matmul(H, cov_new))
            mu_new = mu_new + np.matmul(K, (y0 - np.matmul(H, mu_new)))
            # For each observation, the mu_new is calculated from the mu_new of the last observation

        return mu_new, cov_new

    def weightsToTrajs(self, promp, wts):
        basis = promp[u"basis"]
        nDofs = promp[u"nJoints"]
        traj = np.zeros((basis.shape[0], nDofs))
        for i in xrange(nDofs):
            wi = wts[xrange(i * basis.shape[1], (i + 1) * basis.shape[1])]
            traj[:, i] = np.matmul(basis, wi).flatten()
        return traj

    def z2t(self, zdata, alpha):
        tmax = int(zdata.shape[0] * alpha)
        dt = self.dt
        nDofs = self.promp[u"nJoints"]
        data = np.zeros((tmax, self.promp[u"nJoints"]))

        for j in xrange(tmax):
            tj_time = j * dt
            corr_phasestep = tj_time / (alpha * dt)
            whole = int(corr_phasestep)
            frac = corr_phasestep - whole
            for k in xrange(nDofs):
                if whole == (zdata.shape[0] - 1):
                    data[j, k] = zdata[whole][k]
                else:
                    data[j, k] = zdata[whole][k] + frac * (zdata[(whole + 1), k] - zdata[whole, k])
        return data

    def plotTrajs(self, op_traj, expected_op):
        nDofs = self.promp[u"nJoints"]

        plt.figure(1)
        plt.title(u"Human Trajectory")
        plt.plot(expected_op[:, 0], expected_op[:, 1], label=u"True")
        plt.plot(op_traj[:, 0], op_traj[:, 1], label=u"Prediction")
        plt.legend()

        plt.figure(2)
        plt.title(u"Robot Trajectory")
        plt.plot(expected_op[:, 2], expected_op[:, 3], label=u"True")
        plt.plot(op_traj[:, 2], op_traj[:, 3], label=u"Prediction")
        plt.legend()

        plt.figure(3)
        plt.title(u"Human x")
        plt.plot(expected_op[:, 0], label=u"True")
        plt.plot(op_traj[:, 0], label=u"Prediction")
        plt.legend()

        plt.figure(4)
        plt.title(u"Human y")
        plt.plot(expected_op[:, 1], label=u"True")
        plt.plot(op_traj[:, 1], label=u"Prediction")
        plt.legend()

        plt.figure(5)
        plt.title(u"Robot x")
        plt.plot(expected_op[:, 2], label=u"True")
        plt.plot(op_traj[:, 2], label=u"Prediction")
        plt.legend()

        plt.figure(6)
        plt.title(u"Robot y")
        plt.plot(expected_op[:, 3], label=u"True")
        plt.plot(op_traj[:, 3], label=u"Prediction")
        plt.legend()

        plt.show()

    def resetProMP(self):
        self.mu_new = self.promp[u"w"][u"mean_full"]
        self.cov_new = self.promp[u"w"][u"cov_full"]
        self.obsdata = np.empty((0, (self.promp[u"nJoints"])))
        # self.obsndata = np.empty((0, (self.promp["nJoints"])))


def printarr(arr, name=u"None"):  # This function is only for debugging purposes
    print u"VariableName: ", name
    for i in xrange(arr.shape[0]):
        print arr[i, :]
def comparePlots(ip_traj, op_traj):  # This function is only for debugging purposes
    plt.figure(1)
    plt.plot(ip_traj[:, 0], ip_traj[:, 1])
    plt.show()
    plt.plot(op_traj[:, 0], op_traj[:, 1])
    plt.show()
def trajectorizePoints(test_point,pmp):
    test_data = np.zeros((len(pmp.ndata[0]),7))
    for i in xrange(len(pmp.ndata[0])):
        test_data[i,0:4] = test_point
    return test_data
    # plt.figure(2)
    # plt.plot(ip_traj[:, 2], ip_traj[:, 3])
    # plt.plot(op_traj[:, 2], op_traj[:, 3])


def main(args):

    # data_addr = '../../training_data/ProMP/'
    data_addr = u'/home/thejus/catkin_ws/src/movement_primitives/training_data/ProMP/'
    ndemos = len(os.listdir(data_addr))
    print ndemos
    pmp = ProMP(ndemos=ndemos, hum_dofs=4, robot_dofs=3, dt=0.1, training_address=data_addr)
    # To give different inputs change the parameters passed to the constructor above.
    # ndemos = No. of demos, hum_dofs = master dofs, robot_dofs = slave dofs, dt = delta t between 2 position measurements in a demo
    # training_address = path of the folder in which your data is present
    # Also to be changed: In loadData line 44 and 45, change the name of your master data and slave data file name (without the demo number).
    # Default master data file name: letterAtr followed by demo number.
    # Default slave data file name: letterBtr followed by demo number.

    # test_data = np.loadtxt(open("Data/Human A Robot B 1/letterAtr43.csv"), delimiter=",")
    # num_pts = 90  # Enter the number of points you want in your input
    # test_data = np.delete(test_data, np.linspace(num_pts, (len(test_data) - 1), (len(test_data) - num_pts), dtype=int),axis=0)  # Trimming data
    # test_data = np.append(test_data, np.zeros((test_data.shape[0], 2)), axis=1)

    # Expected observation data format: [col(obs of human dof1),col(obs of human dof2),...,col(obs of last human dof), (columns of zeros for each robot dof)]
    test_point = [500,1000,1500,1000]
    test_data = trajectorizePoints(test_point,pmp)
    # test_data = np.zeros((len(pmp.ndata[0]),7))
    # for i in range(len(pmp.ndata[0])):
    #     test_data[i,0:4] = test_point
    traj = pmp.predict(test_data)
    plt.figure(1)
    plt.plot(traj[:,4],traj[:,5])
    plt.plot(test_point[0],test_point[1],u'ro')
    plt.plot(test_point[2], test_point[3], u'ro')
    plt.show()
    # print("\nPrediction Executed Successfully \nPlotting...")
    # If you want to use sequential observation data, at each time step, obtain the observation in the "expected format" (line 394) with a single row
    # and pass it in pmp.predict as test_data.

    # Now to plot the results
    # ip_traj = pmp.data[42]
    # pmp.plotTrajs(traj,ip_traj)
    # pmp.resetProMP()

if __name__ == u'__main__':
    main(sys.argv)







# Commands for test variables:
# test1 = [np.round(np.random.rand(3,2)*10),np.round(np.random.rand(3,2)*10)]
# test2 = []
# test2.append(test1[0][:,0].T)
# test2.append(test1[1][:,0].T)
# a = np.loadtxt(open('Data\Human A Robot B\letterAtr2.csv'),delimiter=",")
# A.append(a)
# A.append(a)
