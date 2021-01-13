import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import copy

class ReferenceTrajectoryPoint:
    def __init__(self, t=None, mu=None, sigma=None):
        self.t = t
        self.mu = mu
        self.sigma = sigma


class KMP: #Assumptions: Input is only time; All dofs of output are continuous TODO: Generalize
    def __init__(self, input_dofs, robot_dofs, demo_dt, ndemos, data_address):
        self.input_dofs = input_dofs
        self.robot_dofs = robot_dofs
        self.ndemos = ndemos
        self.demo_dt = demo_dt

        num_gmm_comps = 8

        self.training_data = self.loadData(addr=data_address) # Fetching the data from the saved location
        self.norm_data = self.normalizeData(self.training_data) # Making all demos have the same length
        self.demo_duration = self.training_data[0].shape[0] * self.demo_dt
        self.data = self.combineData(self.norm_data)
        # plt.scatter(self.data[:,1], self.data[:,2])

        self.gmm_model = GaussianMixture(n_components=num_gmm_comps, random_state=0)
        self.model_num_datapts = int(self.demo_duration/self.demo_dt)
        self.data_out, self.sigma_out, _ = self.GMR(self.gmm_model, np.array(range(self.model_num_datapts)) * self.demo_dt, range(self.input_dofs), range(1,(self.input_dofs+self.robot_dofs)))

        self.ref_traj = []
        for i in range(self.model_num_datapts):
            self.ref_traj.append(ReferenceTrajectoryPoint(t=i*self.gmm_model.dt, mu=self.data_out[:,i], sigma=self.sigma_out[:,:,i]))

        print('KMP Initialized with Reference Trajectory')
    ###################################


    def loadData(self, addr):
        data = []
        for i in range(self.ndemos):
            data.append(np.loadtxt(open(addr + str(i + 1) + ""), delimiter=","))
        # data is saved as a list of nparrays [nparray(Demo1),nparray(Demo2),...]
        return data
    #########################

    def normalizeData(self, data):
        dofs = self.input_dofs + self.robot_dofs
        dt = self.demo_dt

        sum = 0
        for i in range(len(data)):
            sum += len(data[i])
        mean = sum / len(data)
        alpha = []
        for i in range(len(data)):
            alpha.append(len(data[i]) / mean)

        alpha_mean = np.mean(alpha)
        mean_t = int(mean)

        # normalize the data to contain same number of data points
        ndata = []
        for i in range(len(alpha)):
            demo_ndata = np.empty((0, dofs))
            for j in range(
                    mean_t):  # Number of phase steps is same as number of time steps in nominal trajectory, because for nominal traj alpha is 1
                z = j * alpha[i] * dt
                corr_timestep = z / dt
                whole = int(corr_timestep)
                frac = corr_timestep - whole
                row = []
                for k in range(dofs):
                    if whole == (data[i].shape[0] - 1):
                        row.append(data[i][whole][k])
                    else:
                        row.append(data[i][whole][k] + frac * (data[i][whole + 1][k] - data[i][whole][k]))
                demo_ndata = np.append(demo_ndata, [row], axis=0)
            ndata.append(demo_ndata)

        return ndata
    ############################

    def combineData(self,data):
        # total_time_steps = 0
        # for i in range(len(data)):
        #     total_time_steps = total_time_steps + data[i].shape[0]
        # time = np.array(range(total_time_steps)) * self.demo_dt
        positions = data[0].T
        for i in range(1,len(data)):
            positions = np.append(positions,data[i].T,axis=1)
        # data = np.vstack((time,positions))
        return positions
    #############################

    def GMR(self, model, data_in_raw, input_dofs_list, output_dofs_list):
        data_in = data_in_raw.reshape((1,data_in_raw.shape[0]))
        num_datapts = data_in.shape[1]
        num_varout = len(output_dofs_list)
        diag_reg_factor = 1e-8

        mu_tmp = np.zeros((num_varout, self.gmm_model.n_components))
        exp_data = np.zeros((num_varout, num_datapts))
        exp_sigma = np.zeros((num_varout, num_varout, num_datapts))

        H = np.empty((self.gmm_model.n_components, num_datapts))
        for t in range(num_datapts):
            # Compute activation weights
            for i in range(self.gmm_model.n_components):
                H[i,t] = self.gmm_model.priors[i] * gaussPDF(data_in[:,t].reshape((-1,1)), self.gmm_model.mu[input_dofs_list,i], self.gmm_model.sigma[input_dofs_list,input_dofs_list,i])
            H[:,t] = H[:,t]/(sum(H[:,t]) + 1e-10)
            # Compute conditional means
            for i in range(self.gmm_model.n_components):
                mu_tmp[:,i] = self.gmm_model.mu[output_dofs_list,i] + self.gmm_model.sigma[output_dofs_list,input_dofs_list,i]/self.gmm_model.sigma[input_dofs_list,input_dofs_list,i] * (data_in[:,t].reshape((-1,1)) - self.gmm_model.mu[input_dofs_list,i])
                exp_data[:,t] = exp_data[:,t] + H[i,t] * mu_tmp[:,i]
            # Compute conditional covariance
            for i in range(self.gmm_model.n_components):
                sigma_tmp = self.gmm_model.sigma[output_dofs_list,output_dofs_list,i] - self.gmm_model.sigma[output_dofs_list,input_dofs_list,i]/self.gmm_model.sigma[input_dofs_list,input_dofs_list,i] * self.gmm_model.sigma[input_dofs_list,output_dofs_list,i]
                exp_sigma[:,:,t] = exp_sigma[:,:,t] + H[i,t] * (sigma_tmp + mu_tmp[:,i]*mu_tmp[:,i].T)
            exp_sigma[:,:,t] = exp_sigma[:,:,t] - exp_data[:,t] * exp_data[:,t].T + np.eye(num_varout) * diag_reg_factor
        return exp_data, exp_sigma, H

    def setParams(self, dt, lamda, kh):
        self.dt = dt
        self.len = int(self.demo_duration/dt)
        self.lamda = lamda
        self.kh = kh
    ##################################

    def addViaPts(self, via_pts, via_pt_var):
        # Search for closest point in ref trajectory
        self.new_ref_traj = copy.deepcopy(self.ref_traj)
        replace_ind = 0
        for via_pt in via_pts:
            min_dist = distBWPts(self.ref_traj[0].mu[0:2],via_pt)
            for i in range(len(self.ref_traj)):
                dist = distBWPts(self.ref_traj[i].mu[0:2],via_pt)
                if dist<min_dist:
                    min_dist = dist
                    replace_ind = i
            via_pt = np.append(np.array(via_pt),self.ref_traj[replace_ind].mu[2:4])
            self.new_ref_traj[replace_ind] = ReferenceTrajectoryPoint(t=self.ref_traj[replace_ind].t, mu=via_pt, sigma=via_pt_var)
    ###################################

    def estimateMatrixMean(self):
        D = self.robot_dofs
        N = self.len
        kc = np.empty((D*N, D*N))
        for i in range(N):
            for j in range(N):
                kc[i*D:(i+1)*D, j*D:(j+1)*D] = kernelExtend(self.new_ref_traj[i].t, self.new_ref_traj[j].t, self.kh, self.robot_dofs)
                if i==j:
                    c_temp = self.new_ref_traj[i].sigma
                    kc[i*D:(i+1)*D, j*D:(j+1)*D] = kc[i*D:(i+1)*D, j*D:(j+1)*D] + self.lamda * c_temp

        Kinv = np.linalg.inv(kc)
        return Kinv
    ###############################

    def prediction(self, via_pts, via_pt_var):
        self.addViaPts(via_pts, via_pt_var)
        Kinv = self.estimateMatrixMean()
        self.kmp_pred_traj = []
        for index in range(self.len):
            t = index * self.dt
            mu = self.kmpPredictMean(t, Kinv)
            self.kmp_pred_traj.append(ReferenceTrajectoryPoint(t=index*self.dt, mu=mu))
        return self.kmp_pred_traj
    #################################

    def kmpPredictMean(self, t, Kinv):
        D = self.robot_dofs
        N = self.len

        k = np.empty((D,N*D))
        Y = np.empty((N * D, 1))
        for i in range(N):
            k[0:D, i*D:(i+1)*D] = kernelExtend(t, self.new_ref_traj[i].t, self.kh, D)
            for h in range(D):
                Y[i*D+h] = self.new_ref_traj[i].mu[h]
        return np.matmul(np.matmul(k,Kinv),Y)


###############################################################################
# Functions
def gaussPDF(data, mu, sigma):
    num_vars, num_datapts = data.shape
    data = data.T - np.repeat(mu.reshape((1, mu.shape[0])), [num_datapts], axis=0)
    if num_vars==1 and num_datapts==1:
        prob = data**2 * sigma
        prob = np.e ** (-0.5 * prob) / (np.sqrt((2 * np.pi) ** num_vars * sigma))
    else:
        prob = np.sum(np.multiply(np.matmul(data, np.linalg.inv(sigma)), data), axis=1)
        prob = np.e ** (-0.5 * prob) / (np.sqrt((2 * np.pi) ** num_vars * abs(np.linalg.det(sigma))))
    return prob
#################

def distBWPts(pt1, pt2):
    return abs(np.linalg.norm(pt1,2) - np.linalg.norm(pt2,2))
##################

def kernelExtend(ta, tb, h, dim):
    dt = 0.01
    tadt = ta + dt
    tbdt = tb + dt

    kt_t = np.e**(-h * (ta-tb) * (ta-tb))

    kt_dt_temp = np.e**(-h * (ta-tbdt) * (ta-tbdt))
    kt_dt = (kt_dt_temp - kt_t)/dt

    kdt_t_temp = np.e**(-h*(tadt-tb)*(tadt-tb))
    kdt_t = (kdt_t_temp - kt_t) / dt

    kdt_dt_temp = np.e**(-h * (tadt - tbdt) * (tadt - tbdt))
    kdt_dt = (kdt_dt_temp - kt_dt_temp - kdt_t_temp + kt_t) / dt / dt

    kernel_matrix = np.zeros((dim,dim))
    for i in range(int(dim/2)):
        kernel_matrix[i,i] = kt_t
        kernel_matrix[i,i+int(dim/2)] = kt_dt
        kernel_matrix[i+int(dim/2), i] = kdt_t
        kernel_matrix[i+int(dim/2), i+int(dim/2)] = kdt_dt

    return kernel_matrix

def extractPath(traj):
    path = np.empty((len(traj),4))
    for i in range(len(traj)):
        path[i,:] = (traj[i].mu).flatten()
    return path


###############################################################################

if __name__ == "__main__":
    data_addr = '../../training_data/KMP_static/'
    ndemos = len(os.listdir(data_addr))
    # ndemos = 10
    kmp = KMP(input_dofs=1, robot_dofs=4, demo_dt=0.1, ndemos=ndemos, data_address=data_addr)

    # Set KMP params (This dt is KMP's dt)
    kmp.setParams(dt=0.05, lamda=1, kh=6)
    # Set desired via points
    via_pts = [[500,1000],[1500,1000]]
    via_pt_var = 1e-6 * np.eye(kmp.ref_traj[0].sigma.shape[0])

    #KMP Prediction
    pred_traj = kmp.prediction(via_pts,via_pt_var)
    pred_path = extractPath(pred_traj)
    ref_traj = kmp.ref_traj
    ref_path = extractPath(ref_traj)

    plt.figure(1)
    plt.plot(pred_path[:, 0], pred_path[:, 1], 'r')
    # plt.plot(ref_path[:, 0], ref_path[:, 1], 'g')
    plt.plot(via_pts[0][0], via_pts[0][1], 'bo')
    plt.plot(via_pts[1][0], via_pts[1][1], 'bo')
    plt.show()

