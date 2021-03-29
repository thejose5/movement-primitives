import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import copy

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

class ReferenceTrajectoryPoint:
    def __init__(self, t=None, mu=None, sigma=None):
        self.t = t
        self.mu = mu
        self.sigma = sigma

class GMM:
    def __init__(self, num_vars, data, num_states=8, dt=0.05):
        self.num_vars = num_vars
        self.num_states = num_states
        self.dt = dt #dt for the GMM model
        self.init_GMM_timebased(data) #Initialize the GMM model with priors, means and co/variances

        #Defining params for expectation maximization algorithm
        self.em_num_min_steps = 5
        self.em_num_max_steps = 100
        self.em_max_diffLL = 1e-4 #Likelihood increase threshold to stop iterations
        self.em_diag_reg_factor = 1e-4
        self.em_update_comp = np.ones((3,1))
        self.EM_GMM(data)

    def init_GMM_timebased(self, data):
        num_vars, num_datapts = data.shape
        diag_reg_factor = 1e-8
        timing_sep = np.linspace(min(data[0,:]), max(data[0,:]), self.num_states+1) #The timings between which we cluster datapoints

        self.priors = np.empty((self.num_states,))
        self.mu = np.empty((self.num_vars,self.num_states))
        self.sigma = np.empty((self.num_states, self.num_vars, self.num_vars))
        for i in range(self.num_states):
            idtmp = np.where((timing_sep[i]<=data[0,:]) & (data[0,:]<timing_sep[i+1]))[0] # Extracts the indices of data points that lie within the ith time bracket
            self.priors[i] = idtmp.shape[0] #This will be changed to a normalized value after the for loop
            self.mu[:,i] = [np.mean(data[j,idtmp]) for j in range(self.num_vars)] # Average of each row
            self.sigma[i,:,:] = np.cov(data[:,idtmp])
            self.sigma[i,:,:] = self.sigma[i,:,:] + np.eye(num_vars)*diag_reg_factor # Adding regularization factor
        self.priors = self.priors/sum(self.priors)

    def EM_GMM(self, data):
        LL = []
        for iter in range(self.em_num_max_steps):
            print(".",end="")
            #Expectation
            L, gamma = self.computeGamma(data)
            gamma_colwise_sum = np.sum(gamma,axis=1).reshape((np.sum(gamma,axis=1).shape[0],1))
            gamma2 = np.divide(gamma, np.repeat(gamma_colwise_sum,data.shape[1],axis=1))
            #Maximization
            for i in range(self.num_states):
                #Update Priors
                self.priors[i] = sum(gamma[i,:])/data.shape[1]
                #Update Mu
                self.mu[:,i] = (np.matmul(data, gamma2[i,:].reshape((gamma2.shape[1],1)))).reshape(self.mu[:,i].shape)
                #Update sigma
                datatmp = data - np.repeat(self.mu[:,i].reshape((self.mu.shape[0],1)),data.shape[1],1)
                self.sigma[i,:,:] = np.matmul(np.matmul(datatmp, np.diag(gamma2[i,:])), datatmp.T) + np.eye(data.shape[0]) * self.em_diag_reg_factor
                # self.sigma[:, :, i] = np.asarray(np.asmatrix(datatmp) * np.asmatrix(np.diag(gamma2[i,:])) * np.asmatrix(datatmp.T)) + np.eye(data.shape[0]) * self.em_diag_reg_factor
            LL.append(sum(np.log(np.sum(L,axis=0)))/data.shape[1]) # Average Log likelihood
            if iter>=self.em_num_min_steps:
                if LL[iter]-LL[iter-1]<self.em_max_diffLL or iter==self.em_num_max_steps-1:
                    print('EM converged after ',str(iter),' iterations.')
                    return
        print('Max no. of iterations reached')
        return


    def computeGamma(self,data):
        L = np.zeros((self.num_states,data.shape[1]))
        for i in range(self.num_states):
            L[i,:] = self.priors[i]*gaussPDF(data,self.mu[:,i],self.sigma[i,:,:])
        L_axis0_sum = np.sum(L,axis=0)
        gamma = np.divide(L, np.repeat(L_axis0_sum.reshape(1,L_axis0_sum.shape[0]), self.num_states, axis=0))
        return L,gamma

#######################################################################################################################

class KMP: #Assumptions: Input is only time; All dofs of output are continuous TODO: Generalize
    def __init__(self, input_dofs, robot_dofs, demo_dt, ndemos, data_address, via_pts = [[0.89,0.37],[0.5,0.9],[1.5,0.75],[1.06,0.37]]):
        self.input_dofs = input_dofs
        self.robot_dofs = robot_dofs
        self.ndemos = ndemos
        self.demo_dt = demo_dt
        self.via_pts = via_pts

        self.training_data = self.loadData(addr=data_address) # Fetching the data from the saved location
        self.norm_data = self.normalizeData(self.training_data) # Making all demos have the same length
        self.demo_duration = 200 * self.demo_dt #self.training_data[0].shape[0] * self.demo_dt
        self.data = self.combineData(self.norm_data)

        self.gmm_model = GMM(num_vars=(self.input_dofs+self.robot_dofs), data=self.data)
        self.model_num_datapts = int(self.demo_duration/self.gmm_model.dt)
        self.data_out, self.sigma_out, _ = self.GMR(np.array(range(1,self.model_num_datapts+1)) * self.gmm_model.dt)

        ####### DEBUGGING ##############
        # plt.scatter(self.data[1,:],self.data[2,:])
        # for i in range(self.gmm_model.num_states):
        #     plt.plot(self.gmm_model.mu[1,i],self.gmm_model.mu[2,i], 'ro')
        # plt.show()
        ##################################

        self.ref_traj = []
        self.ref_traj_mus = []
        for i in range(self.model_num_datapts):
            self.ref_traj.append(ReferenceTrajectoryPoint(t=(i+1)*self.gmm_model.dt, mu=self.data_out[:,i], sigma=self.sigma_out[i,:,:]))
            self.ref_traj_mus.append(self.data_out[:,i])
        self.via_pt_inds = self.findViaPointIndices()
        ####### DEBUGGING ##############
        ref_path = extractPath(self.ref_traj)
        plt.plot(ref_path[:, 0], ref_path[:, 1], 'r')
        for i in range(len(via_pts)):
            plt.plot(via_pts[i][0],via_pts[i][1],'bo')
            plt.plot(self.ref_traj_mus[self.via_pt_inds[i]][0], self.ref_traj_mus[self.via_pt_inds[i]][1], 'bo')
        plt.title('Reference Path')
        plt.show()
        ##################################
        print(self.via_pt_inds)
        print(self.ref_traj_mus)
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

        # normalize the data so that all demos contain same number of data points
        ndata = []
        for i in range(len(alpha)):
            demo_ndata = np.empty((0, dofs))
            for j in range(mean_t):  # Number of phase steps is same as number of time steps in nominal trajectory, because for nominal traj alpha is 1
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

    def GMR(self, data_in_raw):
        data_in = data_in_raw.reshape((1,data_in_raw.shape[0]))
        num_datapts = data_in.shape[1]
        num_varout = self.robot_dofs
        diag_reg_factor = 1e-8

        mu_tmp = np.zeros((num_varout, self.gmm_model.num_states))
        exp_data = np.zeros((num_varout, num_datapts))
        exp_sigma = np.zeros((num_datapts, num_varout, num_varout))

        H = np.empty((self.gmm_model.num_states, num_datapts))
        for t in range(num_datapts):
            # Compute activation weights
            for i in range(self.gmm_model.num_states):
                H[i,t] = self.gmm_model.priors[i] * gaussPDF(data_in[:,t].reshape((-1,1)), self.gmm_model.mu[0:self.input_dofs,i], self.gmm_model.sigma[i,0:self.input_dofs,0:self.input_dofs])
                # print(gaussPDF(data_in[:,t].reshape((-1,1)), self.gmm_model.mu[0:self.input_dofs,i], self.gmm_model.sigma[0:self.input_dofs,0:self.input_dofs,i]))
            H[:,t] = H[:,t]/(sum(H[:,t]) + 1e-10)
            # print(H)
            # Compute conditional means
            for i in range(self.gmm_model.num_states):
                mu_tmp[:,i] = self.gmm_model.mu[self.input_dofs:(self.input_dofs+self.robot_dofs),i] + self.gmm_model.sigma[i,0:self.input_dofs,self.input_dofs:(self.input_dofs+self.robot_dofs)]/self.gmm_model.sigma[i,0:self.input_dofs,0:self.input_dofs] * (data_in[:,t].reshape((-1,1)) - self.gmm_model.mu[0:self.input_dofs,i])
                exp_data[:,t] = exp_data[:,t] + H[i,t] * mu_tmp[:,i]
                # print("Mu_tmp: ",mu_tmp[:,i])
                # print(H[i,t] * mu_tmp[:,i])

            # Compute conditional covariance
            for i in range(self.gmm_model.num_states):
                sigma_tmp = self.gmm_model.sigma[i,self.input_dofs:(self.input_dofs+self.robot_dofs),self.input_dofs:(self.input_dofs+self.robot_dofs)] - self.gmm_model.sigma[i,self.input_dofs:(self.input_dofs+self.robot_dofs),0:self.input_dofs]/self.gmm_model.sigma[i,0:self.input_dofs,0:self.input_dofs] * self.gmm_model.sigma[i,0:self.input_dofs,self.input_dofs:(self.input_dofs+self.robot_dofs)]
                # print(sigma_tmp)
                exp_sigma[t,:,:] = exp_sigma[t,:,:] + H[i,t] * (sigma_tmp + mu_tmp[:,i]*mu_tmp[:,i].T)
                # print(exp_sigma[t,:,:])
            exp_sigma[t,:,:] = exp_sigma[t,:,:] - exp_data[:,t] * exp_data[:,t].T + np.eye(num_varout) * diag_reg_factor
        return exp_data, exp_sigma, H

    ##################################

    def findViaPointIndices(self):
        via_pt_inds = []
        for via_pt in self.via_pts:
            minim = float("Inf")
            min_ind = 0
            for i in range(len(self.ref_traj_mus)):
                if distBWPts(self.ref_traj_mus[i][0:2],via_pt)<minim:
                    minim = distBWPts(self.ref_traj_mus[i][0:2],via_pt)
                    min_ind = i
            print(self.ref_traj_mus[min_ind])
            via_pt_inds.append(min_ind)
        return via_pt_inds




    def prediction(self, via_pts, via_pt_var):
        print("Finding Indices for each via pt")
        # via_pt_inds = findViaPointIndices()

        ####### DEBUGGING ##############
        # new_ref_path = extractPath(self.new_ref_traj)
        # plt.plot(new_ref_path[:, 0], new_ref_path[:, 1])
        # plt.show()
        ################################
        pass


###############################################################################
# Functions
def gaussPDF(data, mu, sigma):
    num_vars, num_datapts = data.shape
    data = data.T - np.repeat(mu.reshape((1, mu.shape[0])), [num_datapts], axis=0)
    if num_vars==1 and num_datapts==1:
        prob = data**2 / sigma
        prob = np.e ** (-0.5 * prob) / (np.sqrt((2 * np.pi) ** num_vars * sigma))
    else:
        prob = np.sum(np.multiply(np.matmul(data, np.linalg.inv(sigma)), data), axis=1)
        prob = np.e ** (-0.5 * prob) / (np.sqrt((2 * np.pi) ** num_vars * abs(np.linalg.det(sigma))))
    # print(prob)
    return prob
#################

def distBWPts(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return abs(np.linalg.norm([x,y],2))
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

    # print(kernel_matrix)
    return kernel_matrix

def extractPath(traj):
    path = np.empty((len(traj),4))
    for i in range(len(traj)):
        path[i,:] = traj[i].mu.flatten()
    return path


###############################################################################

if __name__ == "__main__":
    data_addr = '../../training_data/KMP_div1000/'
    # ndemos = len(os.listdir(data_addr))
    ndemos = 7
    kmp = KMP(input_dofs=1, robot_dofs=4, demo_dt=0.1, ndemos=ndemos, data_address=data_addr)

    # Set desired via points
    # via_pts = [[350, 1000, -26, 37],[1400, 900, -23, -32]]
    via_pts = [[0.89,0.37], [0.3,1.1], [1.1,1.2], [1.06,0.37]]
    via_pt_var = 1e-6 * np.eye(kmp.ref_traj[0].sigma.shape[0])

    #KMP Prediction
    pred_traj = kmp.prediction(via_pts,via_pt_var)
    # pred_path = extractPath(pred_traj)

    # Plotting
    # ref_traj = kmp.ref_traj
    # ref_path = extractPath(ref_traj)
    # # plt.plot(ref_path[:, 0], ref_path[:, 1], 'r')
    # # plt.show()
    # #
    # plt.figure(1)
    # plt.plot(pred_path[:, 0], pred_path[:, 1], label="KMP Generated Trajectory")
    # plt.plot(ref_path[:, 0], ref_path[:, 1], 'g', label="Demonstrated Trajectory")
    # for via_pt in via_pts:
    #     plt.plot(via_pt[0], via_pt[1], 'bo', label="Via Points")
    #     plt.plot(via_pt[0], via_pt[1], 'bo')
    #     # plt.plot(via_pts[2][0], via_pts[2][1], 'bo')
    #     # plt.plot(via_pts[3][0], via_pts[3][1], 'bo')
    # # plt.legend(loc='lower left')
    # plt.title('KMP generalization over new via points')
    # plt.show()

