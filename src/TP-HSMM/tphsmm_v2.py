import numpy as np
import os
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import copy
import random
import test_sklearn_gmm

class GMM_Component:
    def __init__(self, pi, mus, sigmas):
        self.pi = pi
        self.mus = mus
        self.sigmas = sigmas
        self.dur_mean = None
        self.dur_var = None

class TP_HSMM:
    def __init__(self, data_addr, num_frames, ndemos, dt=0.1):
                    self.dt = dt
                    self.num_frames = num_frames
                    self.num_gmm_components = 4
                    self.num_demos = ndemos

                    raw_data = self.loadData(data_addr, ndemos)
                    self.data, self.queries = self.processData(raw_data)
                    self.combined_data = self.combineData(self.data)

                    self.demos_wrt_frames = [] # combined_data wrt each frame. shape: combined_data.shape[0] X num_frames
                    # The frames here aren't specific coordinates. They are just locations with respect to a demonstration,
                    # e.g. via pts, start point, end point. For each frame index, we iterate over each demonstration, find the
                    # coordinates of the frame for that demo and transform that demo wrt that frame
                    firstpts_wrt_frames = []
                    for f in range(self.num_frames):
                        demos_wrt_f = []
                        firstpts_wrt_frames.append([])
                        for n,d in enumerate(self.data):
                            frames = np.split(self.queries[n],self.num_frames)
                            demo_wrt_f = demoWrtFrame(d,frames[f])
                            firstpts_wrt_frames[f].append(demo_wrt_f[0,:]) #Stores the first point of each demo wrt each frame. Used to find the most likely starting state
                            demos_wrt_f.append(demo_wrt_f)
                        self.demos_wrt_frames.append(self.combineData(demos_wrt_f))

                        # plt.scatter(self.combined_data[:,0], self.combined_data[:,1], s=3)
                        # plt.show()
                        # test_sklearn_gmm.plot_gmms(self.demos_wrt_frames[-1], n_comp=self.num_gmm_components)

                    self.global_gmm = GaussianMixture(n_components=self.num_gmm_components, random_state=0)
                    self.global_gmm.fit(self.combined_data)

                    # Now for the combined demos wrt each frame, we fit a GMM. The pi vectors of all the GMMs will be same.
                    # So we get the data in the form (pi(k), [mu(p),sigma(p) where p is every frame]) where k is every component
                    self.gmms = []
                    for demo_wrt_frame in self.demos_wrt_frames:
                        gmm = GaussianMixture(n_components=self.num_gmm_components,random_state=0)
                        gmm.fit(demo_wrt_frame)
                        self.gmms.append(gmm)
                    # Now convert this to the required format
                    self.gmm_components = []
                    for g in range(self.gmms[0].n_components):
                        pi = [self.gmms[i].weights_[g] for i in range(self.num_frames)]
                        mus = [self.gmms[i].means_[g,:] for i in range(self.num_frames)]
                        sigmas = [self.gmms[i].covariances_[g,:,:] for i in range(self.num_frames)]
                        self.gmm_components.append(GMM_Component(pi,mus,sigmas))

                    self.findInitialStates(firstpts_wrt_frames)
                    self.encodeStateTransitionWRT_Frames()


                    ######################## TEST #########################
                    # frame1 = np.array([900,380])
                    # demo1_wrt_frame1 = demoWrtFrame(self.combined_data[:,1:],frame1)
                    # self.gmm1 = GaussianMixture(n_components=self.num_gmm_components,random_state=0)
                    # self.gmm1.fit(demo1_wrt_frame1)
                    # self.gmm = GaussianMixture(n_components=self.num_gmm_components, random_state=0)
                    # self.gmm.fit(self.combined_data[:, 1:])
                    #######################################################
    ######################## DATA PROCESSING ###############################
    def loadData(self, addr, ndemos):
        data = []
        for i in range(ndemos):
            data.append(np.loadtxt(open(addr + str(i + 1) + ""), delimiter=","))
        # data is saved as a list of nparrays [nparray(Demo1),nparray(Demo2),...]
        return data

    def processData(self, raw_data):
        data = []
        queries = []
        for i in range(len(raw_data)):
            data.append(raw_data[i][:,[0,1]])
            queries.append(raw_data[i][0,2:])
            # Now add time component to data
            # time = np.arange(raw_data[i].shape[0]) * self.dt
            # time = time.reshape((-1,1))
            # data[i] = np.hstack((time,data[i]))
        return data, queries

    def combineData(self, data):
        combined_data = np.array(data[0])
        for i in range(1,len(data)):
            combined_data = np.append(combined_data, data[i], axis=0)
        return combined_data
    ######################## /DATA PROCESSING ###############################
    def findInitialStates(self, firstpts_wrt_frames): #firstpts_wrt_frames is a list of shape (num_frames x num_demos)
        self.frame_initial_state = np.empty((self.num_frames,self.num_gmm_components))
        for f in range(self.num_frames):
            for pt in firstpts_wrt_frames[f]:
                state = self.gmms[f].predict([pt])
                self.frame_initial_state[f,state] += 1
        self.frame_initial_state = np.argmax(self.frame_initial_state,axis=1)

    def encodeStateTransitionWRT_Frames(self):
        self.frame_state_transition = []
        for f in range(self.num_frames):
            prev_state = -1
            transition_matrix = np.zeros((self.num_gmm_components, self.num_gmm_components))
            for point in self.demos_wrt_frames[f]:
                state = self.gmms[f].predict([point])
                # print(state)
                if prev_state == -1:
                    prev_state = state
                    continue
                if state != prev_state:
                    # Code for state transition
                    transition_matrix[prev_state, state] += 1
                    prev_state = state
            transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:, None]
            self.frame_state_transition.append(transition_matrix)
        self.frame_state_transition_static = copy.deepcopy(self.frame_state_transition)

    def predict(self, via_pts):
        state_transition_sequence = []
        frames = np.split(np.array(via_pts),self.num_frames)
        print(frames)
        for f in range(self.num_frames):
            seq = [self.frame_initial_state[f]]
            pts = [self.gmms[f].means_[seq[-1],:]]
            for state_num in range(1, self.num_gmm_components):
                next_state = np.argmax(self.frame_state_transition[f][seq[-1],:])
                seq.append(next_state)
                pts.append(self.gmms[f].means_[seq[-1],:])

            state_transition_sequence.append(seq)
        pred_traj_frame_wrt_global = []
        pred_traj_frame_wrt_frames = []
        for f in range(self.num_frames):
            traj_wrt_global = np.empty((self.num_gmm_components,2)) #TODO: Change 2 to self.robot_dofs
            traj_wrt_frames = np.empty((self.num_gmm_components, 2))
            for state_num, state in enumerate(state_transition_sequence[f]):
                traj_wrt_frames[state_num,:] = np.array(self.gmms[f].means_[state,:])
                traj_wrt_global[state_num,:] = demoWrtFrame(traj_wrt_frames[state_num,:].reshape((1,-1)),-1*frames[f])
            pred_traj_frame_wrt_frames.append(traj_wrt_frames)
            pred_traj_frame_wrt_global.append(traj_wrt_global)

        ################# Plotting resulting trajectory #########################
        # print(pred_traj_frame_wrt_global)
        # print(pred_traj_frame_wrt_frames)
        # print(state_transition_sequence)
        # print(self.frame_state_transition_static)
        # colors = ['red','blue','green','black']
        # for n,traj_wrt_global in enumerate(pred_traj_frame_wrt_global):
        #     plt.plot(traj_wrt_global[:, 0], traj_wrt_global[:, 1], color=colors[n])
        # plt.show()
        ################# /Plotting resulting trajectory #########################

        # Now that we have the trajectories according to each frame, we have to find out which of the trajectories
        # should be followed for each point in the trajectory. To find the importance of each point, we find its probability
        # of being in its own frames gmm. For each point in the actual trajectory, the point from all the frame trajs
        # with the highest probability of being in its frame's gmm is selected as the most important.
        final_traj = np.empty((self.num_gmm_components,2))
        for pt in range(self.num_gmm_components):
            pt_probabs = []
            for f in range(self.num_frames):
                frame_pt = pred_traj_frame_wrt_frames[f][pt,:]
                frame_pt_prob = calcProbPtInGMM(frame_pt, self.gmms[f], state_transition_sequence[f][pt])
                pt_probabs.append(frame_pt_prob)
            most_imp_pt = pred_traj_frame_wrt_global[np.argmax(pt_probabs)][pt,:]
            final_traj[pt,:] = most_imp_pt
        # plt.plot(final_traj[:, 0], final_traj[:, 1])
        # for pt in frames:
        #     pt = pt.flatten()
        #     plt.plot(pt[0],pt[1],'bo')
        # plt.show()
        return final_traj

##########################  /CLASS TP-HSMM  #########################################
def demoWrtFrame(data, frame_loc, rotation=np.array([[1,0,0],[0,1,0],[0,0,1]])):
    frame_loc = np.append(frame_loc, 0)
    ht_matrix = np.hstack((rotation,-1*frame_loc.reshape((3,1))))
    ht_matrix = np.vstack((ht_matrix,[0,0,0,1]))
    data_wrt_frame = np.empty((data.shape))
    for i in range(data.shape[0]):
        point = data[i,:]
        point = (np.append(point,[0,1])).reshape((-1,1))
        point_wrt_frame = np.matmul(ht_matrix,point)
        data_wrt_frame[i,:] = point_wrt_frame[[0,1],:].flatten()
    return data_wrt_frame

def plotTraj(final_traj, pts):
    plt.plot(final_traj[:, 0], final_traj[:, 1])
    for pt in pts:
        plt.plot(pt[0], pt[1], 'bo')
    plt.show()

def calcProbPtInGMM(pt, gmm, state):
    # return np.sum(gmm.predict_proba(pt.reshape(1,-1)) * gmm.weights_)
    return gmm.predict_proba(pt.reshape(1,-1)).flatten()[state]
##################################################
if __name__ == "__main__":
    data_addr = '../../training_data/TP-HSMM_2frame/'
    # ndemos = len(os.listdir(data_addr))
    ndemos = 10
    tp_hsmm = TP_HSMM(data_addr=data_addr,num_frames=2,ndemos=ndemos,dt=0.1)
    # data: tp_hsmm (9 demos), comp: 5, query: [[500,1100],[1500,1000]]
    query = [[random.randint(300,700),random.randint(875,1200)],[random.randint(1300,1700),random.randint(875,1200)]]
    # query = [[303, 966],[1543,  974]]
    # query = [[500, 1100], [1500, 1000]]
    final_traj = tp_hsmm.predict(query)
    plotTraj(final_traj, query)
