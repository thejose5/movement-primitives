import numpy as np
import os
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class GMM_Component:
    def __init__(self, pi, mus, sigmas):
        self.pi = pi
        self.mus = mus
        self.sigmas = sigmas
        self.dur_mean = None
        self.dur_var = None

class TP_HSMM:
    def __init__(self, data_addr, num_viapts, ndemos, dt=0.1):
                    self.dt = dt
                    self.num_viapts = num_viapts
                    self.num_gmm_components = 10
                    self.num_demos = ndemos

                    raw_data = self.loadData(data_addr, ndemos)
                    self.data, self.queries = self.processData(raw_data)
                    self.combined_data = self.combineData(self.data)

                    self.num_frames = num_viapts + 2 # via pts, start pt, end pt
                    self.demos_wrt_frames = [] # combined_data wrt each frame. shape: combined_data.shape[0] X num_frames
                    # The frames here aren't specific coordinates. They are just locations with respect to a demonstration,
                    # e.g. via pts, start point, end point. For each frame index, we iterate over each demonstration, find the
                    # coordinates of the frame for that demo and transform that demo wrt that frame
                    for f in range(self.num_frames):
                        demos_wrt_f = []
                        for n,d in enumerate(self.data):
                            frames = np.split(self.queries[n],self.num_viapts)
                            frames.append(d[0,:])
                            frames.append(d[-1,:])
                            demo_wrt_f = demoWrtFrame(d,frames[f])
                            demos_wrt_f.append(demo_wrt_f)
                        self.demos_wrt_frames.append(self.combineData(demos_wrt_f))

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

                    self.encodeDuration_and_stateTransition()
                    # self.global_gmm.predict([[500,900],[1300,1100]])

                    # plt.scatter(self.demos_wrt_frames[0][:, 0], self.demos_wrt_frames[0][:, 1], s=0.8, color='red')
                    # plt.scatter(self.demos_wrt_frames[1][:, 0], self.demos_wrt_frames[1][:, 1], s=0.8, color='blue')
                    # plt.scatter(self.demos_wrt_frames[2][:, 0], self.demos_wrt_frames[2][:, 1], s=0.8, color='green')
                    # plt.scatter(self.demos_wrt_frames[3][:, 0], self.demos_wrt_frames[3][:, 1], s=0.8, color='violet')
                    # plt.show()
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
    def encodeDuration_and_stateTransition(self):
        prev_state = -1
        transition_matrix = np.zeros((self.num_gmm_components,self.num_gmm_components))
        duration_matrix = np.zeros((self.num_gmm_components, self.num_demos))
        for demo_num,demo in enumerate(self.data):
            for point in demo:
                state = self.global_gmm.predict([point])
                if prev_state == -1:
                    prev_state = state
                    continue
                #TODO: Do something about cases where the same state is visited multiple times
                if state==prev_state:
                    # Code for duration
                    duration_matrix[state, demo_num] += self.dt
                else:
                    # Code for state transition
                    transition_matrix[prev_state, state] += 1
                    # transition_matrix[prev_state, state] = transition_matrix[prev_state, state] * sum(transition_matrix[prev_state,:].flatten()) + 1
                    # transition_matrix[prev_state, state] = transition_matrix[prev_state, state]/sum(transition_matrix[prev_state, :].flatten())
                    prev_state = state
        transition_matrix = transition_matrix/np.sum(transition_matrix,axis=1)[:,None]
        self.state_transition_mat = transition_matrix
        self.duration_mean = np.mean(duration_matrix,axis=1)
        self.duration_var = np.var(duration_matrix,axis=1)

    # def predict(self, via_pts):




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

##################################################
if __name__ == "__main__":
    data_addr = '../../training_data/TP-HSMM/'
    ndemos = len(os.listdir(data_addr))
    # ndemos = 10
    tp_hsmm = TP_HSMM(data_addr=data_addr,num_viapts=2,ndemos=ndemos,dt=0.1)
    print('Khatam')