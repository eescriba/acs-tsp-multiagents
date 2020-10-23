import numpy as np
import math
import random


class HMM(object):
    """
        :param S: state set
        :param V: observation set
        :param A: transition matrix
        :param B: emision matrix
        :param PI: initial matrix
        :param O: current observations
    """

    def __init__(self):
        self.O = []
        self.S = ['H', 'MH', 'M', 'ML', 'L']
        self.V = ['LL', 'LM' , 'LH', 'ML', 'MM' , 'MH', 'HL', 'HM' , 'HH']
      
        self.A = np.array([
                  [0.5, 0.2, 0.2, 0.2, 0.2], 
                  [0.2, 0.5, 0.2, 0.2, 0.2],
                  [0.2, 0.2, 0.5, 0.2, 0.2],
                  [0.2, 0.2, 0.5, 0.5, 0.2],
                  [0.2, 0.2, 0.2, 0.2, 0.5]])     
        self.B = np.array([
                  [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                  [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                  [0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.2, 0.2],
                  [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                  [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])

        self.PI = np.array([random.random()  for _ in self.S])


    def update_params(self, iteration, diversity, rho, phi):
        obs = self.get_observation(iteration, diversity)
        self.O.append(obs)
        T = len(self.O)
        if T==1:
            return rho, phi
        state = self.viterbi(T)
        return self.reestimate_rho_phi(state)


    def get_observation(self, iteration, diversity):
        it = "H"
        div = "H"
        if 0 < iteration <= 1/3:
            it = "L"
        if 1/3 < iteration <= 2/3:
            it = "M"
        if 0 < diversity <= 1/3:
            div = "L"
        if 1/3 < diversity <= 2/3:
            div = "M" 
        return it + div

    def reestimate_rho_phi(self,state):
        if state == "L":
            return 5/6, 1/6
        if state == "ML":
            return 4/6, 2/6
        if state == "M":
            return 3/6, 3/6
        if state == "MH":
            return 2/6, 4/6
        return 1/6, 5/6

    def obs_index(self, t: int):
        return self.V.index(self.O[t])
        
        
    def viterbi(self, T: int):
        self.baum_welch(T)

        (N, _) = self.B.shape
  
        #Initialization
        fi = np.zeros( (T, N) )
        alpha_mat = np.zeros( (T, N) )
        alpha_mat[0,:] = self.PI * self.B[:,self.obs_index(0)]

        # #Recursion
        for t in range(1, T):
            for j in range(N):
                alpha_mat[t,j] = np.max([alpha_mat[t-1,i] * self.A[i,j] * self.B[j,self.obs_index(t)] for i in range(N)])
                fi[t,j]= np.argmax([alpha_mat[t-1,i] * self.A[i,j] for i in range(N)])
   
        #Termination
        Z = np.zeros(T)
        Z[-1] = np.argmax(alpha_mat[T-1,:])
        #print("Z",Z)

        #Reconstruction
        for t in range(T-1, 0, -1):
            #print("Z",Z)
            Z[t-1] = fi[t, int(Z[t])]

        return self.S[int(Z[0])]


    def baum_welch(self, T: int):
        (N, M) = self.B.shape
        while True:
            old_A = np.copy(self.A)
            old_B = np.copy(self.B)
            alpha_mat= self._forward_baum_welch(T, N)
            beta_mat = self._backward_baum_welch(T, N)
            alpha_mat, beta_mat = self._normalize(alpha_mat, beta_mat)
            self._update_baum_welch(T, N, M, alpha_mat, beta_mat)
            
            # Convergence condition
            if np.linalg.norm(old_A-self.A) <0.0001 and np.linalg.norm(old_B-self.B) <0.0001:
                break

    def _forward_baum_welch(self, T: int, N: int):
    
        alpha_mat = np.zeros( (T, N) )

        for i in range(N):
            alpha_mat[0][i] = self.PI[i] * self.B[i][self.obs_index(0)]
        
        for t in range(T-1):
            for j in range(N):
                sumat = [alpha_mat[t][i] * self.A[i][j] for i in range(N)]
                alpha_mat[t+1,j] = self.B[j][self.obs_index(t+1)] * sum(sumat)
      
        return alpha_mat

        
    def _backward_baum_welch(self, T: int, N: int):
       
        beta_mat = np.zeros( (T, N) )
        
        beta_mat[-1,:] = 1
        for t in range(T-1, 0, -1):
            for i in range(N):
                beta_mat[t-1][i] = sum(beta_mat[t,:] * self.A[i,:] * self.B[:,self.obs_index(t)])
                
        return beta_mat

    def _normalize(self, alpha_mat, beta_mat):
        Z = np.sum(alpha_mat)
        if Z!=0:
            alpha_mat = alpha_mat/Z
            beta_mat = beta_mat/Z
        return alpha_mat, beta_mat

    def _update_baum_welch(self, T: int, N: int, M: int, alpha_mat, beta_mat):
        eth = np.zeros( (T, N, N) )
        gamma = alpha_mat*beta_mat
        sum_gamma = np.sum(gamma)
        if sum_gamma > 0:
            gamma  = gamma/sum_gamma

        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    eth[t][i][j] = alpha_mat[t][i] * self.A[i][j]* beta_mat[t+1][j]  * self.B[j][self.obs_index(t+1)]
      
        sum_eth = np.sum(eth)
        if sum_eth > 0:
            eth = eth/sum_eth
                
        for i in range(N):
            self.PI[i] = gamma[0][i]
            sum_gamma_i = np.sum(gamma[:,i])
            for j in range(N):
                if sum_gamma_i == 0:
                    sum_gamma_i = 1
                self.A[i][j] = np.sum(eth[:-1,i,j]) / sum_gamma_i
            
            for k in range(M):
                sumat = 0
                for t in range(T):
                    if k == self.obs_index(t):
                        sumat+=gamma[t][i]
                self.B[i][k]= sumat / sum_gamma_i