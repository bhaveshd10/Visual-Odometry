from get_sift import *
from essential_mat import *
from plot_trajectory import *
from descriptors import *
import argparse

'''
The following script calculates the essential matrix based on K
Breaking down the essential matrix to R and T to get the trajectory for VO
'''
parser = argparse.ArgumentParser()
parser.add_argument('folder_path', action="store")
parser.add_argument('K_path', action="store")
args = parser.parse_args()

K1 = np.loadtxt(args.K_path)
K1 = K1.reshape(3,3)
K2 = K1.copy()

getsift = get_sift(args.folder_path)
list_kp1,list_kp2,list_dp1 = getsift.getsift()

get_essential = get_essential(list_kp1,list_kp2,K1,K2)
R_est,T_est = get_essential.essential_mat()

R_inital = np.identity(3)
t_zero = np.zeros((3, 1))
path_ = []

for i in range(len(R_est)):
    rotation = np.dot(R_est[i],R_inital)
    translation = t_zero + np.dot(rotation,T_est[i])
    path_.append(translation)
    R_inital = rotation
    t_zero = translation

fig, ax = plt.subplots()

for i in range(len(path_)):
    ax.scatter(path_[i][0],path_[i][1],c='r')
    fig.savefig('C:/Users/bhave/PycharmProjects/SIR/out_traj/'+str(i)+'.png')
    plt.close(fig)


