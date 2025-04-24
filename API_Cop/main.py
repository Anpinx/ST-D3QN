import argparse
import random
import sim
import time
import pickle
import math
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir_ST-D3QN', type=str, default='../data/ST-D3QN/path_xy.pkl')
parser.add_argument('--ckpt_dir_D3QN', type=str, default='../data/D3QN/path_xy.pkl')
parser.add_argument('--ckpt_dir_DQN', type=str, default='../data/DQN/path_xy.pkl')
args = parser.parse_args()

#simRemoteApi.start(19999)

def Coppelia_cordinition(sizeRate,path,coppelia_init,Coppelia_uav_cordinate):
    bpath = []
    Coppelia_path=[]
    before_pos = Coppelia_uav_cordinate
    for i in range(len(path[0])):
        x, y = [path[0][i], path[1][i]]
        new_x = coppelia_init - x / sizeRate
        new_y = coppelia_init - y / sizeRate
        d = math.sqrt((before_pos[0] - new_x) ** 2 + (before_pos[1] - new_y) ** 2)
        step_N = round(d / 0.05)
        # new_x=round(new_x,5)
        # new_y=round(new_y,5)
        bpath.append([new_x, new_y, step_N])
        before_pos = [new_x, new_y]

    A = [bpath[0][0], bpath[0][1]]
    Coppelia_path.append([A[0], A[1]])
    for i in range(1, len(bpath)):
        x, y = A
        for j in range(bpath[i][2]):
            t=j+1
            Dx = abs(bpath[i][0] - x)
            Dy = abs(bpath[i][1] - y)
            dx = x - (Dx / bpath[i][2])*t
            dy = y - (Dy / bpath[i][2])*t
            Coppelia_path.append([dx, dy])
        A = [bpath[i][0], bpath[i][1]]
    print(bpath)
    print(Coppelia_path)

    return Coppelia_path

step=0.05

with open(args.ckpt_dir_D3QNT, 'rb') as f:
    path_D3QNT = path_calculating(pickle.load(f))

sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
if clientID != -1:
    print("Connected to remote API server!")
else:
    print("Failed connecting to remote API server")
sim.simxGetPingTime(clientID)

ret, targetObj1 = sim.simxGetObjectHandle(clientID, 'Quadricopter_target', sim.simx_opmode_blocking)
_,positon= sim.simxGetObjectPosition(clientID, targetObj1, -1, sim.simx_opmode_blocking)
Coppelia_uav_cordinate=[positon[0],positon[1]]
Coppelia_path=Coppelia_cordinition(10.0/10.0,path_D3QNT,5.0,Coppelia_uav_cordinate)

count=1
index=0

for i in range(1,len(Coppelia_path)):
    ret, targetObj1 = sim.simxGetObjectHandle(clientID, 'Quadricopter_target', sim.simx_opmode_blocking)
    ret, arr = sim.simxGetObjectPosition(clientID, targetObj1, -1, sim.simx_opmode_blocking)
    # if ret == sim.simx_return_ok:
    #     print("UAV:1")
    #     print(arr)
    #     print([Coppelia_path[i][0],Coppelia_path[i][1]])
    #     print(i)
    arr[0]=Coppelia_path[i][0]
    arr[1]=Coppelia_path[i][1]
    print("UAV：{}下一步位置坐标：{}".format(i,arr))
    sim.simxSetObjectPosition(clientID, targetObj1, -1, (arr[0], arr[1], arr[2]), sim.simx_opmode_blocking)
    time.sleep(0.1)

print("安全到达目标区域")

sim.simxFinish(clientID)
