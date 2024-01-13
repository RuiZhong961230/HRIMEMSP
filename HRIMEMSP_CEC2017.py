import os
from copy import deepcopy
import numpy as np
from cec17_functions import cec17_test_func


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = DimSize * 1000
curFEs = 0

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
Off = np.zeros((PopSize, DimSize))

FitPop = np.zeros(PopSize)
FitOff = np.zeros(PopSize)

FuncNum = 0

BestIndi = None
BestFit = float("inf")


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, curFEs, DimSize, BestIndi, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])



def HRIMEMSP():
    global Pop, FitPop, Off, FitOff, curIter, MaxIter, LB, UB, PopSize, DimSize, curFEs, BestIndi, BestFit
    w = 5
    Off = deepcopy(Pop)
    factor = np.random.uniform(-1, 1) * np.cos(np.pi * (curIter + 1) / (MaxIter / 10)) * (
            1 - np.round((curIter + 1) * w / MaxIter) / w)
    E = np.sqrt((curIter + 1) / MaxIter)
    NorFit = FitPop / np.linalg.norm(FitPop, axis=0, keepdims=True)

    sort_idx = np.argsort(FitPop)

    for i in range(PopSize):
        idx = sort_idx[i]
        if i < 20:
            Off[idx] = Pop[idx] + np.random.uniform(-1, 1, DimSize)
            Off[idx] = np.clip(Off[idx], LB, UB)
            FitOff[idx] = fitness(Off[idx])
            curFEs += 1
            if FitOff[idx] < FitPop[idx]:
                FitPop[idx] = deepcopy(FitOff[idx])
                Pop[idx] = deepcopy(Off[idx])
                if FitOff[idx] < BestFit:
                    BestFit = deepcopy(FitOff[idx])
                    BestIndi = deepcopy(Off[idx])

        elif i < 80:
            for j in range(DimSize):
                if np.random.rand() < E:  # Soft rime
                    Off[idx][j] = BestIndi[j] + factor * (np.random.rand() * (UB[j] - LB[j]) + LB[j])
                if np.random.rand() < NorFit[idx]:  # Hard rime
                    Off[idx][j] = BestIndi[j]
            Off[idx] = np.clip(Off[idx], LB, UB)
            FitOff[idx] = fitness(Off[idx])
            curFEs += 1
            if FitOff[idx] < FitPop[idx]:
                FitPop[idx] = deepcopy(FitOff[idx])
                Pop[idx] = deepcopy(Off[idx])
                if FitOff[idx] < BestFit:
                    BestFit = deepcopy(FitOff[idx])
                    BestIndi = deepcopy(Off[idx])

        else:
            Off[idx] = (np.array(UB) + np.array(LB)) - np.random.rand(DimSize) * Pop[idx]
            Off[idx] = np.clip(Off[idx], LB, UB)
            FitOff[idx] = fitness(Off[idx])
            curFEs += 1
            FitPop[idx] = deepcopy(FitOff[idx])
            Pop[idx] = deepcopy(Off[idx])
            if FitOff[idx] < BestFit:
                BestFit = deepcopy(FitOff[idx])
                BestIndi = deepcopy(Off[idx])


def RunHRIMEMSP():
    global curFEs, curIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        Initialization()
        Best_list.append(BestFit)
        np.random.seed(2022 + 88 * i)
        while curFEs < MaxFEs:
            HRIMEMSP()
            curIter += 1
            Best_list.append(BestFit)
        All_Trial_Best.append(Best_list)
    np.savetxt("./HRIMEMSP_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    LB = [-100] * dim
    UB = [100] * dim
    FuncNum = 1
    for i in range(1, 31):
        FuncNum = i
        if i == 2:
            continue
        RunHRIMEMSP()


if __name__ == "__main__":
    if os.path.exists('./HRIMEMSP_Data/CEC2017') == False:
        os.makedirs('./HRIMEMSP_Data/CEC2017')
    Dims = [30, 50]
    for Dim in Dims:
        main(Dim)
