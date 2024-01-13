import os
from copy import deepcopy
import numpy as np
from opfunu.cec_based.cec2022 import *


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


# initialize the Pop randomly
def Initialization(func):
    global Pop, FitPop, curFEs, DimSize, BestIndi, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
        curFEs += 1
    BestFit = min(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])



def HRIMEMSP(func):
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
            FitOff[idx] = func(Off[idx])
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
            FitOff[idx] = func(Off[idx])
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
            FitOff[idx] = func(Off[idx])
            curFEs += 1
            FitPop[idx] = deepcopy(FitOff[idx])
            Pop[idx] = deepcopy(Off[idx])
            if FitOff[idx] < BestFit:
                BestFit = deepcopy(FitOff[idx])
                BestIndi = deepcopy(Off[idx])


def RunHRIMEMSP(func):
    global curFEs, curIter, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        curIter = 0
        Initialization(func)
        Best_list.append(BestFit)
        np.random.seed(2022 + 88 * i)
        while curFEs < MaxFEs:
            HRIMEMSP(func)
            curIter += 1
            Best_list.append(BestFit)
        All_Trial_Best.append(Best_list)
    np.savetxt("./HRIMEMSP_Data/CEC2022/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, Pop, MaxFEs, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    LB = [-100] * dim
    UB = [100] * dim
    FuncNum = 1
    
    CEC2022 = [F12022(Dim), F22022(Dim), F32022(Dim), F42022(Dim), F52022(Dim), F62022(Dim),
               F72022(Dim), F82022(Dim), F92022(Dim), F102022(Dim), F112022(Dim), F122022(Dim)]
    
    for i in range(len(CEC2022)):
        FuncNum = i+1
        RunHRIMEMSP(CEC2022[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./HRIMEMSP_Data/CEC2022') == False:
        os.makedirs('./HRIMEMSP_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
