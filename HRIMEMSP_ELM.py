import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from mafese import get_dataset
from intelelm import MhaElmClassifier
from copy import deepcopy
import warnings


warnings.filterwarnings("ignore")


PopSize = 20
DimSize = 10
LB = [-10] * DimSize
UB = [10] * DimSize
TrialRuns = 30
MaxFEs = 1000
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

def fit_func(indi):
    global X_train, y_train, model
    return model.fitness_function(indi)


def score_func(indi):
    global Xtest, ytest, model
    return model.score(indi)


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, curFEs, DimSize, BestIndi, BestFit
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = -fit_func(Pop[i])
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
            FitOff[idx] = -fit_func(Off[idx])
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
            FitOff[idx] = -fit_func(Off[idx])
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
            FitOff[idx] = -fit_func(Off[idx])
            curFEs += 1
            FitPop[idx] = deepcopy(FitOff[idx])
            Pop[idx] = deepcopy(Off[idx])
            if FitOff[idx] < BestFit:
                BestFit = deepcopy(FitOff[idx])
                BestIndi = deepcopy(Off[idx])


def RunHRIMEMSP(setname):
    global curIter, TrialRuns, Pop, FitPop, DimSize, X_train, y_train, Xtest, ytest, model, DimSize, LB, UB, BestIndi
    dataset = get_dataset(setname)
    X = dataset.X
    y = dataset.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)

    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    le_y = LabelEncoder()
    le_y.fit(y)
    y_train = le_y.transform(y_train)
    y_test = le_y.transform(y_test)

    All_Trial_Best = []
    All_score = []
    for i in range(TrialRuns):
        BestIndi = None

        model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="AS")
        model.network, model.obj_scaler = model.create_network(X_train, y_train)
        y_scaled = model.obj_scaler.transform(y_train)
        model.X_temp, model.y_temp = X_train, y_scaled

        DimSize = len(X_train[0]) * 10 + 10
        Pop = np.zeros((PopSize, DimSize))
        LB = [-10] * DimSize
        UB = [10] * DimSize

        Best_list = []
        curIter = 0
        Initialization()
        Best_list.append(min(FitPop))
        np.random.seed(2022 + 88 * i)
        while curIter <= MaxIter:
            HRIMEMSP()
            curIter += 1
            Best_list.append(min(FitPop))
            # print("Iter: ", curIter, "Best: ", Fgbest)
        model.network.update_weights_from_solution(BestIndi, model.X_temp, model.y_temp)
        All_score.append(model.score(X_test, y_test))
        All_Trial_Best.append(np.abs(Best_list))
    np.savetxt("./HRIMEMSP_Data/ELM/" + str(FuncNum) + ".csv", All_score, delimiter=",")


def main(setname):
    global FuncNum
    FuncNum = setname
    RunHRIMEMSP(setname)


if __name__ == "__main__":
    if os.path.exists('./HRIMEMSP_Data/ELM') == False:
        os.makedirs('./HRIMEMSP_Data/ELM')
    Datasets = ['aggregation', 'aniso', 'appendicitis', 'balance', 'banknote', 'blobs', 'Blood', 'BreastCancer',
                'BreastEW', 'circles', 'CongressEW', 'diagnosis_II', 'Digits', 'ecoli', 'flame', 'Glass', 'heart',
                'HeartEW', 'Horse', 'Ionosphere', 'Iris', 'jain', 'liver', 'Madelon', 'Monk1', 'Monk2', 'Monk3',
                'moons', 'mouse', 'pathbased', 'seeds', 'smiley', 'Sonar', 'Tic-tac-toe', 'varied',
                'vary-density', 'vertebral2', 'Vowel', 'WaveformEW', 'wdbc', 'Wine']
    for setname in Datasets:
        main(setname)
