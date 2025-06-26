from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def get_data(name = "iris", ev_size = 0.2,max_angle = np.pi / 4, zzmapping = False):
    
    if name == 'pima':
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome"
        ]
        df = pd.read_csv(url, header=None, names=column_names)

        # 3. 데이터 분할 및 NumPy 배열 변환
        X = df.drop("Outcome", axis=1).values
        y = df["Outcome"].values
        y = np.where(y == 0, -1, y)

        scaler = MinMaxScaler(feature_range=(0, max_angle))
    
        # 훈련 데이터에 스케일러 학습 및 변환
        X_train_scaled = scaler.fit_transform(X)
        if ev_size == 0.:
            return X_scaled, y
        
        else : 
            X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, test_size=ev_size)
            return X_train, X_test, y_train, y_test


    elif name=='iris':
        iris = load_iris()
        X = iris.data       # 특징 데이터
        y = iris.target     # 원래 레이블 (Setosa=0, Versicolor=1, Virginica=2)

        # Setosa와 Versicolor만 선택 (레이블 0과 1)
        mask = y < 2
        X_binary = X[mask]
        y_binary = y[mask]

        # 레이블 변환: Setosa(0) -> 1, Versicolor(1) -> -1
        y_binary = np.where(y_binary == 0, 1, -1)

        if zzmapping == True:
            scaler = MinMaxScaler(feature_range=(0,np.pi))
            X_scaled = scaler.fit_transform(X_binary)
        else: 
            scaler = MinMaxScaler(feature_range=(0,2 * np.pi))
            X_scaled = scaler.fit_transform(X_binary)

        if ev_size == 0.:
            return X_scaled, y_binary
        
        else : 
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=ev_size)
            return X_train, X_test, y_train, y_test


    else:
        print("Choose \'pima\' or \'iris\'")
        return None
    
