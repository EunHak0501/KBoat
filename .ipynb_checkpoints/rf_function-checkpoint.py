from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek
import pandas as pd
import numpy as np

def cal_rf_params(target_value='복승', seed=42):
    params_list = []
    if target_value == '복승':
        params_list = []
        params_1 = {
            'random_state': seed,
            'class_weight': {0: 1, 1: 2.0},
            'max_features': 30,
            'n_jobs':4,
        }
        params_list.append(params_1)
        params_2 = {
            'random_state': seed,
            'class_weight': {0: 1, 1: 2.5},
            'max_features': 30,
            'n_jobs':4,
        }
        params_list.append(params_2)
        params_3 = {
            'random_state': seed,
            'class_weight': {0: 1, 1: 3.0},
            'max_features': 30,
            'n_jobs':4,
        }
        params_list.append(params_3)
        params_4 = {
            'random_state': seed,
            'class_weight': {0: 1, 1: 3.5},
            'max_features': 30,
            'n_jobs':4,
        }
        params_list.append(params_4)
        params_5 = {
            'random_state': seed,
            'class_weight': {0: 1, 1: 4.0},
            'max_features': 30,
            'n_jobs':4,
        }
        params_list.append(params_5)
        params_6 = {
            'random_state': seed,
            'class_weight': {0: 1, 1: 4.5},
            'max_features': 30,
            'n_jobs':4,
        }
        params_list.append(params_6)

    return params_list

class custom_randomforest():
    def __init__(self, params_list, smote_strategies=None, ctgan_list=None, target_value='복승', seed=42):
        self.models = [RandomForestClassifier(**params_list[i]) for i in range(6)]
        self.feature_names = None
        self.X = None
        self.seed = seed
        self.smote_strategies = smote_strategies
        self.ctgan_list = ctgan_list
        self.target_value = target_value

    def fit(self, X, y):
        # feature_names를 저장 (X가 pandas DataFrame인 경우)
        if isinstance(X, pd.DataFrame):
            self.X = X
            self.feature_names = X.columns
        else:
            raise ValueError("X must be a pandas DataFrame with column names.")

        for i in range(6):
            y_i = y.iloc[:, i] # i번째 레이블에 대한 y 값

            if self.ctgan_list is None and self.smote_strategies is None:
                X_train = X.copy()
                y_train = y_i.copy()
            elif self.ctgan_list is not None: # CTGAN
                new_data = self.ctgan_list[i].sample(3000)

                xy_train = pd.concat([X, y_i], axis=1)
                xy_train = pd.concat([xy_train, new_data], axis=0).reset_index(drop=True)

                target_col = f'{self.target_value}_{i+1}번선수'
                X_train = xy_train.drop(columns=[target_col])
                y_train = xy_train[target_col]
            elif self.smote_strategies is not None:
                smote_tomek = SMOTETomek(sampling_strategy=self.smote_strategies[i], random_state=self.seed)
                X_train, y_train = smote_tomek.fit_resample(X, y_i)

            self.models[i].fit(X_train, y_train)

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.vstack(preds).T

    def predict_proba(self, X):
        probas = [model.predict_proba(X)[:, 1] for model in self.models]
        return np.vstack(probas).T

    # 변수 중요도를 저장하고 출력하는 함수
    def get_feature_importance(self):
        # 각 모델에서 변수 중요도를 추출하고 평균 계산
        importances = np.mean([model.feature_importances_ for model in self.models], axis=0)

        # 변수 이름과 중요도를 데이터프레임으로 결합
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })

        return importance_df

    # 변수 중요도를 출력하는 함수 (선수별로 나눈 변수를 평균내서 계산)
    def print_feature_importance(self):
        # 전체 중요도를 가져오기
        importance_df = self.get_feature_importance()

        # 선수 번호를 제거한 변수 이름의 앞부분을 기준으로 그룹화하여 평균 계산
        importance_df['Group'] = importance_df['Feature'].str[:-5]  # 마지막 5자('_1번선수') 제거
        averaged_importances = importance_df.groupby('Group')['Importance'].mean().reset_index()

        return averaged_importances



def evaluate_(y_pred, y_val, target_value='단승', contain_연승=False):
    y_pred_max = np.zeros_like(y_pred)

    # 각 샘플에 대해 가장 높은 확률의 인덱스를 선택
    for i in range(y_pred.shape[0]):  # 각 샘플에 대해 반복
        if target_value == '단승':
            max_indices = np.argsort(y_pred[i])[-1:]  # 가장 큰 값의 인덱스 찾기
        elif target_value == '복승':
            max_indices = np.argsort(y_pred[i])[-2:]
        elif target_value == '삼복승':
            max_indices = np.argsort(y_pred[i])[-3:]
        y_pred_max[i, max_indices] = 1  # 해당 인덱스에 1 설정

    accuracy = accuracy_score(y_pred_max, y_val)
    if contain_연승 == False:
        return accuracy

    if contain_연승:
        y_pred_max_for_연승 = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            max_indices = np.argsort(y_pred[i])[-1:] # 가장 큰 값의 인덱스
            y_pred_max_for_연승[i, max_indices] = 1

        hits = 0
        for i in range(y_pred.shape[0]):
            # 예측된 최대 확률의 인덱스 중 하나라도 y_val에서 1인 인덱스와 겹치면 적중으로 간주
            pred_indices = np.where(y_pred_max_for_연승[i] == 1)[0]  # 예측에서 1인 인덱스
            true_indices = np.where(y_val[i] == 1)[0]  # 실제 값에서 1인 인덱스
            if len(np.intersect1d(pred_indices, true_indices)) > 0:  # 교집합이 있으면 적중
                hits += 1
        연승_score = hits / y_pred.shape[0]  # 적중한 비율 계산
        return accuracy, 연승_score