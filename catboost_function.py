from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def cal_params(target_value='복승', is_smote=False, seed=42):
    # if target_value == '복승':
    params_list = []
    params_1 = {
        'random_seed': seed,
        'iterations': 100,
        'learning_rate': 0.04911925044181611,
        'l2_leaf_reg': 5.877994583956269,
        'scale_pos_weight': 2.5884228912972223,
        'verbose': 0,
    }
    params_list.append(params_1)
    params_2 = {
        'random_seed': seed,
        'iterations': 100,
        'learning_rate': 0.025440461597786845,
        'l2_leaf_reg': 3.4564413079908647,
        'scale_pos_weight': 2.939776818158681,
        'verbose': 0,
    }
    params_list.append(params_2)
    params_3 = {
        'random_seed': seed,
        'iterations': 100,
        'learning_rate': 0.03436874835604066,
        'l2_leaf_reg': 9.448472248389246,
        'scale_pos_weight': 2.9798708469113033,
        'verbose': 0,
    }
    params_list.append(params_3)
    params_4 = {
        'random_seed': seed,
        'iterations': 100,
        'learning_rate': 0.09313363959534163,
        'l2_leaf_reg': 7.714619082072882,
        'scale_pos_weight': 3.0470667425661673,
        'verbose': 0,
    }
    params_list.append(params_4)
    params_5 = {
        'random_seed': seed,
        'iterations': 100,
        'learning_rate': 0.075751665806952,
        'l2_leaf_reg': 7.108368767089462,
        'scale_pos_weight': 3.4592120013671392,
        'verbose': 0,
    }
    params_list.append(params_5)
    params_6 = {
        'random_seed': seed,
        'iterations': 100,
        'learning_rate': 0.13452772939902888,
        'l2_leaf_reg': 9.466009502178139,
        'scale_pos_weight': 3.3783480597904103,
        'verbose': 0,
    }
    params_list.append(params_6)
    if is_smote:
        smote_list = []

        return params_list, smote_list

    return params_list

class custom_CatBoostClassifier():
    def __init__(self, params_list=None, smote_strategies=None, ctgan_list=None, target_value='복승', seed=42):
        if params_list is None:
            self.models = [CatBoostClassifier(verbose=0) for i in range(6)]
        else:
            self.models = [CatBoostClassifier(**params_list[i]) for i in range(6)]
        self.feature_names = None
        self.X = None
        self.seed = seed
        self.smote_strategies = smote_strategies
        self.ctgan_list = ctgan_list
        self.target_value = target_value

    def fit(self, X, y):
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
                new_data = self.ctgan_list[i].sample(5000)

                xy_train = pd.concat([X, y_i], axis=1)
                xy_train = pd.concat([xy_train, new_data], axis=0).reset_index(drop=True)

                target_col = f'{self.target_value}_{i+1}번선수'
                X_train = xy_train.drop(columns=[target_col])
                y_train = xy_train[target_col]
            elif self.smote_strategies is not None:
                smote_ = SMOTE(sampling_strategy=self.smote_strategies[i], random_state=self.seed)
                X_train, y_train = smote_.fit_resample(X, y_i)

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
        importances = np.mean([model.get_feature_importance() for model in self.models], axis=0)

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


def evaluate_(y_pred, y_val, target_value='단승', contain_연승=False, y_val_연승=None):
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
            if y_val_연승 is not None:
                true_indices = np.where(y_val_연승[i] == 1)[0]  # 실제 값에서 1인 인덱스
            else:
                true_indices = np.where(y_val.iloc[:, i] == 1)[0]  # 실제 값에서 1인 인덱스
                if target_value != '복승':
                    print('target value가 단승, 삼복승이면, 복승식에 대한 y_val 값이 필요합니다.')

            if len(np.intersect1d(pred_indices, true_indices)) > 0:  # 교집합이 있으면 적중
                hits += 1
        연승_score = hits / y_pred.shape[0]  # 적중한 비율 계산
        return accuracy, 연승_score