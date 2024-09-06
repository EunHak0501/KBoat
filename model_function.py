from sklearn.metrics import accuracy_score, f1_score
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

def cal_params(target_value='복승', seed=42):
    params_list = []
    if target_value == '복승':
        params_list = []
        params_1 = {
            'random_seed': seed,
            'iterations': 200,
            'learning_rate': 0.09719738901239813,
            'l2_leaf_reg': 8.08214833285572,
            'class_weights': {0: 1.0, 1: 3.615033608391579},
            'verbose': 0,
        }
        params_list.append(params_1)
        params_2 = {
            'random_seed': seed,
            'iterations': 200,
            'learning_rate': 0.08986154076025232,
            'l2_leaf_reg': 5.356161464826255,
            'class_weights': {0: 1.0, 1: 4.271282409160504},
            'verbose': 0,
        }
        params_list.append(params_2)
        params_3 = {
            'random_seed': seed,
            'iterations': 200,
            'learning_rate': 0.06849990307395196,
            'l2_leaf_reg': 2.885808980448922,
            'class_weights': {0: 1.0, 1: 4.367286456552369},
            'verbose': 0,
        }
        params_list.append(params_3)
        params_4 = {
            'random_seed': seed,
            'iterations': 200,
            'learning_rate': 0.09474850624939075,
            'l2_leaf_reg': 6.999359411562832,
            'class_weights': {0: 1.0, 1: 2.5335515942659743},
            'verbose': 0,
        }
        params_list.append(params_4)
        params_5 = {
            'random_seed': seed,
            'iterations': 200,
            'learning_rate': 0.08674133482890571,
            'l2_leaf_reg': 7.063997127332498,
            'class_weights': {0: 1.0, 1: 3.0492130863383484},
            'verbose': 0,
        }
        params_list.append(params_5)
        params_6 = {
            'random_seed': seed,
            'iterations': 200,
            'learning_rate': 0.15744810753127567,
            'l2_leaf_reg': 1.0798243810773542,
            'class_weights': {0: 1.0, 1: 3.0154819061761944},
            'verbose': 0,
        }
        params_list.append(params_6)

    return params_list

class custom_CatBoostClassifier():
    def __init__(self, params_list):
        self.models = [CatBoostClassifier(**params_list[i]) for i in range(6)]
        self.feature_names = None
        self.X = None

    def fit(self, X, y, eval_set=None, cat_features=None):
        y = np.array(y)  # y를 numpy 배열로 변환

        # feature_names를 저장 (X가 pandas DataFrame인 경우)
        if isinstance(X, pd.DataFrame):
            self.X = X
            self.feature_names = X.columns
        else:
            raise ValueError("X must be a pandas DataFrame with column names.")

        if eval_set is not None:
            X_val, y_val = eval_set

        for i in range(6):
            y_i = y[:, i]  # i번째 레이블에 대한 y 값

            if eval_set is not None:
                y_val_i = y_val[:, i]  # i번째 레이블에 대한 y_val 값
                eval_set_i = (X_val, y_val_i)
                self.models[i].fit(
                    X, y_i,
                    eval_set=eval_set_i,
                    cat_features=cat_features
                )
            else:
                eval_set_i = None
                self.models[i].fit(
                    X, y_i,
                    cat_features=cat_features
                )
            # print(f'{i+1}번째 레이블 학습 완료')

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

        # 컬럼 이름을 정렬된 형태로 반환
        # averaged_importances = averaged_importances[['Group', 'Importance']].sort_values(by='Importance', ascending=False)

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