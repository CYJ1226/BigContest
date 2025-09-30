import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             matthews_corrcoef, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve)
from sklearn import metrics

from lightgbm import LGBMClassifier
import lightgbm as lgbm
from xgboost import XGBClassifier
import xgboost as xgb


#각 모델의 평가 지표를 보여주는 함수
#정확도,정밀도,재현율,,F1 점수, mcc, 혼동 행렬, AUC-ROC,PR-AUC

def model_eval(model_object, model_name: str, X_test, y_test):
    prediction = model_object.predict(X_test)
    
    # 평가지표 계산
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    mcc = matthews_corrcoef(y_test, prediction)

    print(f'--- {model_name} 모델 평가 ---')
    print(f'정확도(Accuracy): {accuracy:.4f}')
    print(f'정밀도(Precision): {precision:.4f}')
    print(f'재현율(Recall): {recall:.4f}')
    print(f'F1 점수(F1 Score): {f1:.4f}')
    print(f'매튜 상관 계수(MCC): {mcc:.4f}\n')
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 그래프 레이아웃 설정
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} 모델 평가', fontsize=18)

    # 혼동 행렬
    conf_matrix = confusion_matrix(y_test, prediction, labels=[1, 0])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['폐업(1)', '운영 중(0)'],
                yticklabels=['폐업(1)', '운영 중(0)'])
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_title('Confusion Matrix (혼동 행렬)', fontsize=14)
    axes[0, 1].axis('off')

    # ROC & PR 커브
    if hasattr(model_object, "predict_proba"):
        y_pred_proba = model_object.predict_proba(X_test)[:, 1]
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        axes[1, 0].set_title('ROC Curve', fontsize=14)
        axes[1, 0].legend(loc="lower right")

        # PR-AUC
        prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(rec, prec)
        axes[1, 1].plot(rec, prec, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        axes[1, 1].set_title('Precision-Recall Curve', fontsize=14)
        axes[1, 1].legend(loc="lower left")

    plt.subplots_adjust(top=0.9, hspace=0.2)
    plt.show()


class ModelTuner:
    """
    데이터를 받아 머신러닝 모델의 하이퍼파라미터 튜닝, 학습, 평가를 수행하는 클래스.
    """
    def __init__(self, model_name,X_train, X_test,y_train,y_test):
        """
        클래스를 초기화합니다.
        
        Args:
            model_name (str): 사용할 모델의 이름. ('logistic', 'knn', 'gnb', 'svc_linear', 'svc_rbf','xgb','lgbm')
            X (pd.DataFrame): 입력 피처 데이터
            y (pd.Series): 타겟 데이터
        """
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train)
        self.cv_number = 5
        self.best_params = None
        self.final_model = None

    def _define_search_space(self):
        """모델 이름에 따라 Hyperopt 탐색 공간을 정의합니다."""
        if self.model_name == 'logistic':
            return {
                'C': hp.loguniform('C', np.log(0.01), np.log(100)),
                'penalty': hp.choice('penalty', ['l1', 'l2']),
                'solver': hp.choice('solver', ['liblinear', 'saga']),
                'max_iter': hp.quniform('max_iter', 100, 1000, 100),
                'class_weight': hp.choice('class_weight', [None, 'balanced'])
            }
            
        elif self.model_name == 'knn':
            return {
                'n_neighbors': hp.quniform('n_neighbors', 2, 50, 1),
                'weights': hp.choice('weights', ['uniform', 'distance']),
                'metric': hp.choice('metric', ['euclidean', 'manhattan', 'minkowski'])
            }
            
        elif self.model_name == 'gnb':
             return {'var_smoothing': hp.loguniform('var_smoothing', np.log(1e-10), np.log(1e-8))}
            
        elif self.model_name == 'svc_linear':
            return {
                'C': hp.loguniform('C', np.log(0.01), np.log(1000)),
                'class_weight': hp.choice('class_weight', [None, 'balanced'])
            }
            
        elif self.model_name == 'svc_rbf':
            return {
                'C': hp.loguniform('C', np.log(0.01), np.log(1000)),
                'gamma': hp.loguniform('gamma', np.log(0.0001), np.log(1)),
                'class_weight': hp.choice('class_weight', [None, 'balanced'])
            }
            
        elif self.model_name == 'xgb':
            return {
                'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                'max_depth': hp.quniform('max_depth', 3, 15, 1),
                'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                'subsample': hp.uniform('subsample', 0.6, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                'gamma': hp.uniform('gamma', 0, 0.5),
                'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 40) # 데이터 불균형 처리
            }
            
        elif self.model_name == 'lgbm':
            return {
                'n_estimators': hp.quniform('n_estimators', 100, 2000, 100),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
                'max_depth': hp.quniform('max_depth', 3, 15, 1),
                'min_child_samples': hp.quniform('min_child_samples', 20, 100, 5),
                'subsample': hp.uniform('subsample', 0.6, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                'reg_alpha': hp.uniform('reg_alpha', 0, 1), # L1 규제
                'reg_lambda': hp.uniform('reg_lambda', 0, 1) # L2 규제
            }
            
        else:
            raise ValueError("지원하지 않는 모델 이름입니다.")

    def _get_model(self, params):
        """파라미터를 받아 모델 객체를 반환합니다."""
        if self.model_name == 'logistic':
            params['max_iter'] = int(params['max_iter'])
            return LogisticRegression(**params, random_state=42)
            
        elif self.model_name == 'knn':
            params['n_neighbors'] = int(params['n_neighbors'])
            return KNeighborsClassifier(**params)
            
        elif self.model_name == 'gnb':
            return GaussianNB(**params)
            
        elif self.model_name == 'svc_linear':
            return SVC(kernel='linear', probability=True, **params, random_state=42)
            
        elif self.model_name == 'svc_rbf':
            return SVC(kernel='rbf', probability=True, **params, random_state=42)
            
        elif self.model_name == 'xgb':
            for p in ['n_estimators', 'max_depth', 'min_child_weight']:
                if p in params:  # 파라미터가 존재할 때만 변환
                    params[p] = int(params[p])
            return XGBClassifier(**params, tree_method="hist", device="cuda", random_state=42,
                             n_jobs=-1,early_stopping_rounds=50, use_label_encoder=False) 
                             

        elif self.model_name == 'lgbm':
        
            for p in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
                if p in params:
                    params[p] = int(params[p])
            return LGBMClassifier(**params, device='cuda', random_state=42, n_jobs=-1, verbose=-1)

    def _objective(self, params):
        """Hyperopt를 위한 목적 함수입니다."""
        model = self._get_model(params)
        
        # XGB,LGBM의 early_stopping_rounds를 위해 eval_set 준비
        if self.model_name in ['xgb', 'lgbm']:
            recall_scores_list= []
            skf = StratifiedKFold(n_splits=self.cv_number, shuffle=True, random_state=42)
        
            for tr_index, val_index in skf.split(self.X_train, self.y_train):
            
                X_tr, X_val = self.X_train.iloc[tr_index], self.X_train.iloc[val_index]
                y_tr, y_val = self.y_train.iloc[tr_index], self.y_train.iloc[val_index]

                if self.model_name == 'xgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
                
                elif self.model_name == 'lgbm':
                    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)],
                          callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)])

                preds = model.predict(X_val)
                recall_scores_list.append(recall_score(y_val, preds))
            score = np.mean(recall_scores_list)
                
        else:
            X_data = self.X_train.toarray() if self.model_name in ['gnb'] else self.X_train
            score = cross_val_score(model, X_data, self.y_train, cv=self.cv_number, scoring='recall').mean()

        return {'loss': -score, 'status': STATUS_OK}

    def tune(self, max_evals=50):
        """하이퍼파라미터 튜닝을 실행합니다."""
        print(f"--- {self.model_name} 모델의 하이퍼파라미터 튜닝 시작 ---")
        space = self._define_search_space()
        trials = Trials()
    
        # fmin의 반환값을 best_params_from_fmin으로 받습니다.
        best_params_from_fmin = fmin(
            fn=self._objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42)
        )
        print(f"--- 튜닝 완료 ---")
    
        # 인덱스를 실제 값으로 매핑할 목록
        choice_mappings = {
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'class_weight': [None, 'balanced']
        }
    
        for name, index in best_params_from_fmin.items():
            if name in choice_mappings:
                best_params_from_fmin[name] = choice_mappings[name][int(index)]

        self.best_params = best_params_from_fmin
        print("########최적의 하이퍼파라미터########")
        print(self.best_params)
        print("\n")
        
    def train_final_model(self):
        """최적의 파라미터로 최종 모델을 학습합니다."""
        if self.best_params is None:
            raise Exception("튜닝이 먼저 실행되어야 합니다. tune() 메소드를 호출하세요.")

        self.final_model = self._get_model(self.best_params)
        
        if self.model_name in ['gnb']:
            self.final_model.fit(self.X_train.toarray(), self.y_train)
            
        elif self.model_name == 'xgb':
            self.final_model.fit(self.X_train, self.y_train, eval_set=[(self.X_tr, self.y_tr), (self.X_val, self.y_val)], verbose=False)
            
        elif self.model_name == 'lgbm':
            self.final_model.fit(self.X_train, self.y_train, eval_set=[(self.X_tr, self.y_tr), (self.X_val, self.y_val)],callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)])
            
        else:
            self.final_model.fit(self.X_train, self.y_train)

        print("최종 모델 학습 완료.")

    def evaluate(self):
        if self.final_model is None:
            raise Exception("모델이 먼저 학습되어야 합니다.")
        
        X_test_data = self.X_test.toarray() if self.model_name in ['gnb'] else self.X_test
        model_eval(self.final_model, self.model_name.upper(), X_test_data, self.y_test)

    def run(self, tune_evals=50):
        """전체 워크플로우를 실행합니다."""
        self.tune(max_evals=tune_evals)
        self.train_final_model()
        self.evaluate()
