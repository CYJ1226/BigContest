import numpy as np
import pandas as pd
import seaborn as sns
import shap
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
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier
import lightgbm as lgbm
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
 
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
                'C': hp.loguniform('C', np.log(0.01), np.log(100)),
                'class_weight': hp.choice('class_weight', [None, 'balanced'])
            }
            
        elif self.model_name == 'svc_rbf':
            return {
                'C': hp.loguniform('C', np.log(0.01), np.log(100)),
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
            X_data = self.X_train if self.model_name in ['gnb'] else self.X_train
            #X_data = self.X_train.toarray() if self.model_name in ['gnb'] else self.X_train
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
            self.final_model.fit(self.X_train, self.y_train)
            
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
        
        X_test_data = self.X_test if self.model_name in ['gnb'] else self.X_test
        model_eval(self.final_model, self.model_name.upper(), X_test_data, self.y_test)

    def run(self, tune_evals=50):
        """전체 워크플로우를 실행합니다."""
        self.tune(max_evals=tune_evals)
        self.train_final_model()
        self.evaluate()

class Boost_model:
    def __init__(self,model_type: str,train_data,GPU ='off',
                 early_stopping_rounds = 50,n_estimators=500,learning_rate=0.05,
                 min_gain_to_split=0.05, random_state=42, subsample=0.8,colsample_bytree=0.8):
        
        self.train_data = train_data
        self.model_type = model_type
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.learning_rate= learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.min_gain_to_split = min_gain_to_split
        self.scale_pos_weight = (len(self.train_data.y_train) - sum(self.train_data.y_train)) / sum(self.train_data.y_train)
        self.check_point = 0
        self.GPU = GPU

        params = {
        'n_estimators': self.n_estimators,
        'learning_rate': self.learning_rate,
        'subsample': self.subsample,
        'colsample_bytree': self.colsample_bytree,
        'random_state': self.random_state}

        if self.model_type == 'lgbm':

            params.update({
            'class_weight': 'balanced',
            'min_gain_to_split': self.min_gain_to_split})    
            
            if self.GPU == 'on':
                params['device'] = 'cuda'                  
                
            self.model = LGBMClassifier(**params)

        elif self.model_type == 'xgb':

            params.update({
            'scale_pos_weight': self.scale_pos_weight,
            'n_jobs': -1,
            'use_label_encoder': False,
            'early_stopping_rounds':self.early_stopping_rounds})

            if self.GPU == 'on':
                params['device'] = 'cuda'
                params['tree_method'] = 'hist'

            self.model = XGBClassifier(**params)
        else:
            raise ValueError("정확한 부스트 모델명을 입력해주세요(lgbm,xgb)")                      
        
    def fit(self):
        if self.model is None:
            raise ValueError("정확한 부스트 모델명을 입력해주세요(lgbm,xgb)")
    
        elif self.model_type == 'lgbm':
            self.model.fit(
            self.train_data.X_tr, self.train_data.y_tr,
            eval_set=[(self.train_data.X_tr, self.train_data.y_tr), (self.train_data.X_val, self.train_data.y_val)]
            ,callbacks=[lgbm.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=-1)])

        elif self.model_type == 'xgb':
            self.model.fit(
            self.train_data.X_tr, self.train_data.y_tr,
            eval_set=[(self.train_data.X_tr, self.train_data.y_tr), (self.train_data.X_val, self.train_data.y_val)], verbose=False)
        
        self.check_point += 1

    def evaluation(self):
        self.fit()
        if self.model_type == 'lgbm':
            title = "Light Gradient Boosting Machine"
        elif self.model_type == 'xgb':
            title = "Exreme Gradient Boosting"
        else:
            raise ValueError("정확한 부스트 모델명을 입력해주세요(lgbm,xgb)")
        
        model_eval(self.model,title,self.train_data.X_test,self.train_data.y_test)

    def plot_feature_importance(self,top_n=20):
        if self.check_point == 0:
            raise ValueError("모델을 먼저 학습해야 합니다")
        
        importances = self.model.feature_importances_
        features = self.train_data.X_val.columns

        idx = np.argsort(importances)[::-1][:top_n]
        plt.barh(np.array(features)[idx][::-1], np.array(importances)[idx][::-1])
        plt.title(f"{self.model_type} Feature Importance (Top {top_n})")
        plt.xlabel("Importance score")
        plt.ylabel("Features")
        plt.show()
    
    def permutation_importance_plot(self):
        if self.check_point == 0:
            raise ValueError("모델을 먼저 학습해야 합니다")

        result = permutation_importance(
            self.model, self.train_data.X_val, self.train_data.y_val,
            scoring="f1",
            n_repeats=1,
            random_state=42,
            n_jobs=-1)

        fi = pd.DataFrame({
            "feature": self.train_data.X_val.columns,
            "importance": result.importances_mean
        }).sort_values(by="importance", ascending=False)

        print(fi.head(10))

        top_features = fi.head(20)

        plt.figure(figsize=(8, 6))
        plt.barh(top_features["feature"], top_features["importance"], color="skyblue")
        plt.xlabel("Permutation Importance")
        plt.ylabel("Feature")
        plt.title(f"{self.model_type} Top 20 Features by Permutation Importance")
        plt.gca().invert_yaxis() 
        plt.show()   

class ShapAnalysis:
    """
    학습된 트리 기반 모델(XGBoost, LightGBM 등)의 예측 결과를
    SHAP을 이용해 분석하고 시각화하는 클래스입니다.
    """
    def __init__(self, boost_model, train_data):
        """
        클래스를 초기화하고 모델과 데이터를 설정합니다.

        Args:
            train_data: train/test 데이터가 분리된 객체. 
                        '.X_test', '.y_test' 속성으로 데이터에 접근 가능해야 합니다.
        """
        self.model = boost_model
        self.X_test = train_data.X_test
        self.y_test = train_data.y_test
        self.explainer = shap.TreeExplainer(self.model)
        
        # 분석 결과를 저장할 인스턴스 변수
        self.single_data_point = None
        self.shap_values_single = None
        self.expected_value = self.explainer.expected_value

    def select_sample(self, index=0):
        """분석을 수행할 단일 데이터 샘플을 선택합니다."""
        self.single_data_point = self.X_test.iloc[[index]]
        self.shap_values_single = self.explainer.shap_values(self.single_data_point)
        print(f"--- 데이터 인덱스 {index}번 샘플에 대한 분석을 시작합니다. ---")
        return self

    def text_summary(self):
        """
        선택된 샘플의 예측 결과와 SHAP 기여도를 텍스트로 요약합니다.
        폐업으로 예측하는 데 가장 큰 영향을 준 상위 5개 피쳐만 출력합니다.
        """
        if self.single_data_point is None:
            raise ValueError("먼저 `select_sample` 메서드를 호출하여 분석할 데이터를 선택해주세요.")

        probability_class_1 = self.model.predict_proba(self.single_data_point)[0, 1]
        
        print("\n" + "="*60)
        print("예측 기여도 텍스트 요약")
        print("="*60)
        print(f"🎯 가맹점이 폐업할 예측 확률: {probability_class_1:.4f}\n")
        print(f"📊 모델의 평균 예측 기준값 (Base Value): {self.expected_value[0]:.4f}\n")

        shap_df = pd.DataFrame({
            'Feature': self.single_data_point.columns,
            'SHAP Value (기여도)': self.shap_values_single.flatten()
        })
        
        positive_shap_df = shap_df[shap_df['SHAP Value (기여도)'] > 0]
        
        top5_positive = positive_shap_df.sort_values(by='SHAP Value (기여도)', ascending=False)
        
        print("폐업 예측 요인 TOP 5 피쳐:")

        print(top5_positive.head(5).to_string())
   
        
    def force_plot(self):
        """단일 예측에 대한 SHAP Force Plot을 시각화합니다."""
        if self.single_data_point is None:
            raise ValueError("먼저 `select_sample` 메서드를 호출하여 분석할 데이터를 선택해주세요.")
        shap.initjs()
        print("\n>>>Force Plot (단일 데이터 예측 설명)")
        display(shap.force_plot(
            self.expected_value[0],
            self.shap_values_single,
            self.single_data_point
        ))

    def summary_plot(self):
        """전체 테스트 데이터에 대한 SHAP Summary Plot을 시각화합니다."""
        print("\n>>>Summary Plot (전체 특성 중요도)")
        shap_values_all = self.explainer.shap_values(self.X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_all, self.X_test, show=False)
        plt.title("SHAP Summary Plot", fontsize=14)
        plt.tight_layout()
        plt.show()

    def custom_bar_plot(self):
        """
        폐업으로 예측하게 만드는 TOP 5 피쳐에 대한
        Custom Bar Plot을 시각화합니다.
        """
        if self.single_data_point is None:
            raise ValueError("먼저 `select_sample` 메서드를 호출하여 분석할 데이터를 선택해주세요.")
        
        print("\n>>>Custom Bar Plot (폐업 예측 긍정 영향 TOP 5)")
        
        shap_df = pd.DataFrame({
            'Feature': self.single_data_point.columns,
            'SHAP Value': self.shap_values_single.flatten()
        })

        positive_shap_df = shap_df[shap_df['SHAP Value'] > 0].sort_values(by='SHAP Value', ascending=False)
        top5_positive_plot = positive_shap_df.head(5).sort_values(by='SHAP Value', ascending=True)
        colors = ['red'] * len(top5_positive_plot)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top5_positive_plot['Feature'], top5_positive_plot['SHAP Value'], color=colors)
        plt.title("폐업 예측 긍정 영향 TOP 5", fontsize=16)
        plt.xlabel("SHAP Value (Contribution to Prediction)", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.show()

    def single_sample_analysis(self,index=0):
        self.select_sample(index=index)
        self.text_summary()
        self.custom_bar_plot()
        self.force_plot()
