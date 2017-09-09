import os
import numpy as np
import datetime

import scipy.stats as st

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

from utils import write_result, write_params, get_params, get_nums
from utils import matric_score, search_combination, write_features
from nn_model import keras_model


def select_model(estimator_name, n_features, n_classes, seed=42):
    
    if estimator_name == 'xgb':
        estimator = XGBClassifier(nthreads=-1)
        if n_classes == 2:
            objective = 'binary:logistic'
        elif n_classes > 2:
            objective = 'multi:softprob'
        # Parameter for XGBoost
        params = {  
            "n_estimators": st.randint(3, 40),
            "max_depth": st.randint(3, 30),
            "learning_rate": st.uniform(0.05, 0.4),
            "colsample_bytree": st.beta(10, 1),
            "subsample": st.beta(10, 1),
            "gamma": st.uniform(0, 10),
            'objective': [objective],
            'scale_pos_weight': st.randint(0, 2),
            "min_child_weight": st.expon(0, 50),
            "seed": [seed],
        }

    if estimator_name == 'lgb':
        estimator = LGBMClassifier(nthread=-1)
        if n_classes == 2:
            objective = 'binary'
        elif n_classes > 2:
            objective = 'multiclass'
        # Parameter for LGBMClassifier
        params = {  
            "boosting_type": ["gbdt","rf","dart"],
            "colsample_bytree": st.beta(10, 1),
            "learning_rate": st.uniform(0.05, 0.4),
            "max_depth": st.randint(3, 30),
            "min_child_weight": st.expon(0, 50),
            "n_estimators": st.randint(3, 40),
            "num_leaves": st.randint(30, 50),
            'objective': [objective],
            "subsample": st.beta(10, 1),
            "seed": [seed],
        }
    
    elif estimator_name == 'log': 
        estimator = LogisticRegression()
        # Parameter for LogisticRegression
        params = {
            "penalty": ['l2'],
            "C": [0.001, 0.01, 0.1, 1, 10],
            "tol": [1e-4, 1e-3, 1e-2, 1e-1],
            "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
            "max_iter": st.randint(50, 100),
            'random_state': [seed],
        }      

    elif estimator_name == 'knn': 
        estimator = KNeighborsClassifier()
        # Parameter for KNeighborsClassifier
        params = {
            "n_neighbors": st.randint(2, 50),
            "weights": ['uniform', 'distance'],
            "algorithm": ['ball_tree', 'kd_tree', 'brute'],
            "leaf_size": st.randint(10, 30),
            "p": st.randint(1, 2),
        }  

    elif estimator_name == 'rfo': 
        estimator = RandomForestClassifier()
        # Parameter for RandomForestClassifier
        params = {
            "max_depth": [3, None],
            "max_features": st.randint(1, n_features),
            "min_samples_split": st.randint(2, 10),
            "min_samples_leaf": st.randint(1, n_features),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
            'random_state': [seed],
        }  
        
    elif estimator_name == 'ext': 
        estimator = ExtraTreesClassifier()
        # Parameter for ExtraTreesClassifier
        params = {
            "n_estimators": st.randint(5, 50),
            "max_depth": [3, None],
            "max_features": st.randint(1, n_features),
            "min_samples_split": st.randint(2, 10),
            "min_samples_leaf": st.randint(1, n_features),
            "bootstrap": [True],
            "oob_score": [True],
            "criterion": ["gini", "entropy"],
            'random_state': [seed],
        }      

    elif estimator_name == 'ada':
        estimator = AdaBoostClassifier()
        # Parameter for AdaBoost
        params = { 
            'n_estimators':st.randint(10, 100), 
            'learning_rate':st.beta(10, 1), 
            'algorithm':['SAMME', 'SAMME.R'],
            'random_state': [seed],
        }
    
    elif estimator_name == 'svc':
        estimator = SVC()
        # Parameter for SVC
        params = {  
            'C':[0.001, 0.01, 0.1, 1, 10], 
            'degree': st.randint(1, 10),
            'shrinking': [True, False],
            'probability': [True],
            'tol': [1e-3],
            'random_state': [seed],
        }
            
    return estimator, params


def random_model(estimator, params, 
          X_train, y_train, 
          X_val, y_val, n_iter,
          score_name, report=True,
          cv=5, random_state=None):

    # Random Search CV
    clf = RandomizedSearchCV(estimator, params, cv=cv,
                             n_jobs=1, n_iter=n_iter, 
                             scoring=score_name, verbose=1,
                             random_state=random_state)  
    clf.fit(X_train, y_train)  
    best_params = clf.best_params_
    best_score = clf.best_score_
    
    # make predictions for validation data
    y_pred = clf.predict(X_val)
        
    # Prediction evaluate score
    val_score = matric_score(y_val, y_pred, score_name)    
    conf_matrix = confusion_matrix(y_val, y_pred)    
    cl_report = classification_report(y_val, y_pred)
    
    if report:
        print("Train complete.")
        print("Best Parameter: \n", best_params)
        print("Cross Validation Best %s Score: %.2f%%" % (score_name, best_score * 100.0))      
        print("\nValidation Test %s score: %.2f%%" % (score_name, val_score*100.0))
        print("\nConfusion Matrix: \n", conf_matrix)    
        print("\nClassification Report: \n", cl_report)
    
    return best_params, best_score, val_score

class ensemble:
    def __init__(self, estimators, threshold, 
                 X_train, y_train, X_val, y_val, 
                 X_test, score_name, n_classes):
        self.estimators = estimators
        self.threshold = threshold
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.score_name = score_name
        self.n_classes = n_classes
    
    def train(self):
        # train all estimators and return list of prediction for train, validation and test data
        train_predict = []
        val_predict = []
        test_predict = [] 
        estimator_list =[]
        imp_features = []
        features_name = []

        for estimator_name in self.estimators:  
            nums = get_nums(estimator_name)
            try:
                for num in nums:
                    params, val_score, timestamp = get_params(estimator_name, num)
                    if val_score > self.threshold:                    
                        estimator_list.append(estimator_name + num)
                        if estimator_name == 'keras':
                            model_path = os.path.join('saves/', timestamp + '.json')
                            weights_path = os.path.join('saves/', timestamp + '.h5')
                            clf = keras_model(**params)
                            tr_pred = clf.predict_proba(self.X_train, model_path, weights_path)
                            train_predict.append(tr_pred)
                            val_pred = clf.predict_proba(self.X_val, model_path, weights_path)
                            val_predict.append(val_pred)
                            test_pred = clf.predict_proba(self.X_test, model_path, weights_path)
                            test_predict.append(test_pred)
                            print('Ensemble added %s%s' % (estimator_name, num))
                        else:
                            if estimator_name == 'xgb':
                                clf = XGBClassifier(**params)  
                            elif estimator_name == 'lgb':
                                clf = LGBMClassifier(**params)
                            elif estimator_name == 'log':
                                clf = LogisticRegression(**params)
                            elif estimator_name == 'knn':
                                clf = KNeighborsClassifier(**params)
                            elif estimator_name == 'rfo':
                                clf = RandomForestClassifier(**params)
                            elif estimator_name == 'ada':
                                clf = AdaBoostClassifier(**params)
                            elif estimator_name == 'ext':
                                clf = ExtraTreesClassifier(**params)
                            elif estimator_name == 'svc':
                                clf = SVC(**params)

                            clf.fit(self.X_train, self.y_train)
                            tr_pred = clf.predict_proba(self.X_train)
                            train_predict.append(tr_pred)
                            val_pred = clf.predict_proba(self.X_val)
                            val_predict.append(val_pred)
                            test_pred = clf.predict_proba(self.X_test)
                            test_predict.append(test_pred)
                            print('Ensemble added %s%s' % (estimator_name, num))

                            try: 
                                features = clf.feature_importances_
                                imp_features.append(features)
                                features_name.append(estimator_name + num)
                            except: AttributeError

            except: FileNotFoundError
        
        # Array of feature importances, write to file  
        imp_features = np.array(imp_features)
        write_features(imp_features, features_name)
        return train_predict, val_predict, test_predict, estimator_list, imp_features
    
    def vote(self):
        '''Soft vote all combination and search for the best'''
        
        # train all estimators and get probability prediction
        train_predict, val_predict, test_predict, estimator_list, imp_features = self.train()
        
        # search best combination of ensemble vote result 
        best_ensemble = search_combination(val_predict, self.y_val, self.score_name)
        
        # Prediction from training data and score
        tr_list = [train_predict[i] for i in best_ensemble]
        tr_pred = np.mean(tr_list, axis=0)
        tr_pred = np.argmax(tr_pred, axis=1)
        tr_score = matric_score(self.y_train, tr_pred, self.score_name)        
        
        # Predict validation data and score
        pred_list = [val_predict[i] for i in best_ensemble]
        y_pred = np.mean(pred_list, axis=0)
        y_pred = np.argmax(y_pred, axis=1)
        val_score = matric_score(self.y_val, y_pred, self.score_name)
        conf_matrix = confusion_matrix(self.y_val, y_pred)      
        cl_report = classification_report(self.y_val, y_pred)
        
        # Create results from test data   
        res_list = [test_predict[i] for i in best_ensemble]
        y_result = np.mean(res_list, axis=0)
        y_result = np.argmax(y_result, axis=1)

        # Estimators in best ensemble
        est_list = [estimator_list[i] for i in best_ensemble]
        
        # Print report
        print("Vote complete.")
        print("\nTraining data %s score: %.2f%%" % (self.score_name, tr_score*100.0))
        print("\nValidation Test %s score: %.2f%%" % (self.score_name, val_score*100.0))
        print("\nConfusion Matrix: \n", conf_matrix)
        print("\nClassification Report: \n", cl_report)
        print("\nBest Ensemble: \n", est_list)
        
        return y_result
    
    def pred_digit(self, pred_list):
        pred_stack = []
        for predict in pred_list:
            y_pred = np.argmax(predict, axis=1)
            pred_stack.append(y_pred)
        pred_stack = np.array(pred_stack)
        pred_stack = np.transpose(pred_stack)
        return pred_stack
    
    def train_stack(self, n_features, n_classes, 
                    estimators, tr_stack, y_train, 
                    val_stack, y_val, test_stack, score_name):

        tr_pred = []
        val_pred = []
        test_pred = []
        
        for estimator_name in estimators:        
            estimator, params = select_model(estimator_name, n_features, n_classes)
            
            # Train 2nd and 3rd layer with val_stack and evaluate with tr_stack
            train_kwargs = {
                            'estimator': estimator,
                            'params': params,
                            'X_train': val_stack,
                            'y_train': y_val,
                            'X_val': tr_stack,
                            'y_val': y_train,
                            'n_iter': 100,
                            'score_name': score_name,
                            'report': False,
                            'cv': 3,
                            'random_state': 42,
            }
        
            # Random train with stacked data and get best_params
            params, _, _ = random_model(**train_kwargs)
        
            if estimator_name == 'xgb':
                clf = XGBClassifier(**params)
            elif estimator_name == 'lgb':
                clf = LGBMClassifier(**params)
            elif estimator_name == 'rfo':
                clf = RandomForestClassifier(**params)
            elif estimator_name == 'log':
                clf = LogisticRegression(**params)
            elif estimator_name == 'svc':
                clf = SVC(**params)
            elif estimator_name == 'knn':
                clf = KNeighborsClassifier(**params)
            elif estimator_name == 'ada':
                clf = AdaBoostClassifier(**params)
            elif estimator_name == 'ext':
                clf = ExtraTreesClassifier(**params)

            clf.fit(val_stack, y_val)
            tr_prob = clf.predict_proba(tr_stack)
            val_prob = clf.predict_proba(val_stack)
            test_prob = clf.predict_proba(test_stack)
            
            tr_pred.append(tr_prob)
            val_pred.append(val_prob)
            test_pred.append(test_prob)
        return tr_pred, val_pred, test_pred
    
    def stack(self, num_imp):
        '''Stacking models and predict result'''
        
        X_train = self.X_train
        y_train = self.y_train
        X_val = self.X_val
        y_val = self.y_val
        X_test = self.X_test
        score_name = self.score_name
        
        stack_prob = True
        stack_feat = True
        dropout = 0.
        
        # train all estimators and get probability prediction
        train_predict, val_predict, test_predict, estimator_list, imp_features = self.train()

        if stack_prob:
            tr_stack = np.column_stack(train_predict)
            val_stack = np.column_stack(val_predict)
            test_stack = np.column_stack(test_predict)
        else:
            tr_stack = self.pred_digit(train_predict)
            val_stack = self.pred_digit(val_predict)
            test_stack = self.pred_digit(test_predict)
            
        if stack_feat and not len(imp_features) == 0:  
            imp = np.mean(imp_features, axis=0)            
            imp_idx = imp.argsort()[::-1][:num_imp]
            feature_train = X_train[:, imp_idx]
            feature_val = X_val[:, imp_idx]
            feature_test = X_test[:, imp_idx]
            tr_stack = np.column_stack([feature_train, tr_stack])
            val_stack = np.column_stack([feature_val, val_stack])
            test_stack = np.column_stack([feature_test, test_stack])
        
        if dropout > 0.:
            size = len(tr_stack)
            keep = int(size * (1-dropout))
            idx = np.random.choice(size, keep, replace=False)
            tr_stack = tr_stack[idx, :]
            y_train = y_train[idx]
        
        n_features = tr_stack.shape[1]
        n_classes = self.n_classes     
        estimators_2nd = ['rfo', 'xgb', 'lgb']
        estimators_3rd = ['log', 'xgb']
        estimators_list = [estimators_2nd, estimators_3rd]
        
        tr_pred = None
        val_pred = None
        test_pred = None
        
        for estimators in estimators_list:
            kwargs = {
                'n_features': tr_stack.shape[1],
                'n_classes': self.n_classes,
                'estimators': estimators,
                'tr_stack': tr_stack,
                'y_train': y_train,
                'val_stack': val_stack,
                'y_val': y_val,
                'test_stack': test_stack,
                'score_name': score_name,
            }
            tr_pred, val_pred, test_pred = self.train_stack(**kwargs)
            tr_stack = np.column_stack(tr_pred)
            val_stack = np.column_stack(val_pred)
            test_stack = np.column_stack(test_pred)
                    
        # make predictions for train data            
        tr_pred = np.mean(tr_pred, axis=0)
        tr_pred = np.argmax(tr_pred, axis=1)      
                
        # make predictions for validation data
        y_pred = np.mean(val_pred, axis=0)
        y_pred = np.argmax(y_pred, axis=1)

        # Prediction evaluate score
        tr_score = matric_score(y_train, tr_pred, score_name)
        val_score = matric_score(y_val, y_pred, score_name)    
        conf_matrix = confusion_matrix(y_val, y_pred)    
        cl_report = classification_report(y_val, y_pred)

        print("Stack complete.") 
        print("Training data %s score: %.2f%%" % (score_name, tr_score * 100.0))
        print("\nValidation Test %s score: %.2f%%" % (score_name, val_score*100.0))
        print("\nConfusion Matrix: \n", conf_matrix)    
        print("\nClassification Report: \n", cl_report)
      
        # make predictions for test data
        y_result = np.mean(test_pred, axis=0)
        y_result = np.argmax(y_result, axis=1)
        
        return y_result
