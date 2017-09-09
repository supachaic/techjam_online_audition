
# coding: utf-8

# In[3]:

import os
import datetime
import argparse
import logging
import sys
import numpy as np

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from DataProvider import get_data
from utils import write_result, write_params, matric_score, get_timestamp
from models import select_model, random_model, ensemble
from nn_model import keras_model



def _build_parser():
    parser = argparse.ArgumentParser()
    
    # Training arguments for scikit-learn models
    parser.add_argument("--train_random", action="store_true",
                        help="Randomized Training model")
    parser.add_argument("--estimator", type=str, default='all',
                        help="Estimator to train")
    parser.add_argument("-n", "--n_iter", type=int, default=10,
                        help="Number of Iteration for training")
    parser.add_argument("--num", type=int, default=1,
                        help="Number of each Model for training")    
    parser.add_argument("--score", type=str, default='accuracy',
                        help="Type of score to evaluate model")
    parser.add_argument("--seed", type=int, default=None,
                        help="Number for random seed") 
    
    # Training arguments for neural network
    parser.add_argument("--train_nn", action="store_true",
                        help="Training neural network using Keras/Tensorflow")
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of Epoch for training')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='Batch size for training')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument("--layers", type=int, default=1,
                        help='number of additional hidden layer')
    parser.add_argument("--hidden_unit", type=int, default=10,
                        help='number of hidden unit for each layer')
    parser.add_argument("--optimizer", type=str, default='adam',
                        help='training optimizer')
    parser.add_argument("--init", type=str, default='normal',
                        help='weight initialization')
    
    # Evaluation arguments
    parser.add_argument("--predict", action="store_true",
                        help="Evaluate test data and predict result")
    parser.add_argument("--ensemble", type=str, default='vote',
                        help="Ensemble method, vote or stack")
    parser.add_argument("--threshold", type=float, default=0.,
                        help="Model Best Score Threshold for ensemble")
    parser.add_argument("--num_imp", type=int, default=10,
                        help='number of feature importances for stack')
    
    return parser.parse_args()



def _check_args(args):
    est_set = {'xgb', 'lgb', 'log', 'rfo', 'ext', 'ada', 'knn', 'svc', 'keras', 'all'}
    score_set = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
    optimizer_set = {'adam', 'adadelta', 'sgd'}
    init_set = {'glorot_uniform', 'normal', 'uniform'}
    ensemble_set = {'vote', 'stack'}
    
    assert args.estimator in est_set, 'please select estimator'
    assert isinstance(args.n_iter, int), 'please enter integer number'
    assert isinstance(args.num, int), 'please enter integer number'
    assert args.score in score_set, 'please select correct score method'
    assert isinstance(args.epochs, int), 'epochs must be interger'
    assert isinstance(args.batch_size, int), 'batch size must be interger'
    assert isinstance(args.threshold, float) and args.threshold < 1.0, 'threshold must be float between 0.0 and 1.0'
    assert isinstance(args.dropout, float) and args.dropout < 1.0, 'dropout must be float between 0.0 and 1.0'
    assert isinstance(args.layers, int), 'number of layers must be interger'
    assert isinstance(args.hidden_unit, int), 'hidden unit must be interger'
    assert args.optimizer in optimizer_set, 'please select optimizer from (adam, adadelta, sgd)'
    assert args.init in init_set, 'please select weight init from (glorot_uniform, normal, uniform)'
    assert args.ensemble in ensemble_set, 'please select ensemble method from (vote, stack)'
    assert isinstance(args.num_imp, int), 'Number must be interger'


def main():
    
    # build parser and check arguments
    args = _build_parser()
    _check_args(args)
    
    # Setup Estimator
    '''Estimator name: 
    xgb: XGBoost Classifier
    log: Logistic Regression
    knn: KNeighbors Classifier
    rfo: RandomForest Classifier 
    ada: AdaBoost Classifier
    ext: ExtraTrees Classifier
    svc: Support Vector Classifier
    keras: Keras Neural Networks
    '''
    
    if not args.estimator == 'all':
        estimators = [args.estimator]
    elif args.estimator == 'all':
        estimators = ['xgb', 'lgb', 'log', 'rfo', 'ext', 'ada', 'knn', 'svc']
    
    # Training neural nets with keras
    if args.train_nn:
        estimator_name = 'keras'
        print('Training %s...' % estimator_name)
        
        params = {
            'n_features': n_features,
            'n_classes': n_classes,
            'dropout': args.dropout,
            'hidden_unit': args.hidden_unit,
            'n_layers': args.layers,
            'optimizer': args.optimizer,
            'init': args.init,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
        }
        estimator = keras_model(**params)
        
        train_kwargs = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'score_name': args.score,
                    'num': args.num
            }
        _ = estimator.train(**train_kwargs)
        print('params: \n', params)
            
    # Training random search CV with scikit-learn models
    if args.train_random:
        for estimator_name in estimators:
            print('Training %s...' % estimator_name)
            
            if not estimator_name == 'keras':
                seed = args.seed if args.seed != None else np.random.randint(100)
                estimator, params = select_model(estimator_name, n_features, n_classes, seed)

                # kwargs dict for train and predict
                train_kwargs = {
                        'estimator': estimator,
                        'params': params,
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_val': X_val,
                        'y_val': y_val,
                        'n_iter': args.n_iter,
                        'score_name': args.score,
                }

                # Train model and Predict results
                best_params, best_score, val_score = random_model(**train_kwargs)
                timestamp = get_timestamp()

                # Write params to file
                write_params(estimator_name, best_params, best_score, val_score, timestamp, args.num)
                
            elif estimator_name == 'keras':

                space_params = {
                    'n_features': n_features,
                    'n_classes': n_classes,
                    'dropout': hp.uniform('dropout', .20, .80),
                    'hidden_unit': hp.quniform('hidden_unit', 10, 50, q=1),
                    'n_layers': hp.choice('n_layers', [1, 2, 3, 4]),
                    'optimizer': hp.choice('optimizer', ['adam', 'adadelta', 'sgd']),
                    'init': hp.choice('init', ['glorot_uniform', 'normal', 'uniform']),
                    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
                    'epochs': hp.quniform('epochs', 100, 1000, q=1),
                    'score_name': args.score,
                    'num': args.num,
                }
                trials = Trials()
                best_params = fmin(random_nn, space_params, algo=tpe.suggest, max_evals=args.n_iter, trials=trials)
                print('best_params \n', best_params)
                

    # Evaluate with ensemble method and predict result
    if args.predict:
        
        eva_kwargs = {
            'estimators': estimators,
            'threshold': args.threshold,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'score_name': args.score,
            'n_classes': n_classes,
        }
        
        # Predict with ensemble voting and write result
        prediction = ensemble(**eva_kwargs)
        if args.ensemble == 'vote':
            result = prediction.vote()
        elif args.ensemble == 'stack':
            result = prediction.stack(args.num_imp)
            
        timestamp = get_timestamp()
        write_result(result, label_list, timestamp)

def random_nn(space_params):
    params = {
            'n_features': space_params['n_features'],
            'n_classes': space_params['n_classes'],
            'dropout': space_params['dropout'],
            'hidden_unit': int(space_params['hidden_unit']),
            'n_layers': int(space_params['n_layers']),
            'optimizer': space_params['optimizer'],
            'init': space_params['init'],
            'batch_size': int(space_params['batch_size']),
            'epochs': int(space_params['epochs']),            
    }
    estimator = keras_model(**params)
    
    train_kwargs = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'score_name': space_params['score_name'],
                    'num': space_params['num'],
        }   
    acc = estimator.train(**train_kwargs)
    
    return {'loss': -acc, 'status': STATUS_OK}        

# In[ ]:

if __name__ == '__main__':
    #sys.stdout = open('logs/train.log', 'a')
    
    # Load data and split to train_set and validation_set
    X_train, X_val, y_train, y_val, X_test, label_list = get_data()
    n_features = X_train.shape[1]
    n_classes = len(label_list)
    print('Load data complete.')

    main()

