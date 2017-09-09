# techjam_online_audition
Code from Data Track Online Competition (question2)

This program run on Python3.6 environment with dependency library below.

### Requirement:
1. scikit_learn 0.19b2
2. XGBoost 0.6
3. TensorFlow 1.2
4. numpy 1.13.0
5. hyperopt 0.1
6. Scipy 0.19.0
7. Keras 2.0.5
8. pandas 0.20.1
9. h5py 2.7.0
10. lightgbm 2.0

To install requirement libraries please use this command
```
 $ pip install -r requirements.txt 
```

============================================================
### Folder Structure:
		- data		  folder contain preprocessed dataset and label
		- params	  folder contain best parameter from each training model
		- results	  folder contain result files predicted from ensemble model
		- saves		  folder contain model and weight of Keras trained model
		- features  folder contain feature important
    
============================================================
### File explanation:
		- run_model.py		File to run training and evaluation 
		- utils.py		Collection of utility scripts, such as...
					- get_params
					- write_params
					- write_result
					- matric_score
		- DataProvider.py	Function to preprocessing data and create train, test, 											and validation dataset file for training
		- models.py		Collection of estimators for training
		- nn_model.py       	Neural Network using keras

============================================================
### How to train random search of specific model (e.g. XGBoost)
```
 $ python run_model.py --train_random --estimator 'xgb'
```
	
Option Estimator name: (replace 'all' with estimator options below to train specific model)
    xgb: XGBoost Classifier
    lgb: LightGBM Classifier
    log: Logistic Regression
    knn: KNeighbors Classifier
    rfo: RandomForest Classifier 
    ada: AdaBoost Classifier
    ext: ExtraTrees Classifier
    svc: Support Vector Classifier
    keras: Keras Neural Networks

 ** Note: all model except 'keras' will run with RandomizedSearchCV algorithm in Scikit learn. 
 ** For 'keras', the model will run with TPE search algorithm using hyperopt library to find best parameter and save model and weights in folder 'saves/'

 ### Training iteration option: (default 10)
You can set number of iteration for training, for example: training XGBoost 100 iteration
```
 $ python run_model.py --train_random --estimator 'xgb' -n 100
```
** Note: training keras neural nets with more iteration will take more time. (recommend to use --train_nn for neural nets training to find more specific best parameters

 ### Training score option: (default 'accuracy')
You can define score options to select best parameter base on other score except 'accuracy'. For example, training RandomForest 100 iterations and use score 'recall' to select best parameter.
```
 $ python run_model.py --train_random --estimator 'rfo' -n 100 --score 'recall'
```
	#### Available score type:
	'accuracy' 		accuracy score
	'precision'		precision score
	'recall'		recall score
	'f1'			f1 score
	'roc-auc'		ROC-AUC score

============================================================
### How to manual train neural nets with Keras
This command will manually train data with neural nets using Keras and save the model and weight in folder 'saves/'
```
 $ python run_model.py --train_nn 
```
 ### Fine tuning parameter
You can fine tuning training parameters below
	
	dropout: dropout rate per each layer (default 0.5)
	hidden_unit: number of neural unit per hidden layer (default 10)
	layers: number of additional hidden layer (default 1)
	optimizer: optimizer method (default adam), currently support adam, adadelta, sgd
	init: weight initializing method (default normal), currently support 'glorot_uniform', 'normal', 'uniform'
	batch_size: number of batch size for training (default 32)
	epochs: number of training epochs (default 100)
	
For example, we can train 100 epochs with 'adadelta' optimizer, dropout 0.4, add 3 more hidden layers, set hidden unit to 20, and initialize weight using 'uniform'. (and use 'f1' for checking model performance score
```
 $ python run_model.py --train_nn --epochs 100 --score 'f1' --optimizer 'adadelta' --dropout 0.4 --layers 3 --hidden_unit 20 --init 'uniform'
```
============================================================
### How to evaluate model with ensemble voting method and predict result
This command will run all model to predict probability of each input and then grouping all combination matching of model group to make vote, then select only the best result of best combination of voting model.
```
 $ python run_model.py —-predict --ensemble 'vote'
```

### How to evaluate model with ensemble stacking method and predict result
You can use all base model to predict prob and stack for 2nd layer training.
2nd layer model are XGBoost, RandomForest, and LogisticRegression
3rd layer model are XGBoost and LogisticRegression
```
 $ python run_model.py —-predict --ensemble 'stack'
```
** Note: this version cannot change 2nd and 3rd layer model from command line. If you want to change, please reconfigure file models.py 

 #### specify threshold score for evaluation 
This command will select models which validation score greater than threshold. For example, evaluating all models with threshold more than 0.9 (90%)
```
 $ python run_model.py —-predict --ensemble 'vote' --threshold 0.9
```
** Note: the ensemble voting function will automatically use validation score from best_params of each model regardless of score type we trained. Please make sure using same type of score during train

 #### specify score type of ensemble model (default 'accuracy')
This command will select score type to evaluate the ensemble model. For example, 'f1' score
```
 $ python run_model.py —-predict --ensemble 'vote' --score 'f1'
```
** Note: this command doesn't change the score type of trained model parameter, as the validation score was stored according to score type during training. This command only check performance of prediction from ensemble voting result with specific score type.

** Note: result file will be stored in folder 'results/' with timestamp as file name

============================================================
### For Help
```
 $ python run_model.py —-help
```
