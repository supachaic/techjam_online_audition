import os
import numpy as np
import datetime
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

from utils import matric_score, write_params

class keras_model:
    def __init__(self, n_features, n_classes,
                 dropout, hidden_unit, n_layers,
                 batch_size, epochs,
                 optimizer, init):
                
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout = dropout
        self.hidden_unit = hidden_unit
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.init = init
        self.batch_size = batch_size
        self.epochs = epochs
    
    def train(self, X_train, y_train, X_val, y_val, score_name, num):
        
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        
        params = {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'dropout': self.dropout,
            'hidden_unit': self.hidden_unit,
            'n_layers': self.n_layers,
            'optimizer': self.optimizer,
            'init': self.init,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }

        # set last activation function and loss function
        if self.n_classes == 2:
            last_activation = 'sigmoid'
            loss_fn = 'binary_crossentropy'
            n_output = 1
        else:
            last_activation = 'softmax'
            loss_fn = 'categorical_crossentropy'
            n_output = self.n_classes

        lb = LabelBinarizer()
        y_train_onehot = lb.fit_transform(y_train) 
        y_val_onehot = lb.fit_transform(y_val)
            
        # create model
        model = Sequential()
        model.add(Dense(self.hidden_unit, input_dim=self.n_features, kernel_initializer=self.init, activation='relu'))
        model.add(Dropout(rate=self.dropout))
        for i in range(self.n_layers):
            model.add(Dense(self.hidden_unit, kernel_initializer=self.init, activation='relu'))
            model.add(Dropout(rate=self.dropout))
        model.add(Dense(n_output, kernel_initializer=self.init, activation=last_activation))
        
        # Compile model
        model.compile(loss=loss_fn, optimizer=self.optimizer, metrics=['accuracy'])

        model.fit(X_train, y_train_onehot, batch_size=self.batch_size, epochs=self.epochs)
        
        best_score = model.evaluate(X_train, y_train_onehot)
        best_score = best_score[1]
        print("\n %s: %.2f%%" % (model.metrics_names[1], best_score*100))
        
        y_pred = model.predict(X_val)
        
        if self.n_classes == 2:
            y_pred = np.hstack([np.ones_like(y_pred) - y_pred, y_pred])
            
        y_pred = np.argmax(y_pred, axis=1)
        
        # Prediction evaluate score
        val_score = matric_score(y_val, y_pred, score_name)
        print("\nValidation Test %s score: %.2f%%" % (score_name, val_score*100.0))

        conf_matrix = confusion_matrix(y_val, y_pred)
        print("\nConfusion Matrix: \n", conf_matrix)

        cl_report = classification_report(y_val, y_pred)
        print("\nClassification Report: \n", cl_report)
        
        # Save model and weights
        timestamp = f"{datetime.datetime.now():%Y%m%d%H%M}"
        save_path = 'saves/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        model_json = model.to_json()
        model_path = os.path.join(save_path, timestamp + '.json')
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
            print('Saved model to %s' % model_path)
            
        # serialize weights to HDF5
        weights_path = os.path.join(save_path, timestamp + '.h5')
        model.save_weights(weights_path)
        print("Saved weight to %s" % weights_path)

        # Write params to file
        estimator_name = 'keras'
        write_params(estimator_name, params, best_score, val_score, timestamp, num)
        
        return val_score

    def predict_proba(self, X, model_path, weights_path):
        
        from keras.models import model_from_json
        
        # load model from json file and create model
        with open(model_path, 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        
        # load weights from h5 file
        model.load_weights(weights_path)
        
        y_result = model.predict(X)   
        if self.n_classes == 2:
            y_result = np.hstack([np.ones_like(y_result) - y_result, y_result])
            
        return y_result