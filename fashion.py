
# Machine Learning Homework 4 - Image Classification


# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import os
import sys
import pandas as pd

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend import softmax
from keras import regularizers

from math import tanh


### Already implemented
def get_data(datafile):
    dataframe = pd.read_csv(datafile)
    data = list(dataframe.values)
    labels, images = [], []
    for line in data:
        labels.append(line[0])
        images.append(line[1:])
    labels = np.array(labels)
    images = np.array(images).astype('float32')
    images /= 255
    return images, labels


### Already implemented
def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
    layer1 = trained_model.layers[0]
    weights = layer1.get_weights()[0]

    # Feel free to change the color scheme
    colors = 'hot' if hot else 'binary'
    try:
        os.mkdir('weight_visualizations')
    except FileExistsError:
        pass
    for i in range(num_to_display):
        wi = weights[:,i].reshape(28, 28)
        plt.imshow(wi, cmap=colors, interpolation='nearest')
        if save:
            plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
        else:
            plt.show()


### Already implemented
def output_predictions(predictions, model_type):
    if model_type == 'CNN':
        with open('predictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(np.argmax(pred)) + '\n')
    if model_type == 'MLP':
        with open('MLPpredictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')


def plot_history(history):
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    train_acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    # plot
    plt.figure()
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.show()

    plt.figure()
    plt.plot(train_acc_history)
    plt.plot(val_acc_history)
    plt.show()

    


def create_mlp(args=None):
    # You can use args to pass parameter values to this method

    # Define model architecture
    model = Sequential()
    model.add(Dense(units=512, activation="sigmoid", input_dim=28*28))
    # add more layers...
    model.add(Dense(units=256, activation="sigmoid", input_dim=512))
    model.add(Dense(units=128, activation="sigmoid", input_dim=256))
    model.add(Dense(units=64, activation="sigmoid", input_dim=128))
    model.add(Dense(units=32, activation="sigmoid", input_dim=64))
    model.add(Dense(units=10, activation="softmax", input_dim=32))

    # Define Optimizer
    if args['opt'] == 'sgd':
        optimizer = SGD(lr=args['learning_rate'])
    elif args['opt'] == 'adam':
        optimizer = Adam(lr=args['learning_rate'])

    # Compile
    model.compile(loss= "categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    y_train = keras.utils.to_categorical(y_train, num_classes= 10)
    model = create_mlp(args)
    history = model.fit(x_train, y_train, batch_size=args['batch_size'], validation_split=args['validation_split'], epochs=args['epoch'])
    return model, history


def create_cnn(args=None):
    # You can use args to pass parameter values to this method

    # 28x28 images with 1 color channel
    input_shape = (28, 28, 1)

    # Define model architecture
    
    model = Sequential()
    model.add(Conv2D(filters=32, activation="relu", kernel_size=(3,3), strides=(1,1), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	# can add more layers here...
    model.add(Conv2D(filters=64, activation="relu", kernel_size=(3,3), strides=(1,1), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=128, activation="relu", kernel_size=(3,3), strides=(1,1), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
	# can add more layers here...
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation="softmax"))



    # Optimizer
    if args['opt'] == 'sgd':
        optimizer = SGD(lr=args['learning_rate'])
    elif args['opt'] == 'adam':
        optimizer = Adam(lr=args['learning_rate'])

    # Compile
    model.compile(loss= "categorical_crossentropy" , optimizer=optimizer, metrics=['accuracy'])

    return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes=10 )
    model = create_cnn(args)
    history = model.fit(x_train, y_train, batch_size=args['batch_size'], validation_split=args['validation_split'], epochs=args['epoch'])
    return model, history

def pca_log(x_train, y_train, num_components):
    pc_values = [10, 100, 200]
    error_results_pca = [0,0,0]
    
    for i in range(len(pc_values)):
        pca = decomposition.PCA(pc_values[i])
        x_train_pca = pca.fit(x_train).transform(x_train)
        logisticModel_pca = LogisticRegression(solver = 'liblinear', max_iter=1000).fit(x_train_pca, y_train)
        y_predictions_pca = logisticModel_pca.predict(x_train_pca)
        error_results_pca[i] = 1 - (logisticModel_pca.score(x_train_pca, y_train))
    
    logisticModel_no_pca = LogisticRegression(solver = 'liblinear', max_iter=1000).fit(x_train, y_train)
    y_predictions_no_pca = logisticModel_no_pca.predict(x_train)
    train_error = 1 - (logisticModel_no_pca.score(x_train, y_train))
    
    plotfig(pc_values, error_results_pca, train_error)
    
    print(train_error)
    print(error_results_pca)

def plotfig(pc_values, error_results_pca, y_predictions_no_pca):
    plt.figure
    plt.plot(pc_values, error_results_pca, marker='o')
    plt.axhline(y=y_predictions_no_pca, color = 'r', linestyle = '-')
    plt.xlabel('PC values') 
    plt.ylabel('Prediction Error') 
    plt.title("Prediction Error vs PC values")
    plt.legend(["PCA + LoG", "LoG"], loc ="upper right")
    plt.show()

def train_and_select_model(train_csv, model_type, grading_mode):
    """Optional method. You can write code here to perform a 
    parameter search, cross-validation, etc. """

    x_train, y_train = get_data(train_csv)

    args = {
        'batch_size': 128,
        'validation_split': 0.1,
		'epoch': 10
    }
    
    best_valid_acc = 0
    best_hyper_set = {}
    
    
    ## Select best values for hyperparamters such as learning_rate, optimizer, hidden_layer, hidden_dim, regularization...
   
    if not grading_mode:
        for learning_rate in [0.05, 0.01, 0.005, 0.006]:
            for opt in ['adam', 'sgd']:
                for other_hyper in ['activation']:  ## search over other hyperparameters
                    args['opt'] = opt
                    args['learning_rate'] = learning_rate
                    args['other_hyper'] = other_hyper
                    
                    if model_type == 'MLP':
                        model, history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
                        print(model.summary())
                        # print(pca_log(x_train,y_train,784))
                    else:
                        model, history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args)
                        print(model.summary())
                        # print(pca_log(x_train,y_train,784))
                    
                    validation_accuracy = history.history['val_accuracy']
                    
                    max_valid_acc = max(validation_accuracy)
                    if max_valid_acc > best_valid_acc:
                        best_model = model
                        best_valid_acc = max_valid_acc
                        best_hyper_set['learning_rate'] = learning_rate
                        best_hyper_set['opt'] = opt
                        best_history = history
    else:
        ## In grading mode, use best hyperparameters you found 
        if model_type == 'MLP':
            args['opt'] = 'adam'
            args['learning_rate'] = 0.004
		## other hyper-parameters
            args['hidden_dim'] = 128
            args['hidden_layer'] = 'relu'
            args['activation'] = 'activation'
            
            best_model, best_history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
            # print(best_model.summary())
            # print(pca_log(x_train,y_train,784))
        
        if model_type == 'CNN':
            args['opt'] = 'adam'
            args['learning_rate'] = 0.004
            args['hidden_dim'] = 128
            args['hidden_layer'] = 'relu'
            args['activation'] = 'activation'
            
            best_model, best_history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args)
            # print(best_model.summary())
            # print(pca_log(x_train,y_train,784))
            
        
    return best_model, best_history


if __name__ == '__main__':
    ### Before you submit, switch this to grading_mode = True and rerun ###
    grading_mode = True
    if grading_mode:
        # When we grade, we'll provide the file names as command-line arguments
        if (len(sys.argv) != 3):
            print("Usage:\n\tpython3 fashion.py train_file test_file")
            exit()
        train_file, test_file = sys.argv[1], sys.argv[2]
        

        # train your best model
        best_mlp_model, mlp_history = train_and_select_model(train_file, model_type='MLP', grading_mode=True)
        
        
        x_test, y_test = get_data(test_file)
        # use your best model to generate predictions for the test_file
        mlp_predictions = best_mlp_model.predict_classes(x_test)
        output_predictions(mlp_predictions, model_type='MLP')
        
        x_test = x_test.reshape(-1, 28, 28, 1)
        best_cnn_model, cnn_history = train_and_select_model(train_file, model_type='CNN', grading_mode=True)
        cnn_predictions = best_cnn_model.predict_classes(x_test)
        output_predictions(cnn_predictions, model_type='CNN')

        # Include all of the required figures in your report. Don't generate them here.

    else:
        train_file = '/fashion_train.csv'
        test_file = '/fashion_test.csv'
        # MLP
        mlp_model, mlp_history = train_and_select_model(train_file, model_type='MLP', grading_mode=False)
        plot_history(mlp_history)
        visualize_weights(mlp_model)

        # CNN
        cnn_model, cnn_history = train_and_select_model(train_file, model_type='CNN', grading_mode=False)
        plot_history(cnn_history)