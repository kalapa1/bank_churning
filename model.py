import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier


def preprocessing_input_output(file_path): 
    churn_dataset=pd.read_csv(file_path)
    indp_input=churn_dataset.iloc[:,3:13].values
    depd_output=churn_dataset.iloc[:,13].values
    return indp_input,depd_output


def categorical_to_numeric(Xinput,Youtput):
    label_encode_gender = LabelEncoder()
    label_encode_country = LabelEncoder()
    Xinput[:,1] = label_encode_country.fit_transform(Xinput[:,1])
    Xinput[:,2] = label_encode_gender.fit_transform(Xinput[:,2])
    onehotencoding = OneHotEncoder(categorical_features=[1])

    Xinput = onehotencoding.fit_transform(Xinput).toarray()
    Xinput = Xinput[:,1:]#Dummy variable trap avoiding
    return Xinput,Youtput


def train_test_splitting(Xinput,Youtput):
    input_train,input_test,output_train,output_test=train_test_split(Xinput,Youtput,test_size=0.2,random_state=0)
    return input_train,input_test,output_train,output_test

def standardize_input(xtrain,xtest,ytrain,ytest):
    sc_obj = StandardScaler()
    xtrain = sc_obj.fit_transform(xtrain)
    xtest = sc_obj.transform(xtest)
    return xtrain,xtest,ytrain,ytest



def neural_network(input_train,output_train,output_test):

    ''' 
        NOt using k-cross-validation for evaluating 
        artifitial neural network
    '''
    #Starting/Initializing ANN
    classifier = Sequential()

    #Adding first layer that is INPUT Layer and first hidden layer
    classifier.add(Dense(15,activation="relu",kernel_initializer="uniform",input_dim=11))


    #Adding Second  Hidden Layeer
    classifier.add(Dense(15,activation="relu",kernel_initializer="uniform"))
    classifier.add(Dense(15,activation="relu",kernel_initializer="uniform"))


    #Adding Outpur Layer (Use activation simoid in case of buianry output 
    # otherwisee use softmax in case of nore then 2 output)
    classifier.add(Dense(1,activation="sigmoid",kernel_initializer="uniform"))

    #Start compiling network by using backpropagation on cost function
    classifier.compile("adam",loss="binary_crossentropy",metrics=["accuracy"])


    #Fit the model in to training set
    classifier.fit(x=input_train,y=output_train,batch_size=10,epochs=100)

    #Predicting output from testing data
    output_pred=classifier.predict(input_test)

    output_pred = (output_pred > 0.5)
    return output_pred

def build_trivial_classifier(init_mode,learning_rate):
    '''
        Using K cross Validation to evaluating artifitial
        neural network.Here we are creating KerasClassifier 
        object to pass as a argument to scikitlearn 
        cross_val_score function
    '''
    #Starting/Initializing ANN
    trivial_classifier = Sequential()

    #Adding first layer that is INPUT Layer and first hidden layer
    trivial_classifier.add(Dense(120,activation="relu",kernel_initializer=init_mode,input_dim=11))
    trivial_classifier.add(Dropout(0.2))
    trivial_classifier.add(BatchNormalization())

    #Adding Second  Hidden Layeer
    trivial_classifier.add(Dense(60,activation="relu",kernel_initializer=init_mode))
    trivial_classifier.add(Dropout(0.2))
    trivial_classifier.add(BatchNormalization())
    #trivial_classifier.add(Dense(30,activation="relu",kernel_initializer = init_mode))
    #trivial_classifier.add(BatchNormalization())

    #Adding Outpur Layer (Use activation simoid in case of buianry output 
    # otherwisee use softmax in case of nore then 2 output)
    trivial_classifier.add(Dense(1,activation="sigmoid",kernel_initializer=init_mode))

    #Start compiling network by using backpropagation on cost function
    optimizer=optimizers.adam(lr=learning_rate)
    trivial_classifier.compile(optimizer,loss="binary_crossentropy",metrics=["accuracy"])

    return trivial_classifier

def build_keras_classifir_from_trivial(trivial_classifier_function):
    keras_classifier = KerasClassifier(build_fn = trivial_classifier_function,\
                                      batch_size = 32,epochs = 100)
    return keras_classifier

def evaluating_cross_validation(keras_classifier,X_train,Y_train):
    accuracy_array = cross_val_score(estimator=keras_classifier,\
                                     X=X_train,y=Y_train,cv=10,n_jobs=-1)
    print(accuracy_array)
    return accuracy_array.mean(),accuracy_array.std()

def create_gridserach(hyper_params,keras_classifier,train,test):
    grid_obj = GridSearchCV(estimator=keras_classifier,\
                          param_grid = hyper_params,\
                          scoring = 'accuracy',\
                          cv = 10)
    grid_obj = grid_obj.fit(train,test)
    print(grid_obj.best_params_)
    print(grid_obj.best_score_)
    means = grid_obj.cv_results_['mean_test_score']
    stds = grid_obj.cv_results_['std_test_score']
    params = grid_obj.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    

def calculating_accuracy(output_pred,output_test,input_test):
    #Calulating accuracy from confusion matrix
    matric_table = confusion_matrix(output_test,output_pred)
    accuracy_test_set = (matric_table[0,...][0] +\
    matric_table[1,...][1])/input_test.shape[0]
    print(f"ACCURACY ON TEST SET : {accuracy_test_set}")

if __name__ == "__main__":
    PATH = "Artificial_Neural_Networks/Churn_Modelling.csv"
    Xinput,Youtput = preprocessing_input_output(PATH)
    Xinput,Youtput = categorical_to_numeric(Xinput,Youtput)
    xtrain,xtest,ytrain,ytest = train_test_splitting(Xinput,Youtput)
    input_train,input_test,output_train,output_test = standardize_input(xtrain,xtest,ytrain,ytest)
    #init_mode = ['uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #"batch_size":[32],"epochs":[100],"optimizer":["adam"]
    init_mode = ['normal']
    learning_rate = [0.005]
    HYPER_PARAMS = {"init_mode":init_mode,\
                    "learning_rate":learning_rate}
    #Using Cross Validation
    #classifier_object = build_trivial_classifier()
    keras_classifier = build_keras_classifir_from_trivial(build_trivial_classifier)
    create_gridserach(HYPER_PARAMS,keras_classifier,input_train,output_train)
    #mean , std = evaluating_cross_validation(keras_classifier,input_train,output_train)
    #print(mean,std)
    #output_pred = neural_network(input_train,output_train,output_test)
    #print("Calculating Accuracy in Unseen data set=>=>=>=>")
    #print("........................................")
    #calculating_accuracy(output_pred,output_test,input_test)
