#https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
#pip install numpy
#pip install pandas
#pip install sklearn
#pip install scikit-learn
#pip install keras
#pip install tensorflow
#pip install matplotlib

#Dependencies
import numpy as np
import pandas as pd
#dataset import
dataset = pd.read_csv('data/train.csv') #You need to change #directory accordingly
view = dataset.head(10) #Return 10 rows of data
print(view)
print('--------------------------')
print('linhas , colunas')
print(dataset.shape)
print('--------------------------')
print()
print()


#Changing pandas dataframe to numpy array
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

print('--------------------------')
print(X)
print('--------------------------')
print(y)
print('--------------------------')




#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

print(X)
print('shape X: ') 
print(X.shape)



# encode the classes convert integer classes into binary values - cada classe vira uma coluna
#1- 1 0 0
#2- 0 1 0
#3- 0 0 1

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

print(y)
print('shape y: ') 
print(y.shape)


#Split Training data will have 90% samples and test data will have 10% samples.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)




#Building Neural Network
#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu')) #quantidade de colunas da entrada do treinamento
model.add(Dense(12, activation='relu')) #duas camadas escondidas  de 16 e 12 neuronios
model.add(Dense(4, activation='softmax'))#quantidade de saídas da entrada do treinamento




#specify the loss function and the optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
#Training model
history = model.fit(X_train, y_train, epochs=100, batch_size=64)



#performance on test data:
#predict on test data using a simple method of keras, model.predict().
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
    


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
'''

#=======================================

#check the accuracies after every epoch
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)

#performance on test data:
#predict on test data using a simple method of keras, model.predict().
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
    


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
#https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#https://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html
#http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
#https://notebook.community/lukemans/Hello-world/tf_kdd99
#=====================================
history_dict = history.history
print(history_dict.keys())
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('g1.jpg')
plt.show()
#https://stackoverflow.com/questions/39883331/plotting-learning-curve-in-keras-gives-keyerror-val-acc