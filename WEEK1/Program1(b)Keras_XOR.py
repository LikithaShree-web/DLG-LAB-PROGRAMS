import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.optimizers import SGD

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

model=Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(learning_rate=0.1),
    metrics=['accuracy']
)
model.fit(X,y,epochs=1000,verbose=0)

_,accuracy=model.evaluate(X,y)

print(f"Model Accuracy:{accuracy * 100:.2f}%")

predictions=model.predict(X)
print("Predictions: ")
print(np.round(predictions))