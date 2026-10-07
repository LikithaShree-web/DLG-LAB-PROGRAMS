from keras.models import Sequential
from keras.layers import Dense,Input 
from keras.optimizers import SGD
from sklearn.datasets import make_moons

X,y=make_moons(n_samples=400, noise=0.2, random_state=1)
y=y.reshape(-1,1).astype(float)
X=(X-X.mean(0))/X.std(0)

model=Sequential([Input(shape=(2,)),
                  Dense(16,activation='relu'),
                    Dense(16,activation='relu'),
                    Dense(1,activation='sigmoid')
                  ])
model.compile(loss='binary_crossentropy',optimizer=SGD(learning_rate=0.5),metrics=['accuracy'])
model.fit(X,y,epochs=200,batch_size=len(X),verbose=0)

loss_batch, accuracy_batch = model.evaluate(X, y, verbose=0)
print("Batch GD accuracy:", accuracy_batch)

model.fit(X,y,epochs=200,batch_size=16,verbose=0)

loss_sgd, accuracy_sgd = model.evaluate(X, y, verbose=0)
print("SGD accuracy:", accuracy_sgd)