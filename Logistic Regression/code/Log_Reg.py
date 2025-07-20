__package__ = 'Logistic_Regression'
__name__ = 'Log_Reg'

import numpy as np

class Logistic_Regression():
  def __init__(self,learning_rate, no_of_iterations):

    #learning rate = how fast the model learns the data
    self.learning_rate=learning_rate

    #no.of iterations = how many times the model goes through the data
    self.no_of_iterations=no_of_iterations

  def fit(self,X,Y):

    #number of data points and input features = m,n
    self.m, self.n = X.shape

    #initializing weight and bias values with 0
    self.w = np.zeros(self.n)
    self.b = 0

    self.X = X
    self.Y = Y

    # implementing Gradint Desent for model optimization

    for i in range(self.no_of_iterations):
      self.update_weights()

  def update_weights(self):

    #Y_cap formula (Sigmoid Function)
    Y_cap = 1 / ( 1 + np.exp( - ( self.X.dot( self.w ) + self.b )))

    #derivatives

    dw = (1/self.m)*np.dot(self.X.T, (Y_cap - self.Y))
    db = (1/self.m)*np.sum(Y_cap - self.Y)

    #updating the weights and bias using gradient desent
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db

  # Sigmoid Equation & Decision Boundries
  def predict(self,X):

    Y_pred =  1 / ( 1 + np.exp( - ( X.dot( self.w ) + self.b )))
    Y_pred = np.where( Y_pred > 0.5, 1, 0)
    return Y_pred