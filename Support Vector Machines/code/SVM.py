__package__ = 'Support_Vector_Machines'
__name__ = 'SVM'

import numpy as np

class SVM_Classifier():
  def __init__(self,learning_rate,no_of_iterations,lambda_parameter):
    self.learning_rate=learning_rate
    self.no_of_iterations=no_of_iterations
    self.lambda_parameter=lambda_parameter
  
  def fit(self,X,Y):

    self.m, self.n = X.shape

    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    #implementing gradient descent
    for i in range(self.no_of_iterations):
      self.update_weights()


  def update_weights(self):

    #label encodeing
    y_label = np.where(self.Y <=0, -1, 1)


    #gradients (dw,db)
    for index, x_i in enumerate(self.X):

      condition = y_label[index] * ( np.dot( x_i, self.m ) - self.b ) >= 1

      if(condition == True):

        dw = 2 * self.lambda_parameter * self.w
        db = 0
      
      else:

        dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
        db = y_label[index]

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db
  
  def predict(self,X):

    output = self.dot(X, self.w) - self.b

    #to convert any negetive value to -1 and positive value to +1
    predicted_labels = np.sign(output)

    #to convert -1 to 0 
    y_cap = np.where(predicted_labels <= -1, 0, 1)

    return y_cap
