import numpy as np

class ksofmn:

  def __init__(self):
    self.w=np.array([[1, 0.9, 0.7, 0.5, 0.3], [0.3, 0.5, 0.7, 0.9, 1]])
    self.x=np.array([0, 0.5, 1, 1.5, 0])
    self.a=0.25
    self.y=2
    self.d=np.zeros(self.y)
    self.winners=[]

  def winner(self):
    self.min=0
    for p in range(self.y):
      self.d[p] = np.sum((self.x - self.w[p])**2)

    self.min = np.argmin(self.d)

    self.winners.append(self.min)
    
  def update(self):
    self.w[self.min] += self.a * (self.x - self.w[self.min])
    
  def train(self):
    self.winner()
    self.update()
    print("winner is class: ", self.winners[0]+1)
