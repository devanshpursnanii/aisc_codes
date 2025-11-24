import numpy as np

class lvqn:

  def __init__(self):
    # weights (same shape)
    self.w = np.array([[1, 0.9, 0.7, 0.5, 0.3],
                       [0.3, 0.5, 0.7, 0.9, 1]])

    # single input sample (change manually when training)
    self.x = np.array([0, 0.5, 1, 1.5, 0])

    # learning rate
    self.a = 0.25

    # number of output neurons
    self.y = 2

    # distance array
    self.d = np.zeros(self.y)

    # LVQ class labels for neurons
    self.labels = np.array([1, 2])   # neuron 1 → class 1, neuron 2 → class 2

    # target class for current input (set manually)
    self.target = 1

    self.winners = []

  def winner(self):
    self.min = 0
    for p in range(self.y):
      self.d[p] = np.sum((self.x - self.w[p])**2)

    self.min = np.argmin(self.d)
    self.winners.append(self.min)

  def update(self):
    # LVQ update rule
    if self.labels[self.min] == self.target:
      # correct classification -> move TOWARD input
      self.w[self.min] += self.a * (self.x - self.w[self.min])
    else:
      # wrong classification -> move AWAY from input
      self.w[self.min] -= self.a * (self.x - self.w[self.min])

  def train(self):
    self.winner()
    self.update()
    print("winner neuron:", self.winners[0] + 1)
    print("updated weights:\n", self.w)
