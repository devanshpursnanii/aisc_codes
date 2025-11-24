import numpy as np

class bpn:
  def __init__(self, target=1, a=0.5):
    self.i = 3
    self.j = 2
    self.k = 1

    # Initialize weights and biases as numpy arrays for consistent operations
    self.v = np.array([[0.2, 0.3],[0.4, 0.2],[0.5, 0.3]]) # Shape (3, 2)
    self.w = np.array([[0.4], [0.4]])                     # Shape (2, 1)

    self.v0 = np.array([0.2, 0.6])                        # Shape (2,)
    self.w0 = np.array([0.4])                             # Shape (1,)

    self.x = np.array([0.2, 0.3, -0.4])                   # Shape (3,)
    self.z = np.zeros(self.j)                             # Shape (2,)
    self.dj = np.zeros(self.j)                            # Shape (2,)

    self.target = target
    self.y = 0
    self.e = 0

    self.a = a


  def activation(self, x):
    return 1 / (1 + np.exp(-x))


  def feedforward(self):
    for q in range(self.j):
      self.z[q] = self.activation(np.dot(self.x, self.v[:, q]) + self.v0[q])


  def output(self):
    # Corrected: np.dot(self.z, self.w) will result in a (1,) array, so take [0] to get scalar
    self.y = self.activation(np.dot(self.z, self.w)[0] + self.w0[0])


  def err(self):
    self.e = self.target - self.y


  def delk(self):
    self.dk = self.e * self.y * (1 - self.y)


  def delj(self):
    for p in range(self.j):
      # Corrected: Access element from self.w (shape (2,1)) using self.w[p, 0]
      self.dj[p] = self.dk * self.w[p, 0] * self.z[p] * (1 - self.z[p])


  def back1(self):
    # update output weights
    self.wn = self.w + self.a * self.dk * self.z.reshape(self.j, 1)
    self.w0n = self.w0 + self.a * self.dk


  def back2(self):
    # update hidden weights
    self.vn = self.v + self.a * np.outer(self.x, self.dj)
    self.v0n = self.v0 + self.a * self.dj


  def train(self):
    self.feedforward()
    self.output()
    self.err()
    self.delk()
    self.delj()
    self.back1()
    self.back2()

    # assign updated weights back to network
    self.w = self.wn
    self.w0 = self.w0n
    self.v = self.vn
    self.v0 = self.v0n


  def final_weights(self):
    print("new ws: ", self.w)
    print("\nnew w0s: ", self.w0)
    print("\nnew vs:", self.v)
    print("\nnew v0s:", self.v0)


b=bpn()
b.train()
b.final_weights()