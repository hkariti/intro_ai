import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

T = np.linspace(0.01, 5, 100)

X_len = X.shape[0]
T_len = T.shape[0]
# Convert T to -1/T
T_exp = -1/(T.reshape([1, T_len]))
# Get the normalizer constant
alpha = np.min(X)
# Convert X to X**(-1/T)
X_exp = np.power(X.reshape([X_len, 1])/alpha, T_exp)
# Calcualte sum of all X**(-1/T)
X_sum = X_exp.sum(axis=0)
# Now we get P by definition. Transpose is to be compatible with the below for loop
P = (X_exp / X_sum).transpose()

print(P)

for i in range(len(X)):
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
