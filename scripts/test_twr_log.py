import numpy as np

# twr = np.arange(0.11, 0.03, -0.02)
twr_log = np.arange(1.9, -3.1, -0.1)
twr = np.exp(twr_log)
delta = np.hstack(([0.], (twr[1:] - twr[:-1])))

print(np.size(twr), '\n')
print(np.around(twr, 4), '\n')
print(np.around(delta, 4), '\n')
print(np.around(twr_log, 4), '\n')

for i in range(np.size(twr)):
    print(f"\t{twr[i]:.4f}\t{delta[i]:.4f}\t{twr_log[i]:.4f}")