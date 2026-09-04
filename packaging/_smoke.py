import sys, numpy as np
sys.path.insert(0, '.')
from aemtn_b4 import pauli_features_from_counts
counts = np.zeros((9, 64), dtype='int64')
counts[:, 42] = 1024
f = pauli_features_from_counts(counts, shots=1024)
print("local6:", np.round(f, 4))
