from utils import *
import matplotlib.pyplot as plt

# a)
BCs = [
    BC(),
    BC(BCType.NEUMANN, 0)
]

Ms = np.geomspace(20, 500, 10, dtype=int)
errors = find_errors_np(Ms, f, BCs)
header = "M\t" + "\t".join(error_functions.keys())
np.savetxt("errors_np.txt", errors.T, header=header)

