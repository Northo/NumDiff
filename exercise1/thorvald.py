from utils import *
import matplotlib.pyplot as plt

# a)
BCs = [
    BC(),
    BC(BCType.NEUMANN, 0)
]

Ms = np.geomspace(20, 500, 10, dtype=int)
errors = find_errors(Ms, f, BCs)
write_errors_file("errors.txt", Ms, errors)

