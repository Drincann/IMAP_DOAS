import pickle
import numpy as np
import matplotlib.pyplot as plt

SF = pickle.load(open('SanJuanMethane_SF.p', 'rb'))

plt.imshow(SF)
plt.colorbar()
plt.show()


