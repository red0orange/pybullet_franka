import numpy as np
import matplotlib.pyplot as plt


a = 3
trans = np.linspace(0, 2, 200)
trans_score = np.exp(a * trans) - 1
# trans_score[100:] = np.exp(a * trans[100])

b = 2
angle = np.linspace(0, np.pi / 2, 200)
angle_score = np.exp(b * angle) - 1
# angle_score[100:] = np.exp(b * angle[100]) - 1

plt.plot(trans, trans_score, label='trans')
plt.plot(angle, angle_score, label='angle')
plt.scatter([trans[20]], [trans_score[20]], c='r', marker='o')
plt.scatter([angle[66]], [angle_score[66]], c='r', marker='o')
plt.xlim(0, 1)
plt.ylim(0, 30)

plt.legend()
plt.show()