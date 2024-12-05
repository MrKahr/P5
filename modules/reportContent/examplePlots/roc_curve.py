from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay

# ROC Curve used in Theory

# g1
y_true1 = []
y_pred1 = []

for i in range(0,90):
    y_true1.append(True)
    y_pred1.append(True)
for i in range(0,10):
    y_true1.append(False)
    y_pred1.append(True)

# g2
y_true2 = []
y_pred2 = []

for i in range(0,80):
    y_true2.append(True)
    y_pred2.append(True)
for i in range(0,10):
    y_true2.append(False)
    y_pred2.append(False)
for i in range(0,10):
    y_true2.append(True)
    y_pred2.append(False)


g1_disp = RocCurveDisplay.from_predictions(y_true=y_true1, y_pred=y_pred1, name="g1", pos_label=True)
ax = plt.gca()
g2_disp = RocCurveDisplay.from_predictions(y_true=y_true2, y_pred=y_pred2, ax=ax, name="g2", pos_label=True)

plt.show()