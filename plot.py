import matplotlib.pyplot as plt
import numpy as np

# ε = 0.005
x = np.array([0,1,2,3])
models = ['ConViT','ConvNeXT','DeiT','ResNet']
apgd_ce = [0.39625, 0.225, 0.32625,0.1675 ]
apgd_t = [0.4125, 0.25625,0.3125, 0.19375]
fab = [0.805, 0.8575, 0.825, 0.80375]
square = [0.60625, 0.61625,0.565, 0.5275]
plt.xticks(x, models)

plt.plot(x, apgd_ce, label='APGD-CE')
plt.plot(x, apgd_t, label='APGD-T')
plt.plot(x, fab, label='FAB-T')
plt.plot(x, square, label='Square')

plt.legend()

plt.xlabel('Models')
plt.ylabel('Robustness Accuracy')
plt.title('Model VS Robustness Accuracy Against Adversarial Attacks (ε = 0.005)')


plt.savefig('ε = 0.005.png', dpi=300, bbox_inches='tight')
plt.show() 

# ε = 0.031
apgd_ce = [, , ,0.00375]
apgd_t = [, ,, 0.0025]
square = [, , , 0.2125]
plt.xticks(x, models)

plt.plot(x, apgd_ce, label='APGD-CE')
plt.plot(x, apgd_t, label='APGD-T')
plt.plot(x, square, label='Square')

plt.legend()

plt.xlabel('Models')
plt.ylabel('Robustness Accuracy')
plt.title('Model VS Robustness Accuracy Against Adversarial Attacks (ε = 0.03)')


plt.savefig('ε = 0.03.png', dpi=300, bbox_inches='tight')
plt.show() 

