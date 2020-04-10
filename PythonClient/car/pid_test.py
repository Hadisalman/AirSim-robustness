import airsim
import matplotlib.pyplot as plt
import time

c = airsim.CarClient()
c.enableApiControl(True)

inputte = []
output = []

for i in range(12000):
    if i < 2500:
        speed = 5.0
    elif i > 2500 and i < 7500:
        speed = 25.0
    elif i > 7500 and i < 10000:
        speed = 10.0
    else:
        speed = 0.0

    c.setCarSpeed(speed)

    state = c.getCarState().speed
    inputte.append(speed)
    output.append(state)
    time.sleep(0.005)

t = range(12000)
plt.plot(t, inputte, 'g')
plt.plot(t, output, 'r')
plt.show()