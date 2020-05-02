import asyncio
import airsim
from IPython import embed
import copy
import numpy as np
import time

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)


controls = airsim.CarControls()

client.setCarSpeed(5)

controls.throttle = 0
controls.brake = 1
controls.steering = 0
controls.handbrake = False
client.setCarControls(controls)

embed()

car_state = client.getCarState()
print("Speed %f, Gear %f" % (car_state.speed, car_state.gear))
