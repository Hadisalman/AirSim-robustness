import os
import threading
import time

import airsim
import cv2
import numpy as np
from IPython import embed

############################################################
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
embed()
############################################################
car_controls = airsim.CarControls()
car_controls.is_manual_gear = True;

# get state of the car
car_state = client.getCarState()
print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

# go forward
car_controls.manual_gear = 1
car_controls.throttle = 0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go Forward")
time.sleep(3)   # let car drive a bit

# go reverse
car_controls.manual_gear = -1
car_controls.throttle = -0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go reverse")
time.sleep(3)   # let car drive a bit

# Go forward and turn right
car_controls.manual_gear = 1
car_controls.throttle = 1
client.setCarControls(car_controls)
print("Go Forward")
time.sleep(3.2)   
car_controls.throttle = 0.5
car_controls.steering = 1
client.setCarControls(car_controls)
print("Turn Right")
time.sleep(1.35)


# go forward
car_controls.throttle = 0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go Forward")
time.sleep(3)   


# apply brakes
car_controls.brake = 1
client.setCarControls(car_controls)
print("Apply brakes")
time.sleep(3)   
car_controls.brake = 0 #remove brake
client.reset()

# go forward
car_controls.throttle = 0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go Forward")
time.sleep(3)   # let car drive a bit
# apply brakes
car_controls.brake = 1
client.setCarControls(car_controls)
print("Apply brakes")
time.sleep(3)   

embed()
############################################################
## Control objects in the scene
def get_trajectory(start_pose, end_pose, num_waypoints=10):
    inc_vec = (end_pose.position - start_pose.position)/(num_waypoints - 1)
    traj = []
    traj.append(start_pose)
    for _ in range(num_waypoints - 2):
        traj.append(airsim.Pose())
        traj[-1].orientation = traj[-2].orientation
        traj[-1].position = traj[-2].position + inc_vec
    traj.append(end_pose)
    return traj

adv_objects = [
    'Hadi',
    'Adv_Fence',
    'Adv_Hedge',
    'Adv_Car',
    'Adv_House',
    'Adv_Tree'
    ]

scene_objs = client.simListSceneObjects()
for obj in adv_objects:
    print('{} exists? {}'.format(obj, obj in scene_objs))


end_pose = airsim.Pose()
delta = airsim.Vector3r(0, -4, 0)
for obj in adv_objects:
    pose = client.simGetObjectPose(obj)
    end_pose.orientation = pose.orientation
    end_pose.position = pose.position
    end_pose.position += delta
    traj = get_trajectory(pose, end_pose, 100)
    for way_point in traj:
        client.simSetObjectPose(obj, way_point)
        time.sleep(0.01)

    pose = client.simGetObjectPose(obj)
    end_pose.position = pose.position
    end_pose.position -= delta
    traj = get_trajectory(pose, end_pose, 100)
    for way_point in traj:
        client.simSetObjectPose(obj, way_point)
        time.sleep(0.01)
embed()
############################################################
## Control the Weather
client.simEnableWeather(True)
attributes = ['Rain', 'Roadwetness', 'Snow', 'RoadSnow', 'MapleLeaf', 'Dust', 'Fog']

for att in attributes:
    att = airsim.WeatherParameter.__dict__[att]
    client.simSetWeatherParameter(att, 1)
    time.sleep(3)
    client.simSetWeatherParameter(att, 0.0)
client.simEnableWeather(False)


###################################################
# Reset
client.reset()
client.enableApiControl(False)
