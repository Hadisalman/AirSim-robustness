import asyncio
from robustness import airsim
from IPython import embed
import copy
import numpy as np
import time

client = airsim.CarClient()
client.confirmConnection()

player_start_assets = client.simListSceneObjects("PlayerState.*")

# Drive using the keyboard to see the different values of these states
while True:
    vehicle_pose = client.simGetVehiclePose()
    euler_car = list(airsim.to_eularian_angles(vehicle_pose.orientation))
    yaw = euler_car[2]*180/np.pi
    print(f'VehiclePose: X: {vehicle_pose.position.x_val:.3f} -- Y: {vehicle_pose.position.y_val:.3f} -- Yaw: {yaw:.3f}')

    car_state = client.getCarState()
    euler_car = list(airsim.to_eularian_angles(car_state.kinematics_estimated.orientation))
    yaw = euler_car[2]*180/np.pi
    print(f'CarState:    X: {car_state.kinematics_estimated.position.x_val:.3f} -- Y: {car_state.kinematics_estimated.position.y_val:.3f} -- Yaw: {yaw:.3f}')

    player_start = client.simGetObjectPose(player_start_assets[0])
    euler_car = list(airsim.to_eularian_angles(player_start.orientation))
    yaw = euler_car[2]*180/np.pi
    print(f'PlayerStart: X: {player_start.position.x_val:.3f} -- Y: {player_start.position.y_val:.3f} -- Yaw: {yaw:.3f}')

    ped_pose = client.simGetObjectPose('Adv_Ped2')
    euler_car = list(airsim.to_eularian_angles(ped_pose.orientation))
    yaw = euler_car[2]*180/np.pi
    print(f'Pedestrian : X: {ped_pose.position.x_val:.3f} -- Y: {ped_pose.position.y_val:.3f} -- Yaw: {yaw:.3f}')