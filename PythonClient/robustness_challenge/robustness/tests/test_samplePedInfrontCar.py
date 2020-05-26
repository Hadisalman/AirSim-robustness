import asyncio
from robustness import airsim
from IPython import embed
import copy
import numpy as np
import time
from scipy.spatial.transform import Rotation

client = airsim.CarClient()
client.confirmConnection()

def randomSample(value_range):
    return (value_range[1] - value_range[0])*np.random.random() + value_range[0]

def polar_to_cartesian(r, theta):
    '''
    2D polar coordiantes to cartesian
    '''
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return airsim.Vector3r(x, y)


def body_to_world(t_body, q_o_b):
    rotation = Rotation.from_quat([q_o_b.x_val, q_o_b.y_val, q_o_b.z_val, q_o_b.w_val])
    t_body_np = [t_body.x_val, t_body.y_val, t_body.z_val]
    t_world_np = rotation.apply(t_body_np)
    t_world = airsim.Vector3r(t_world_np[0], t_world_np[1], t_world_np[2])
    return t_world


def randomPedPoseInfrontCar(car_pose_world, z_ped, r_range=(4, 7), cam_fov=30.0):
    r = randomSample(r_range)
    alpha = cam_fov/180.0*np.pi/2.0
    theta_range = [-alpha, alpha]
    theta = randomSample(theta_range)

    rel_ped_position_body = polar_to_cartesian(r, theta)

    rel_ped_position_world = body_to_world(rel_ped_position_body, car_pose_world.orientation)

    ped_position_world = car_pose_world.position + rel_ped_position_world

    ped_position_world.z_val = z_ped # overwrite the z_val of the pedestrian

    yaw = randomSample([-np.pi/2, np.pi/2])
    rotation_ped = Rotation.from_euler('ZYX', [yaw, 0, 0])
    q = rotation_ped.as_quat()
    ped_pose_world = airsim.Pose(ped_position_world, airsim.Quaternionr(q[0], q[1], q[2], q[3]))

    return ped_pose_world

def isPedInfrontCar(ped_pose_world, car_pose_world, r_range=(4, 7), cam_fov=30.0):
    r = randomSample(r_range)
    alpha = cam_fov/180.0*np.pi/2.0
    theta_range = [-alpha, alpha]
    theta = randomSample(theta_range)

    rel_ped_position_body = polar_to_cartesian(r, theta)

    rel_ped_position_world = body_to_world(rel_ped_position_body, car_pose_world.orientation)

    ped_position_world = car_pose_world.position + rel_ped_position_world

    ped_position_world.z_val = z_ped # overwrite the z_val of the pedestrian

    yaw = randomSample([-np.pi/2, np.pi/2])
    rotation_ped = Rotation.from_euler('ZYX', [yaw, 0, 0])
    q = rotation_ped.as_quat()
    ped_pose_world = airsim.Pose(ped_position_world, airsim.Quaternionr(q[0], q[1], q[2], q[3]))

    return ped_pose_world



while True:
    ped_pose = client.simGetObjectPose('Adv_Ped2')
    vehicle_pose = client.simGetVehiclePose()
    sampled_pose = randomPedPoseInfrontCar(vehicle_pose, ped_pose.position.z_val)
    client.simSetObjectPose('Adv_Ped2', sampled_pose)
    time.sleep(0.4)