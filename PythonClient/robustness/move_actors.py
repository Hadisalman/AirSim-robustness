import airsim
import time
import numpy as np

def ped_dummy_move(demo_instance, obj):
    def _get_trajectory(start_pose, goal_pose, num_waypoints=-1):
        if num_waypoints == -1:
            num_waypoints = int(40*np.linalg.norm(delta.to_numpy_array(), 2))
        inc_vec = (goal_pose.position - start_pose.position)/(num_waypoints - 1)
        traj = []
        traj.append(start_pose)
        for _ in range(num_waypoints):
            traj.append(airsim.Pose())
            traj[-1].orientation = traj[-2].orientation
            traj[-1].position = traj[-2].position + inc_vec
        traj.append(goal_pose)
        return traj

    def _wrap_angle(angle):
        if angle > np.pi:
            return angle - 2*np.pi
        elif angle <= -np.pi:
            return angle + 2*np.pi
        else:
            return angle

    def _reorient_toward_goal(start_pose, goal_pose):
        pitch, roll, yaw = airsim.to_eularian_angles(start_pose.orientation)
        delta_v = goal_pose.position - start_pose.position
        yaw_goal = _wrap_angle(np.arctan2(delta_v.x_val, delta_v.y_val) + np.pi/2)
        while np.abs(yaw - yaw_goal) >  0.1:
            if np.abs(yaw - yaw_goal) <= np.pi:
                yaw = _wrap_angle(yaw + 0.02 * np.sign(yaw_goal - yaw))
            else:
                yaw = _wrap_angle(yaw - 0.02 * np.sign(yaw_goal - yaw))
            start_pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
            demo_instance.client_ped.simSetObjectPose(obj+'_walking', start_pose)
        goal_pose.orientation = start_pose.orientation
        
    def _swap_animation():
        pose_walking = demo_instance.client_ped.simGetObjectPose(obj+'_walking')
        pose = demo_instance.client_ped.simGetObjectPose(obj)
        demo_instance.client_ped.simSetObjectPose(obj, pose_walking)
        demo_instance.client_ped.simSetObjectPose(obj+'_walking', pose)

    pose = demo_instance.client_ped.simGetObjectPose(obj)

    goal_pose = airsim.Pose()
    delta=airsim.Vector3r(0, -30)
    goal_pose.position = pose.position
    goal_pose.position += delta

    _swap_animation()
    _reorient_toward_goal(pose, goal_pose)
    traj = _get_trajectory(pose, goal_pose)
    for way_point in traj:
        if not demo_instance.is_ped_thread_active:
            break
        demo_instance.client_ped.simSetObjectPose(obj+'_walking', way_point)
        time.sleep(0.01)
    _swap_animation()

def car_dummy_move(demo_instance):
    car_controls = airsim.CarControls()

    # get state of the car
    car_state = demo_instance.client_car.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    # go forward
    car_controls.throttle = 0.5
    car_controls.steering = 0
    demo_instance.client_car.setCarControls(car_controls)
    print("Go Forward")
    time.sleep(4)   # let car drive a bit
    if not demo_instance.is_car_thread_active:
        return

    # go reverse
    # car_controls.throttle = -0.5
    # car_controls.is_manual_gear = True;
    # car_controls.manual_gear = -1
    # car_controls.steering = 0
    # demo_instance.client_car.setCarControls(car_controls)
    # print("Go reverse")
    # time.sleep(3)   # let car drive a bit
    # if not demo_instance.is_car_thread_active:
    #     return
    # car_controls.is_manual_gear = False; # change back gear to auto
    # car_controls.manual_gear = 0  

    # # Go forward
    # car_controls.throttle = 1
    # demo_instance.client_car.setCarControls(car_controls)
    # print("Go Forward")
    # time.sleep(3.5)   
    # if not demo_instance.is_car_thread_active:
    #     return
    # car_controls.throttle = 0.5
    # car_controls.steering = 1
    # demo_instance.client_car.setCarControls(car_controls)
    # print("Turn Right")
    # time.sleep(1.4)
    # if not demo_instance.is_car_thread_active:
    #     return


    # car_controls.throttle = 0.5
    # car_controls.steering = 0
    # demo_instance.client_car.setCarControls(car_controls)
    # print("Go Forward")
    # time.sleep(3)   
    # if not demo_instance.is_car_thread_active:
    #     return


    # # apply brakes
    car_controls.brake = 1
    demo_instance.client_car.setCarControls(car_controls)
    print("Apply brakes")
    time.sleep(3)   
    if not demo_instance.is_car_thread_active:
        return
    car_controls.brake = 0 #remove brake
    # demo_instance.client_car.reset()

    # # go forward
    # car_controls.throttle = 0.5
    # car_controls.steering = 0
    # demo_instance.client_car.setCarControls(car_controls)
    # print("Go Forward")
    # time.sleep(3)   # let car drive a bit
    # if not demo_instance.is_car_thread_active:
    #     return
    # # apply brakes
    # car_controls.brake = 1
    # demo_instance.client_car.setCarControls(car_controls)
    # print("Apply brakes")
    # time.sleep(3)   
