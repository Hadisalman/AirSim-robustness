import copy
import json
import threading

import numpy as np
import torch

from robustness import airsim
from .sim_object import SimObject

class AdversarialObjects(SimObject):
    def __init__(self, name='3DAdversary', car=None, **kwargs):
        super().__init__(name)

        assert 'resolution_coord_descent' in kwargs and 'num_iter' in kwargs and 'adv_config_path' in kwargs

        self.ped_detection_callback = car.detection.ped_detection_callback
        # TODO: un-hardcode this.
        self.ped_object_name = 'Adv_Ped2'

        self.thread = threading.Thread(target=self.coordinate_ascent_object_attack, args=(kwargs['resolution_coord_descent'], kwargs['num_iter']))
        self.is_thread_active = False

        self.scene_objs = self.client.simListSceneObjects()

        self.adv_objects = [
            'Adv_House',
            'Adv_Fence',
            'Adv_Hedge',
            'Adv_Car',
            'Adv_Tree'
            ]
        
        self.adv_config_path = kwargs['adv_config_path']
        for obj in self.adv_objects:
            print('{} exists? {}'.format(obj, obj in self.scene_objs))

        for obj in ['BoundLowerLeft', 'BoundUpperRight']:
            print('{} exists? {}'.format(obj, obj in self.scene_objs))

        self.BoundLowerLeft = self.client.simGetObjectPose('BoundLowerLeft')
        self.BoundUpperRight = self.client.simGetObjectPose('BoundUpperRight')

        self.x_range_adv_objects_bounds = (self.BoundLowerLeft.position.x_val, self.BoundUpperRight.position.x_val)
        self.y_range_adv_objects_bounds = (self.BoundLowerLeft.position.y_val, self.BoundUpperRight.position.y_val)

    def dump_env_config_to_json(self, path):
        def _populate_pose_dic(pose_dic, pose):
            pose_dic['X'] = pose.position.x_val
            pose_dic['Y'] = pose.position.y_val
            pose_dic['Z'] = pose.position.z_val
            euler_angles = airsim.to_eularian_angles(pose.orientation)
            pose_dic['Pitch'] = euler_angles[0]
            pose_dic['Roll'] = euler_angles[1]
            pose_dic['Yaw'] = euler_angles[2]

        with open(path, 'w') as f:
            output = {}
            output['Vehicle'] = {}
            pose = self.client.simGetVehiclePose()
            _populate_pose_dic(output['Vehicle'], pose)

            output[self.ped_object_name] = {}
            pose = self.client.simGetObjectPose(self.ped_object_name)
            _populate_pose_dic(output[self.ped_object_name], pose)

            for obj in self.adv_objects:
                output[obj] = {}
                pose = self.client.simGetObjectPose(obj)
                _populate_pose_dic(output[obj], pose)
            # print(output)
            json.dump(output, f, indent=2, sort_keys=False)
            
    def update_env_from_config(self, path):
        with open(path, 'r') as f:
            dic = json.load(f)
            for obj_name, obj_pose in dic.items():
                pose = airsim.Pose(airsim.Vector3r(obj_pose['X'], obj_pose['Y'], obj_pose['Z']), 
                            airsim.to_quaternion(obj_pose['Pitch'], obj_pose['Roll'], obj_pose['Yaw']))
                if obj_name == 'Vehicle':
                    self.client.simSetVehiclePose(pose, ignore_collison=True)
                else:
                    assert obj_name in self.scene_objs, 'Object {} is not found in the scene'.format(obj_name)
                    self.client.simSetObjectPose(obj_name, pose)
                print('-->[Updated the position of the {}]'.format(obj_name))

    def coordinate_ascent_object_attack(self, resolution=10, num_iter=1):
        x_range = np.linspace(self.x_range_adv_objects_bounds[0], self.x_range_adv_objects_bounds[1], resolution)
        y_range = np.linspace(self.y_range_adv_objects_bounds[0], self.y_range_adv_objects_bounds[1], resolution)
        xv, yv = np.meshgrid(x_range, y_range)

        self.adv_poses = []

        best_loss = -1
        for _ in range(num_iter):
            for obj in np.random.permutation(self.adv_objects).tolist():
                pose = self.client.simGetObjectPose(obj)
                best_pose = copy.deepcopy(pose)
                grid2d_poses_list = zip(xv.flatten(), yv.flatten())
                for grid2d_pose in grid2d_poses_list:
                    pose.position.x_val = grid2d_pose[0]
                    pose.position.y_val = grid2d_pose[1]
                    self.client.simSetObjectPose(obj, pose)
                    if not self.is_thread_active:
                        print('-->[Saving whatever coniguration is reached]')
                        self.dump_env_config_to_json(path=self.adv_config_path)
                        return
                    _, correct, loss = self.ped_detection_callback()
                    if loss > best_loss:
                        best_loss = loss
                        best_pose = copy.deepcopy(pose)
                print('Best loss so far {}'.format(best_loss.item()))

                self.client.simSetObjectPose(obj, best_pose)
        
            # dump results into a json file after each iteration
            self.dump_env_config_to_json(path=self.adv_config_path)

    def spsa_object_attack(self, resolution=10, num_iter=1):
        def calc_est_grad(func, x, y, rad, num_samples):
            B, *_ = x.shape
            Q = num_samples//2
            N = len(x.shape) - 1
            with torch.no_grad():
                # Q * B * C * H * W
                extender = [1]*N
                queries = x.repeat(Q, *extender)
                noise = torch.randn_like(queries)
                norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
                noise = noise / norm
                noise = torch.cat([-noise, noise])
                queries = torch.cat([queries, queries])
                y_shape = [1] * (len(y.shape) - 1)
                l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender) 
                grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
            return grad
        x_range = np.linspace(self.x_range_adv_objects_bounds[0], self.x_range_adv_objects_bounds[1], resolution)
        y_range = np.linspace(self.y_range_adv_objects_bounds[0], self.y_range_adv_objects_bounds[1], resolution)
        xv, yv = np.meshgrid(x_range, y_range)

        self.adv_poses = []

        best_loss = -1
        for _ in range(num_iter):
            for obj in np.random.permutation(self.adv_objects).tolist():
                pose = self.client.simGetObjectPose(obj)
                best_pose = copy.deepcopy(pose)
                grid2d_poses_list = zip(xv.flatten(), yv.flatten())
                for grid2d_pose in grid2d_poses_list:
                    pose.position.x_val = grid2d_pose[0]
                    pose.position.y_val = grid2d_pose[1]
                    self.client.simSetObjectPose(obj, pose)
                    if not self.is_thread_active:
                        print('[-->[Saving whatever coniguration is reached]')
                        self.dump_env_config_to_json(path=self.adv_config_path)
                        return
                    _, correct, loss = self.ped_detection_callback()
                    if loss > best_loss:
                        best_loss = loss
                        best_pose = copy.deepcopy(pose)
                print('Best loss so far {}'.format(best_loss.item()))

                self.client.simSetObjectPose(obj, best_pose)
        
            # dump results into a json file after each iteration
            self.dump_env_config_to_json(path=self.adv_config_path)

    def attack(self):
        if not self.is_thread_active:
            self.is_thread_active = True
            self.thread.start()
            print("-->[Started adv thread]")
