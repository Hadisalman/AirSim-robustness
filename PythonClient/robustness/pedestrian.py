import numpy as np
import threading

from robustness import airsim
from .sim_object import SimObject

class Pedestrian(SimObject):
    def __init__(self, name='Adv_Ped2'):
        super().__init__(name)

        self.thread = threading.Thread(target=self.move_pedestrian)
        self.is_thread_active = False

        self.scene_objs = self.client.simListSceneObjects()
        print(f'Pedestrian in the environment? {self.name in self.scene_objs}')

        self.no_ped_pose = self.client.simGetObjectPose(self.name)
        self.no_ped_pose.position.z_val += 100 # put ped under the ground
        
        found = self.client.simSetSegmentationObjectID("[\w]*", -1, True);
        assert found
        found = self.client.simSetSegmentationObjectID(mesh_name=self.name, object_id=25)
        assert found
        
        self.ped_RGB = [133, 124, 235]
        self.background_RGB = [130, 219, 128]

        self.speed = 1
        
    def hide(self):
        self.client.simSetObjectPose(self.name, self.no_ped_pose)

    def is_ped_in_scene(self, segmentation_response):
        img_rgb_1d = np.frombuffer(segmentation_response.image_data_uint8, dtype=np.uint8) 
        segmentation_image = img_rgb_1d.reshape(segmentation_response.height, segmentation_response.width, 3)
        match = self.ped_RGB == segmentation_image
        return match.sum() > 0

    def set_speed(self, speed=1):
        self.speed = speed

    def move_pedestrian(self):
        pose = self.client.simGetObjectPose(self.name)

        goal_pose = airsim.Pose()
        delta=airsim.Vector3r(0, -30)
        goal_pose.position = pose.position
        goal_pose.position += delta
        
        self.client.simMovePedestrianToGoal(self.name, goal_pose, self.speed)

    def walk(self):
        if not self.is_thread_active:
            self.is_thread_active = True
            self.thread.start()
            print("-->[Started ped thread]")

    def reset(self):
        self.client.simStopPedestrian(self.name)
        super().reset()
