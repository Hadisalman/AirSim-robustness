import os
import shutil

import airsim
import cv2
import numpy as np
from IPython import embed


def randomSample(value_range):
    return (value_range[1] - value_range[0])*np.random.random() + value_range[0]


class ped_recognition_dataset(object):
    def __init__(self, num_samples):
        super(ped_recognition_dataset, self).__init__()
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.num_samples = num_samples
        self.curr_idx = 0
        
        self.scene_objs = self.client.simListSceneObjects()

        ########################################################
        # Scene Hard-coded params
        # Car_R_Start, Car_L_Start, Car_R_End, Car_L_End
        # Ped_Start, Ped_End
        assert ('Car_L_Start' in self.scene_objs and 'Car_R_Start' in self.scene_objs and 
                'Car_L_End' in self.scene_objs and 'Car_R_End' in self.scene_objs and 
                'Ped_Start' in self.scene_objs and 'Ped_End' in self.scene_objs)
                
        self.CAR_L_START = self.client.simGetObjectPose('Car_L_Start')
        self.CAR_R_START = self.client.simGetObjectPose('Car_R_Start')
        self.CAR_L_END = self.client.simGetObjectPose('Car_L_End')
        self.CAR_R_END = self.client.simGetObjectPose('Car_R_End')
        self.x_value_range_car = (self.CAR_L_START.position.x_val, self.CAR_R_END.position.x_val)
        self.y_value_range_car = (self.CAR_L_START.position.y_val, self.CAR_R_END.position.y_val)
        self.yaw_value_range_car = (-10*np.pi/180, 10*np.pi/180)
        self.car_pose = self.client.simGetVehiclePose()
        self.euler_car = list(airsim.to_eularian_angles(self.car_pose.orientation))

        self.PED_START = self.client.simGetObjectPose('Ped_Start')
        self.PED_END = self.client.simGetObjectPose('Ped_End')
        self.x_value_range_ped = (self.PED_START.position.x_val, self.PED_END.position.x_val)
        self.y_value_range_ped = (self.PED_START.position.y_val, self.PED_END.position.y_val)
        self.yaw_value_range_ped = (-180*np.pi/180, 180*np.pi/180)
        self.ped_pose = self.client.simGetObjectPose('Hadi')
        self.euler_ped = list(airsim.to_eularian_angles(self.ped_pose.orientation))
        
        self.no_ped_pose = self.client.simGetObjectPose('Hadi')
        self.no_ped_pose.position.x_val += 100
        
        # Segmentation params
        found = self.client.simSetSegmentationObjectID(mesh_name="Hadi", object_id=2);
        assert found
        # https://microsoft.github.io/AirSim/docs/seg_rgbs.txt
        self.RGB_ped = [112, 105, 191] # Corresponding to object_id=2

        # Save dir
        self.dataset_path = "C:/Users/hasalman/Desktop/datasets/pedestrian_recognition_2"
        self.no_ped_path = os.path.join(self.dataset_path, 'no_ped')
        self.ped_path = os.path.join(self.dataset_path, 'ped')
        if not os.path.isdir(self.no_ped_path):
            os.makedirs(self.no_ped_path)
        if not os.path.isdir(self.ped_path):
            os.makedirs(self.ped_path)

        # Copy settings.json form local directory to the dataset_path
        # to keep track of the config used when generating the dataset
        shutil.copy(os.path.expanduser('~/Documents/AirSim/settings.json'), self.dataset_path)

    def generate(self, print_freq=100):
        while self.curr_idx < self.num_samples:
            if self.curr_idx % print_freq == 0:
                print("Generated {} data samples out of {} so far".format(self.curr_idx+1, self.num_samples))
            self.sample_data_point()

    def sample_data_point(self):
        # Sample car position
        x_car = randomSample(self.x_value_range_car)
        y_car = randomSample(self.y_value_range_car)        
        self.car_pose.position.x_val = x_car
        self.car_pose.position.y_val = y_car
        self.euler_car[2] = randomSample(self.yaw_value_range_car)
        self.car_pose.orientation = airsim.to_quaternion(*self.euler_car)
        self.client.simSetVehiclePose(self.car_pose, ignore_collison=True)

        # Sample pedestrian position
        x_ped = randomSample(self.x_value_range_ped)
        y_ped = randomSample(self.y_value_range_ped)
        self.ped_pose.position.x_val = x_ped
        self.ped_pose.position.y_val = y_ped
        self.euler_ped[2] = randomSample(self.yaw_value_range_ped)
        self.ped_pose.orientation = airsim.to_quaternion(*self.euler_ped)
        self.client.simSetObjectPose('Hadi', self.ped_pose)

        request = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]        
        response = self.client.simGetImages(request)
        self.writeImgToFile(image_response=response[0], path=self.ped_path)

        # Remove Ped from scene, then save a "no_ped" image
        self.client.simSetObjectPose('Hadi', self.no_ped_pose)
        response = self.client.simGetImages(request)
        self.writeImgToFile(image_response=response[0], path=self.no_ped_path)

        #Check of pedestrian is in fov of the car using segmentation ground truth
        # embed()

        # request = [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)]        
        # response = self.client.simGetImages(request)
        # img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8) 
        # img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3) #reshape array to 3 channel image array H X W X 3

        self.curr_idx += 1


    def writeImgToFile(self, image_response, path):
        if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            cv2.imwrite(os.path.join(path, str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)  # write to png
        else:
            print('ERROR IN IMAGE SIZE -- NOT SUPPOSED TO HAPPEN')


if __name__ == "__main__":
    dataset = ped_recognition_dataset(num_samples=20000)
    dataset.generate()
