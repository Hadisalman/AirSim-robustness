import os
import threading
import time

import airsim
import cv2
import numpy as np
from IPython import embed


class Demo():
    def __init__(self, ):
        ###############################################
        # # connect to the AirSim simulator 
        self.client_car = airsim.CarClient()
        self.client_car.confirmConnection()
        self.client_car.enableApiControl(True)

        self.client_ped = airsim.CarClient()
        self.client_ped.confirmConnection()
        self.client_adv = airsim.CarClient()
        self.client_adv.confirmConnection()
        self.client_weather = airsim.CarClient()
        self.client_weather.confirmConnection()
        self.airsim_client_images = airsim.CarClient()
        self.airsim_client_images.confirmConnection()

        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback, args=(self.image_callback, 0.01))
        self.is_image_thread_active = False

        self.car_thread = threading.Thread(target=self.drive)
        self.ped_thread = threading.Thread(target=self.move_pedestrian)
        self.adv_thread = threading.Thread(target=self.start_attack)
        self.weather_thread = threading.Thread(target=self.demo_weather)
        self.camera_thread = threading.Thread(target=self.demo_weather)
        self.is_car_thread_active = False
        self.is_ped_thread_active = False
        self.is_adv_thread_active = False
        self.is_weather_thread_active = False
        
        self.adv_objects = [
            'Adv_Fence',
            'Adv_Hedge',
            'Adv_Car',
            'Adv_House',
            'Adv_Tree'
            ]

        self.scene_objs = self.client_car.simListSceneObjects()
        for obj in self.adv_objects:
            print('{} exists? {}'.format(obj, obj in self.scene_objs))

    def image_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        # request = [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False)]        
        # request = [airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)]
        # request = [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)]

        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        cv2.imshow("img_rgb", img_rgb)
        cv2.waitKey(1)

    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)


    def move_pedestrian(self, obj='Hadi'):
        end_pose = airsim.Pose()
        delta = airsim.Vector3r(0, -30, 0)

        pose = self.client_ped.simGetObjectPose(obj)
        end_pose.orientation = pose.orientation
        end_pose.position = pose.position
        end_pose.position += delta
        traj = self.get_trajectory(pose, end_pose, 1000)
        for way_point in traj:
            if not self.is_ped_thread_active:
                break
            self.client_ped.simSetObjectPose(obj, way_point)
            time.sleep(0.01)
    
    def start_car_thread(self):
        if not self.is_car_thread_active:
            self.is_car_thread_active = True
            self.car_thread.start()
            print("Started car thread")

    def stop_car_thread(self):
        if self.is_car_thread_active:
            self.is_car_thread_active = False
            self.car_thread.do_run = False
            self.car_thread.join()
            print("Stopped car thread.")

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_ped_thread(self):
        if not self.is_ped_thread_active:
            self.is_ped_thread_active = True
            self.ped_thread.start()
            print("Started ped thread")

    def stop_ped_thread(self):
        if self.is_ped_thread_active:
            self.is_ped_thread_active = False
            self.ped_thread.do_run = False
            self.ped_thread.join()
            print("Stopped ped thread.")

    def start_adv_thread(self):
        if not self.is_adv_thread_active:
            self.is_adv_thread_active = True
            self.adv_thread.start()
            print("Started adv thread")

    def stop_adv_thread(self):
        if self.is_adv_thread_active:
            self.is_adv_thread_active = False
            self.adv_thread.join()
            print("Stopped adv thread.")

    def start_weather_thread(self):
        if not self.is_weather_thread_active:
            self.is_weather_thread_active = True
            self.weather_thread.start()
            print("Started weather thread")

    def stop_weather_thread(self):
        if self.is_weather_thread_active:
            self.is_weather_thread_active = False
            self.weather_thread.join()
            print("Stopped weather thread.")

    def start_attack(self):
        end_pose = airsim.Pose()
        delta = airsim.Vector3r(0, -4, 0)
        for obj in self.adv_objects:
            pose = self.client_adv.simGetObjectPose(obj)
            end_pose.orientation = pose.orientation
            end_pose.position = pose.position
            end_pose.position += delta
            traj = self.get_trajectory(pose, end_pose, 100)
            for way_point in traj:
                if not self.is_adv_thread_active:
                    break
                self.client_adv.simSetObjectPose(obj, way_point)
                time.sleep(0.01)

            pose = self.client_adv.simGetObjectPose(obj)
            end_pose.position = pose.position
            end_pose.position -= delta
            traj = self.get_trajectory(pose, end_pose, 100)
            for way_point in traj:
                if not self.is_adv_thread_active:
                    break
                self.client_adv.simSetObjectPose(obj, way_point)
                time.sleep(0.01)

    def demo_weather(self):
        ###############################################
        # Control the weather
        self.client_weather.simEnableWeather(True)
        attributes = ['Rain', 'Roadwetness', 'Snow', 'RoadSnow', 'MapleLeaf', 'Dust', 'Fog']

        for att in attributes:
            att = airsim.WeatherParameter.__dict__[att]
            if not self.is_weather_thread_active:
                break
            self.client_weather.simSetWeatherParameter(att, 0.75)
            time.sleep(3)
            self.client_weather.simSetWeatherParameter(att, 0.0)
        self.client_weather.simEnableWeather(False)


    def get_trajectory(self, start_pose, end_pose, num_waypoints=10):
        inc_vec = (end_pose.position - start_pose.position)/(num_waypoints - 1)
        traj = []
        traj.append(start_pose)
        for _ in range(num_waypoints - 2):
            traj.append(airsim.Pose())
            traj[-1].orientation = traj[-2].orientation
            traj[-1].position = traj[-2].position + inc_vec
        traj.append(end_pose)
        return traj


    def drive(self):
        car_controls = airsim.CarControls()

        # get state of the car
        car_state = self.client_car.getCarState()
        print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

        # go forward
        car_controls.throttle = 0.5
        car_controls.steering = 0
        self.client_car.setCarControls(car_controls)
        print("Go Forward")
        time.sleep(3)   # let car drive a bit
        if not self.is_car_thread_active:
            return

        # go reverse
        car_controls.throttle = -0.5
        car_controls.is_manual_gear = True;
        car_controls.manual_gear = -1
        car_controls.steering = 0
        self.client_car.setCarControls(car_controls)
        print("Go reverse")
        time.sleep(3)   # let car drive a bit
        if not self.is_car_thread_active:
            return
        car_controls.is_manual_gear = False; # change back gear to auto
        car_controls.manual_gear = 0  

        # Go forward
        car_controls.throttle = 1
        self.client_car.setCarControls(car_controls)
        print("Go Forward")
        time.sleep(3.5)   
        if not self.is_car_thread_active:
            return
        car_controls.throttle = 0.5
        car_controls.steering = 1
        self.client_car.setCarControls(car_controls)
        print("Turn Right")
        time.sleep(1.4)
        if not self.is_car_thread_active:
            return


        car_controls.throttle = 0.5
        car_controls.steering = 0
        self.client_car.setCarControls(car_controls)
        print("Go Forward")
        time.sleep(3)   
        if not self.is_car_thread_active:
            return


        # apply brakes
        car_controls.brake = 1
        self.client_car.setCarControls(car_controls)
        print("Apply brakes")
        time.sleep(3)   
        if not self.is_car_thread_active:
            return
        car_controls.brake = 0 #remove brake
        self.client_car.reset()

        # go forward
        car_controls.throttle = 0.5
        car_controls.steering = 0
        self.client_car.setCarControls(car_controls)
        print("Go Forward")
        time.sleep(3)   # let car drive a bit
        if not self.is_car_thread_active:
            return
        # apply brakes
        car_controls.brake = 1
        self.client_car.setCarControls(car_controls)
        print("Apply brakes")
        time.sleep(3)   

    def reset(self):
        self.client_car.reset()
        self.client_car.enableApiControl(False)



if __name__ == "__main__":
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    demo = Demo()

    embed()

    demo.start_car_thread()
    demo.start_image_callback_thread()
    demo.start_adv_thread()
    demo.start_ped_thread()
    demo.start_weather_thread()

    embed()

    demo.stop_image_callback_thread()
    demo.stop_ped_thread()
    demo.stop_adv_thread()
    demo.stop_weather_thread()
    demo.stop_car_thread()
   
    demo.reset()




# ###############################################
# # Computer vision
# # get camera images from the car
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
#     airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
#     airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
#     airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGB array
# print('Retrieved images: %d', len(responses))

# for response in responses:
#     filename = 'c:/temp/py' + str(0)
#     if not os.path.exists('c:/temp/'):
#         os.makedirs('c:/temp/')
#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
#     elif response.compress: #png format
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
#     else: #uncompressed array
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
#         img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
#         cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png 



###############################################
#restore to original state
