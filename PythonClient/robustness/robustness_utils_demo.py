import argparse
import os
import threading
import time
import json

import airsim
import cv2
import numpy as np
import PIL
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from IPython import embed

from attacks import PGD, NormalizeLayer
from utils import PedDetectionMetrics

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

output = open("./stdout.txt", mode = 'w')

# Linf Whitebox
attack_config = {
    'random_start' : True, 
    'step_size' : 1./255,
    'epsilon' : 2./255, 
    'num_steps' : 2, 
    'norm' : 'linf',
    }

# Linf Blackbox
# attack_config = {
#     'random_start' : True, 
#     'step_size' : 4./255,
#     'epsilon' : 16./255, 
#     'num_steps' : 8, 
#     'norm' : 'linf',
#     'est_grad': (5, 200)
#     }

# L2 Whitebox
# attack_config = {
#     'random_start' : True, 
#     'step_size' : 150./255,
#     'epsilon' : 255./255, 
#     'num_steps' : 2, 
#     'norm' : 'l2',
#     }

# L2 Blackbox
# attack_config = {
#     'random_start' : True, 
#     'step_size' : 500./255,
#     'epsilon' : 4000./255, 
#     'num_steps' : 8, 
#     'norm' : 'l2',
#     'est_grad': (5, 200)
#     }

ATTACK = False
ATTACKER = PGD(**attack_config)

class Demo():
    def __init__(self, args):
        self.args = args
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
        self.client_images = airsim.CarClient()
        self.client_images.confirmConnection()
        self.client_ped_detection = airsim.CarClient()
        self.client_ped_detection.confirmConnection()

        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback, args=(self.image_callback, 0.001))
        self.is_image_thread_active = False
        
        #########################
        # Pedestrian detection
        self.ped_detection_callback_thread = threading.Thread(target=self.repeat_timer_ped_detection_callback, args=(self.ped_detection_callback, 0.01))
        self.is_ped_detection_thread_active = False

        checkpoint = torch.load(args.model)
        print("=> creating model '{}'".format(checkpoint["arch"]))

        self.model = models.__dict__[checkpoint["arch"]]()
        self.model.fc = torch.nn.Linear(512, 2)    
        if checkpoint["arch"].startswith('alexnet') or checkpoint["arch"].startswith('vgg'):
            self.model.features = torch.nn.DataParallel(self.model.features)
            self.model.cuda()
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Loading successful. Test accuracy of this model is: {} %".format(checkpoint['test_acc']))
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        self.normalize = NormalizeLayer(means=[0.485, 0.456, 0.406], sds=[0.229, 0.224, 0.225])
        self.transform_test = transforms.Compose([
                                            transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            # normalize
                                            ])
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.model.eval()
        test_image_no_ped = os.path.expanduser('~//Desktop//datasets//pedestrian_recognition_new//no_ped//00000.png')
        test_image_ped = os.path.expanduser('~//Desktop//datasets//pedestrian_recognition_new//ped//00000.png')
        self._loaded_model_unit_test(test_image_ped, test_image_no_ped)
        self.detection_metrics = PedDetectionMetrics()
        #########################

        self.car_thread = threading.Thread(target=self.drive)
        self.ped_thread = threading.Thread(target=self.move_pedestrian)
        self.adv_thread = threading.Thread(target=self.start_attack)
        self.weather_thread = threading.Thread(target=self.demo_weather)
        self.is_car_thread_active = False
        self.is_ped_thread_active = False
        self.is_adv_thread_active = False
        self.is_weather_thread_active = False

        ##############################################
        # Segmentation Settings
        found = self.client_images.simSetSegmentationObjectID("[\w]*", -1, True);
        assert found
        found = self.client_images.simSetSegmentationObjectID(mesh_name='Adv_Ped1', object_id=25)
        assert found
        self.ped_RGB = [133, 124, 235]
        self.background_RGB = [130, 219, 128]

        self.adv_objects = [
            'Adv_Ped1',
            'Adv_Fence',
            'Adv_Hedge',
            'Adv_Car',
            'Adv_House',
            'Adv_Tree'
            ]

        self.scene_objs = self.client_car.simListSceneObjects()
        for obj in self.adv_objects:
            print('{} exists? {}'.format(obj, obj in self.scene_objs))

    def _loaded_model_unit_test(self, test_image_True, test_image_False):
        img = PIL.Image.open(test_image_True)
        X = self.transform_test(img)
        pred = self.model(X.unsqueeze(0))
        assert pred.max(1)[1].item() == 1, "Pedestrian detection unit test failed"

        img = PIL.Image.open(test_image_False)
        X = self.transform_test(img).cuda()
        X = self.normalize(X.unsqueeze(0))
        pred = self.model(X)
        assert pred.max(1)[1].item() == 0, "Pedestrian detection unit test failed"
        print("Loaded detection model unit test succeeded!")

    def is_ped_in_scene(self, segmentation_response):
        img_rgb_1d = np.frombuffer(segmentation_response.image_data_uint8, dtype=np.uint8) 
        segmentation_image = img_rgb_1d.reshape(segmentation_response.height, segmentation_response.width, 3)
        match = self.ped_RGB == segmentation_image
        return match.sum() > 0

    def image_callback(self):
        request = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        # request = [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)]
        # request = [airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)]

        response = self.client_images.simGetImages(request)[0]
        while response.height == 0 or response.width == 0:
            time.sleep(0.001)
            response = self.client_images.simGetImages(request)[0]
        img_rgb_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response.height, response.width, 3)

        print(self.is_ped_in_scene())

        cv2.imshow("img_rgb", img_rgb)
        cv2.waitKey(1)

    def repeat_timer_image_callback(self, task, period):
        max_count = 50
        count = 0
        times = np.zeros((max_count, ))
        while self.is_image_thread_active:
            start_time = time.time()
            task()
            time.sleep(period)
            times[count] = time.time() - start_time
            count += 1
            if count == max_count:
                count = 0
                avg_time = times.mean()
                avg_freq = 1/avg_time
                print('Average camera stream over {} iterations: {} ms | {} Hz'.format(max_count, avg_time*1000, avg_freq))


    def ped_detection_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)]
        response = self.client_ped_detection.simGetImages(request)
        while response[0].height == 0 or response[0].width == 0:
            time.sleep(0.001)
            response = self.client_ped_detection.simGetImages(request)

        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        img_rgb = PIL.Image.fromarray(img_rgb)
        X = self.transform_test(img_rgb).unsqueeze(0).cuda()
        targets = torch.full(X.shape[:1], 1).long().cuda()
        if ATTACK:
            X = ATTACKER.attack(self.model, X, targets, self.normalize)
            cv2.imshow("adversarial image", X.cpu().numpy()[0].transpose(1,2,0))
            cv2.waitKey(1)
        X = self.normalize(X)
        pred = self.model(X)
        is_ped_detected = pred.max(1)[1].item()
        self.detection_metrics.update(pred=is_ped_detected, ground_truth=self.is_ped_in_scene(response[1]))
        
        loss = self.criterion(pred, targets)
        # print("Pedestrian detected? {}".format(is_ped_detected), file=output, flush=True)
        # print("Loss =  {}".format(loss.item()), file=output, flush=True)
        # print("Pedestrian detected? {}".format(is_ped_detected))

    def repeat_timer_ped_detection_callback(self, task, period):
        max_count = 50
        count = 0
        times = np.zeros((max_count, ))
        while self.is_ped_detection_thread_active:
            start_time = time.time()
            task()
            time.sleep(period)
            times[count] = time.time() - start_time
            count += 1
            if count == max_count:
                count = 0
                avg_time = times.mean()
                avg_freq = 1/avg_time
                print('Average pedestrian detection over {} iterations: {} ms | {} Hz'.format(max_count, avg_time*1000, avg_freq))


    def move_pedestrian(self, obj='Adv_Ped1'):
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

    def start_ped_detection_callback_thread(self):
        if not self.is_ped_detection_thread_active:
            self.is_ped_detection_thread_active = True
            self.ped_detection_callback_thread.start()
            print("Started pedestrian detection callback thread")

    def stop_ped_detection_callback_thread(self):
        if self.is_ped_detection_thread_active:
            self.is_ped_detection_thread_active = False
            self.ped_detection_callback_thread.join()
            print(json.dumps(self.detection_metrics.get(), 
                            indent=4, sort_keys=False), 
                            file=output, flush=True)
            print("Stopped pedestrian detection callback thread.")

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
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('model', metavar='DIR',
                        help='path to pretrained model')
    parser.add_argument('--img-size', default=224, type=int, metavar='N',
                        help='size of rgb image (assuming equal hight and width)')

    args = parser.parse_args()

    demo = Demo(args)
    
    # demo.client_car.simSetTimeOfDay(is_enabled=True, 
    #                             start_datetime = "", 
    #                             is_start_datetime_dst = False, 
    #                             celestial_clock_speed = 100, 
    #                             update_interval_secs = 1, 
    #                             move_sun = True)

    embed()

    demo.start_ped_detection_callback_thread()
    time.sleep(3)
    # demo.start_car_thread()
    # demo.start_adv_thread()
    demo.start_ped_thread()
    # demo.start_weather_thread()
    # demo.start_image_callback_thread()

    embed()

    demo.stop_ped_detection_callback_thread()
    demo.stop_ped_thread()
    demo.stop_adv_thread()
    demo.stop_weather_thread()
    demo.stop_car_thread()
    demo.stop_image_callback_thread()
   
    demo.reset()
