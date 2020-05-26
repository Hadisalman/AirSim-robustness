import argparse
import copy
import json
import os
import threading
import time
import sys

from robustness import airsim
import numpy as np
import PIL
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from IPython import embed

import cv2
from tools.attacks import PGD, NormalizeLayer
from tools.utils import PedDetectionMetrics
from tools.car_controller import CarController
from configs import attack_config

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

output = open("./stdout.txt", mode = 'w')

ATTACK = False
ATTACKER = PGD(**attack_config)

class Demo():
    def __init__(self, args):
        self.args = args
        ###############################################
        # # connect to the AirSim simulator 
        self.clients = {
            'car': airsim.CarClient(),
            'ped': airsim.CarClient(),
            'adv': airsim.CarClient(),
            'weather': airsim.CarClient(),
            'images': airsim.CarClient(),
            'detection': airsim.CarClient(),
        }
        self.clients['car'].confirmConnection()
        self.clients['car'].enableApiControl(True)

        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback, 
                                                    args=(self.image_callback, 0.001))
        self.is_image_thread_active = False
        
        #########################
        # Pedestrian detection
        self.ped_detection_callback_thread = threading.Thread(target=self.repeat_timer_ped_detection_callback, 
                                                            args=(self.ped_detection_callback, 0.01))
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
        self.normalize = NormalizeLayer(means=[0.485, 0.456, 0.406], sds=[0.229, 0.224, 0.225])
        self.transform_test = transforms.Compose([
                                            transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            # normalize
                                            ])
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.model.eval()
        # test_image_no_ped = os.path.expanduser('~//Desktop//datasets//pedestrian_recognition_new//no_ped//00000.png')
        # test_image_ped = os.path.expanduser('~//Desktop//datasets//pedestrian_recognition_new//ped//00000.png')
        # self._loaded_model_unit_test(test_image_ped, test_image_no_ped)
        self.detection_metrics = PedDetectionMetrics()
        self.is_ped_detected = False
        #########################
        ## Car controller init
        self.car = CarController()
        self.car.initialize()

        #########################
        self.car_thread = threading.Thread(target=self.drive)
        self.ped_thread = threading.Thread(target=self.move_pedestrian)
        self.adv_attack_resolution = args.resolution_coord_descent
        self.num_iter = args.num_iter
        self.adv_thread = threading.Thread(target=self.coordinate_ascent_object_attack, args=(self.adv_attack_resolution, self.num_iter))
        self.weather_thread = threading.Thread(target=self.demo_weather)
        self.is_car_thread_active = False
        self.is_ped_thread_active = False
        self.is_adv_thread_active = False
        self.is_weather_thread_active = False

        ##############################################
        # Segmentation Settings
        self.ped_object_name = 'Adv_Ped2'
        self.scene_objs = self.clients['car'].simListSceneObjects()
        print(f'Pedestrian in the environment? {self.ped_object_name in self.scene_objs}')

        self.no_ped_pose = self.clients['ped'].simGetObjectPose(self.ped_object_name)
        self.no_ped_pose.position.z_val += 100 # put ped under the ground
        found = self.clients['images'].simSetSegmentationObjectID("[\w]*", -1, True);
        assert found
        found = self.clients['images'].simSetSegmentationObjectID(mesh_name=self.ped_object_name, object_id=25)
        # found = self.clients['images'].simSetSegmentationObjectID(mesh_name=self.ped_object_name+'_walking', object_id=25)
        assert found
        self.ped_RGB = [133, 124, 235]
        self.background_RGB = [130, 219, 128]

        #############################################
        # Adversarial objects
        self.adv_objects = [
            'Adv_House',
            'Adv_Fence',
            'Adv_Hedge',
            'Adv_Car',
            'Adv_Tree'
            ]
        
        self.adv_config_path = args.adv_config_path
        for obj in self.adv_objects:
            print('{} exists? {}'.format(obj, obj in self.scene_objs))

        for obj in ['BoundLowerLeft', 'BoundUpperRight']:
            print('{} exists? {}'.format(obj, obj in self.scene_objs))

        self.BoundLowerLeft = self.clients['adv'].simGetObjectPose('BoundLowerLeft')
        self.BoundUpperRight = self.clients['adv'].simGetObjectPose('BoundUpperRight')

        self.x_range_adv_objects_bounds = (self.BoundLowerLeft.position.x_val, self.BoundUpperRight.position.x_val)
        self.y_range_adv_objects_bounds = (self.BoundLowerLeft.position.y_val, self.BoundUpperRight.position.y_val)


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
        request = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
                    ]
        # request = [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)]
        # request = [airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)]

        response = self.clients['images'].simGetImages(request)
        while response[0].height == 0 or response[0].width == 0:
            time.sleep(0.001)
            response[0] = self.clients['images'].simGetImages(request)[0]
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)

        # print(self.is_ped_in_scene(response[1]))

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
        response = self.clients['detection'].simGetImages(request)
        while response[0].height == 0 or response[0].width == 0:
            time.sleep(0.001)
            response = self.clients['detection'].simGetImages(request)

        ground_truth = self.is_ped_in_scene(response[1])
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        img_rgb = PIL.Image.fromarray(img_rgb)
        X = self.transform_test(img_rgb).unsqueeze(0).cuda()
        
        # target = torch.full(X.shape[:1], 1).long().cuda()
        target = torch.tensor([ground_truth], dtype=torch.long).cuda()
        if ATTACK:
            X = ATTACKER.attack(self.model, X, target, self.normalize)
            cv2.imshow("adversarial image", X.cpu().numpy()[0].transpose(1,2,0))
            cv2.waitKey(1)
        X = self.normalize(X)
        pred = self.model(X)
        self.is_ped_detected = pred.max(1)[1].item()
        self.detection_metrics.update(pred=self.is_ped_detected, ground_truth=ground_truth)
        
        loss = self.criterion(pred, target)
        # print("Pedestrian detected? {}".format(self.is_ped_detected), file=output, flush=True)
        # print("Loss =  {}".format(loss.item()), file=output, flush=True)
        # print("Pedestrian detected? {}".format(self.is_ped_detected))
        return self.is_ped_detected, self.is_ped_detected==ground_truth, loss

    def log_detection_metrics(self):
        metrics = self.detection_metrics.get()
        for metric_name, metric_val in metrics.items():
            if metric_name in ['False Negative', 'False Positive']:
                severity = 2
            else:
                severity = 1
            self.clients['detection'].simPrintLogMessage(message=metric_name+' : ', message_param=str(metric_val), severity=severity) # print in red

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
            self.log_detection_metrics()

    def move_pedestrian(self):
        pose = self.clients['ped'].simGetObjectPose(self.ped_object_name)

        goal_pose = airsim.Pose()
        delta=airsim.Vector3r(0, -30)
        goal_pose.position = pose.position
        goal_pose.position += delta
        
        self.clients['ped'].simMovePedestrianToGoal(self.ped_object_name, goal_pose, 1)
                

    def start_car_thread(self):
        if not self.is_car_thread_active:
            self.is_car_thread_active = True
            self.car_thread.start()
            print("-->[Started car thread]")

    def stop_car_thread(self):
        if self.is_car_thread_active:
            self.is_car_thread_active = False
            self.car_thread.do_run = False
            self.car_thread.join()
            print("-->[Stopped car thread]")

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("-->[Started image callback thread]")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("-->[Stopped image callback thread]")

    def start_ped_detection_callback_thread(self):
        self.detection_metrics.reset()
        if not self.is_ped_detection_thread_active:
            self.is_ped_detection_thread_active = True
            self.ped_detection_callback_thread.start()
            print("-->[Started pedestrian detection callback thread]")

    def stop_ped_detection_callback_thread(self):
        if self.is_ped_detection_thread_active:
            self.is_ped_detection_thread_active = False
            self.ped_detection_callback_thread.join()
            print(json.dumps(self.detection_metrics.get(), 
                            indent=2, sort_keys=False), 
                            file=output, flush=True)
            print("-->[Stopped pedestrian detection callback thread]")

    def start_ped_thread(self):
        if not self.is_ped_thread_active:
            self.is_ped_thread_active = True
            self.ped_thread.start()
            print("-->[Started ped thread]")

    def stop_ped_thread(self):
        if self.is_ped_thread_active:
            self.clients['ped'].simStopPedestrian(self.ped_object_name)
            self.is_ped_thread_active = False
            self.ped_thread.do_run = False
            self.ped_thread.join()
            print("-->[Stopped ped thread]")

    def start_adv_thread(self):
        if not self.is_adv_thread_active:
            self.is_adv_thread_active = True
            self.adv_thread.start()
            print("-->[Started adv thread]")

    def stop_adv_thread(self):
        if self.is_adv_thread_active:
            self.is_adv_thread_active = False
            self.adv_thread.join()
            print("-->[Stopped adv thread]")

    def start_weather_thread(self):
        if not self.is_weather_thread_active:
            self.is_weather_thread_active = True
            self.weather_thread.start()
            print("-->[Started weather thread]")

    def stop_weather_thread(self):
        if self.is_weather_thread_active:
            self.is_weather_thread_active = False
            self.weather_thread.do_run = False
            self.weather_thread.join()
            print("-->[Stopped weather thread]")


    def coordinate_ascent_object_attack(self, resolution=10, num_iter=1):
        x_range = np.linspace(self.x_range_adv_objects_bounds[0], self.x_range_adv_objects_bounds[1], resolution)
        y_range = np.linspace(self.y_range_adv_objects_bounds[0], self.y_range_adv_objects_bounds[1], resolution)
        xv, yv = np.meshgrid(x_range, y_range)

        self.adv_poses = []

        best_loss = -1
        for _ in range(num_iter):
            for obj in np.random.permutation(self.adv_objects).tolist():
                pose = self.clients['adv'].simGetObjectPose(obj)
                best_pose = copy.deepcopy(pose)
                grid2d_poses_list = zip(xv.flatten(), yv.flatten())
                for grid2d_pose in grid2d_poses_list:
                    pose.position.x_val = grid2d_pose[0]
                    pose.position.y_val = grid2d_pose[1]
                    self.clients['adv'].simSetObjectPose(obj, pose)
                    if not self.is_adv_thread_active:
                        break
                    _, correct, loss = self.ped_detection_callback()
                    if loss > best_loss:
                        best_loss = loss
                        best_pose = copy.deepcopy(pose)
                print('Best loss so far {}'.format(best_loss.item()))

                self.clients['adv'].simSetObjectPose(obj, best_pose)
        
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
                pose = self.clients['adv'].simGetObjectPose(obj)
                best_pose = copy.deepcopy(pose)
                grid2d_poses_list = zip(xv.flatten(), yv.flatten())
                for grid2d_pose in grid2d_poses_list:
                    pose.position.x_val = grid2d_pose[0]
                    pose.position.y_val = grid2d_pose[1]
                    self.clients['adv'].simSetObjectPose(obj, pose)
                    if not self.is_adv_thread_active:
                        break
                    _, correct, loss = self.ped_detection_callback()
                    if loss > best_loss:
                        best_loss = loss
                        best_pose = copy.deepcopy(pose)
                print('Best loss so far {}'.format(best_loss.item()))

                self.clients['adv'].simSetObjectPose(obj, best_pose)
        
            # dump results into a json file after each iteration
            self.dump_env_config_to_json(path=self.adv_config_path)


    def demo_weather(self):
        ###############################################
        # Control the weather
        self.clients['weather'].simEnableWeather(True)
        attributes = ['Rain', 'Roadwetness', 'Snow', 'RoadSnow', 'MapleLeaf', 'Dust', 'Fog']

        for att in attributes:
            att = airsim.WeatherParameter.__dict__[att]
            if not self.is_weather_thread_active:
                break
            self.clients['weather'].simSetWeatherParameter(att, 0.75)
            time.sleep(3)
            self.clients['weather'].simSetWeatherParameter(att, 0.0)
        self.clients['weather'].simEnableWeather(False)

    def drive(self):
        while self.is_car_thread_active:
            if self.is_ped_detected and self.car.is_moving(): 
                self.car.stop()
            elif not self.is_ped_detected and not self.car.is_moving(): 
                self.car.resume()
            self.car.run_one_step()
        
        self.car.stop()
        self.car.run_one_step()

    def reset(self):
        for client in self.clients.values():
            client.reset()
            client.enableApiControl(False)

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
            pose = self.clients['adv'].simGetVehiclePose()
            _populate_pose_dic(output['Vehicle'], pose)

            output[self.ped_object_name] = {}
            pose = self.clients['adv'].simGetObjectPose(self.ped_object_name)
            _populate_pose_dic(output[self.ped_object_name], pose)

            for obj in self.adv_objects:
                output[obj] = {}
                pose = self.clients['adv'].simGetObjectPose(obj)
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
                    self.clients['adv'].simSetVehiclePose(pose, ignore_collison=True)
                else:
                    assert obj_name in self.scene_objs, 'Object {} is not found in the scene'.format(obj_name)
                    self.clients['adv'].simSetObjectPose(obj_name, pose)
                print('-->[Updated the position of the {}]'.format(obj_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo for the airsim-robustness package')
    parser.add_argument('model', metavar='DIR',
                        help='path to pretrained model')
    parser.add_argument('--demo-id', type=int, choices=[0, 1, 2, 3, 4],
                        help='which task of the demo to excute'
                        '0 -> image callback thread'
                        '1 -> test all threads'
                        '2 -> search for 3D advesarial configuration'
                        '3 -> read adv_config.json and run ped recognition'
                        '4 -> pixel pgd attack'
                        )
    parser.add_argument('--img-size', default=224, type=int, metavar='N',
                        help='size of rgb image (assuming equal height and width)')
    parser.add_argument('--resolution-coord-descent', default=10, type=int,
                        help='resolution of coord descent 3D object adv attack')
    parser.add_argument('--num-iter', default=1, type=int,
                        help='number of iterators of coord descent 3D object adv attack')
    parser.add_argument('--adv-config-path', type=str, default='./results.json')

    args = parser.parse_args()

    demo = Demo(args)

    if args.demo_id == 0:
        demo.start_image_callback_thread()

    if args.demo_id == 1:
        demo.start_ped_detection_callback_thread()
        time.sleep(3)
        demo.start_ped_thread()
        time.sleep(2)
        demo.start_car_thread()
        # demo.start_weather_thread()

    if args.demo_id == 2:
        # demo.client_car.simPause(True)

        demo.adv_config_path = './adv_configs/config_fp_2.json'
        #remove ped from scene
        demo.clients['ped'].simSetObjectPose(demo.ped_object_name, demo.no_ped_pose)
        demo.start_adv_thread()
        
        # demo.adv_config_path = './config_fn_4.json'
        # demo.start_adv_thread()

    elif args.demo_id == 3:
        demo.client_car.simPause(False)

        demo.update_env_from_config(path='./adv_configs/config_fp.json')
        # demo.update_env_from_config(path='./adv_configs/config_fp_2.json')
        
        # demo.update_env_from_config(path='./adv_configs/config_fn.json')
        # demo.update_env_from_config(path='./adv_configs/config_fn_2.json')
        # demo.update_env_from_config(path='./adv_configs/config_fn_3.json')
        
        demo.start_ped_detection_callback_thread()
        # demo.start_car_thread()

    elif args.demo_id == 4:
        ATTACK = True
        demo.start_ped_detection_callback_thread()
        time.sleep(3)
        demo.start_ped_thread()
        time.sleep(2)
        demo.start_car_thread()
        # demo.start_weather_thread()

    embed()

    demo.stop_ped_detection_callback_thread()
    demo.stop_ped_thread()
    demo.stop_adv_thread()
    demo.stop_car_thread()
    demo.stop_image_callback_thread()
    demo.stop_weather_thread()
    
    embed()
    
    demo.reset()
