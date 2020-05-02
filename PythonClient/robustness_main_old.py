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


class SimObject(object):
    def __init__(self, name):
        self.name = name

        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

    def reset(self): 
        self._stop_thread()
        time.sleep(0.5)
        print(f"-->[Reset {self.name} client]")
        self.client.reset()
        self.client.enableApiControl(False)
 
    def _stop_thread(self):
        if 'is_thread_active' in self.__dict__.keys() and self.is_thread_active:
            self.is_thread_active = False
            self.thread.do_run = False
            self.thread.join()
            print(f"-->[Stopped {self.name} thread]")

    def __del__(self): 
        print('-->[Deleting sim object]')
        self.reset()

class DetectionSystem(SimObject):
    def __init__(self, name='pedestrian_recognition', model_checkpoint=None, img_size=224):
        super().__init__(name)
        self.model_checkpoint = model_checkpoint
        assert model_checkpoint is not None, 'You need to specify a model checkpoint' 
        self.img_size = img_size

        self._init_model()

        # test_image_no_ped = os.path.expanduser('~//Desktop//datasets//pedestrian_recognition_new//no_ped//00000.png')
        # test_image_ped = os.path.expanduser('~//Desktop//datasets//pedestrian_recognition_new//ped//00000.png')
        # self._loaded_model_unit_test(test_image_ped, test_image_no_ped)
        self.detection_metrics = PedDetectionMetrics()
        self.is_ped_detected = False
        self.ped_RGB = [133, 124, 235]

    def _init_model(self):
        checkpoint = torch.load(self.model_checkpoint)
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
                                            transforms.Resize(self.img_size),
                                            transforms.ToTensor(),
                                            # normalize
                                            ])
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.model.eval()

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

    def is_any_ped_in_scene(self, segmentation_response):
        img_rgb_1d = np.frombuffer(segmentation_response.image_data_uint8, dtype=np.uint8) 
        segmentation_image = img_rgb_1d.reshape(segmentation_response.height, segmentation_response.width, 3)
        match = self.ped_RGB == segmentation_image
        return match.sum() > 0

    def ped_detection_callback(self):
        img_rgb, response = self._get_image(return_raw=True)

        ground_truth = self.is_any_ped_in_scene(response[1])
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
            self.client.simPrintLogMessage(message=metric_name+' : ', message_param=str(metric_val), severity=severity) # print in red

    def repeat_timer_ped_detection_callback(self, task, period):
        max_count = 50
        count = 0
        times = np.zeros((max_count, ))
        while self.is_thread_active:
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

    def run(self):
        self.thread = threading.Thread(target=self.repeat_timer_ped_detection_callback, 
                                                            args=(self.ped_detection_callback, 0.01))
        self.is_thread_active = False
        self.detection_metrics.reset()
        if not self.is_thread_active:
            self.is_thread_active = True
            self.thread.start()
            print("-->[Started pedestrian detection callback thread]")

    def _get_image(self, return_raw=False):
        request = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
                    ]
        response = self.client.simGetImages(request)
        while response[0].height == 0 or response[0].width == 0:
            time.sleep(0.001)
            response[0] = self.client.simGetImages(request)[0]
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        if return_raw:
            return img_rgb, response
        return img_rgb

    def repeat_timer_image_callback(self, period=0.001):
        max_count = 50
        count = 0
        times = np.zeros((max_count, ))
        while self.is_thread_active:
            start_time = time.time()

            img_rgb = self._get_image()
            # print(self.is_ped_in_scene(response[1]))
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)

            time.sleep(period)
            times[count] = time.time() - start_time
            count += 1
            if count == max_count:
                count = 0
                avg_time = times.mean()
                avg_freq = 1/avg_time
                print('Average camera stream over {} iterations: {} ms | {} Hz'.format(max_count, avg_time*1000, avg_freq))

    def display_image_stream(self):
        self.thread = threading.Thread(target=self.repeat_timer_image_callback)
        self.is_thread_active = False
        if not self.is_thread_active:
            self.is_thread_active = True
            self.thread.start()
            print("-->[Started image callback thread]")

    def reset(self):
        self._stop_thread()
        print(json.dumps(self.detection_metrics.get(), 
                        indent=2, sort_keys=False), 
                        file=output, flush=True)
        time.sleep(0.5)
        super().reset()

class Car(SimObject):
    def __init__(self, name='car', detection_model=None):
        super().__init__(name)

        ## Car controller init
        self.car_controller = CarController()
        self.car_controller.initialize()

        ## Detection system mounted on the car
        self.detection = DetectionSystem(model_checkpoint=detection_model)

        self.thread = threading.Thread(target=self._drive)
        self.is_thread_active = False
    

    def _drive(self):
        while self.is_thread_active:
            self.car_controller.resume()
            self.car_controller.run_one_step()
        
        self.car_controller.stop()
        self.car_controller.run_one_step()

    def drive(self):
        if not self.is_thread_active:
            self.is_thread_active = True
            self.thread.start()
            print("-->[Driving the car]")

    def reset(self):
        self._stop_thread()
        self.car_controller.stop()
        self.car_controller.run_one_step()
        time.sleep(0.5)
        super().reset()
        car.detection.reset()

class AdversarialObjects(SimObject):
    def __init__(self, name='3DAdversary', car=None, **kwargs):
        super().__init__(name)

        assert 'resolution_coord_descent' in kwargs and 'num_iter' in kwargs

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
        
        self.adv_config_path = args.adv_config_path
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


class Weather(SimObject):
    def __init__(self, name='weather'):
        super().__init__(name)
        self.thread = threading.Thread(target=self._demo_weather)
        self.is_thread_active = False
        self.client.simEnableWeather(True)
        self.attributes = ['Rain', 'Roadwetness', 'Snow', 'RoadSnow', 'MapleLeaf', 'Dust', 'Fog']

    def _demo_weather(self):
        ###############################################
        # Control the weather

        for att in self.attributes:
            att = airsim.WeatherParameter.__dict__[att]
            if not self.is_thread_active:
                break
            self.client.simSetWeatherParameter(att, 0.75)
            time.sleep(3)
            self.client.simSetWeatherParameter(att, 0.0)

    def start(self):
        if not self.is_thread_active:
            self.is_thread_active = True
            self.thread.start()
            print("-->[Started weather thread]")

    def reset(self):
        self.client.simEnableWeather(False)        
        super().reset()

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

    car = Car(detection_model=args.model)
    adversary = AdversarialObjects('adversary', car, 
                            resolution_coord_descent=args.resolution_coord_descent,
                            num_iter=args.num_iter)
    weather = Weather()
    ped = Pedestrian()

    # embed()
    if args.demo_id == 0:
        car.detection.display_image_stream()

    if args.demo_id == 1:
        car.detection.run()
        time.sleep(3)
        ped.walk()
        time.sleep(2)
        car.drive()
        weather.start()

    if args.demo_id == 2:
        car.client.simPause(True)

        adversary.adv_config_path = './adv_configs/config_fp_2.json'
        #remove ped from scene
        ped.hide()
        adversary.attack()        
 
        adversary.adv_config_path = './adv_configs/config_fn_4.json'
        adversary.attack()        

    elif args.demo_id == 3:
        car.client.simPause(True)

        # adversary.update_env_from_config(path='./adv_configs/config_fp.json')
        adversary.update_env_from_config(path='./adv_configs/config_fp_2.json')
        
        # adversary.update_env_from_config(path='./adv_configs/config_fn.json')
        # adversary.update_env_from_config(path='./adv_configs/config_fn_2.json')
        # adversary.update_env_from_config(path='./adv_configs/config_fn_3.json')
        car.detection.run()
        # car.drive()

    elif args.demo_id == 4:
        ATTACK = True
        car.detection.run()
        time.sleep(3)
        ped.walk()
        time.sleep(2)
        car.drive()

    embed()
    
    adversary.reset()
    car.reset()
    weather.reset()
    ped.reset()
