import json
import os
import threading
import time

import numpy as np
import PIL
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import cv2
from .tools.attacks import PGD, NormalizeLayer
from .tools.utils import PedDetectionMetrics

from robustness import airsim
from .sim_object import SimObject

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

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
        self.is_attack_on = False
    
    def setup_attack(self, attack_config):
        self.attacker = PGD(**attack_config)

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
        if self.is_attack_on:
            X = self.attacker.attack(self.model, X, target, self.normalize)
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

    def run(self, with_attack=False):
        self.is_attack_on = with_attack

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
                        file=open("./stdout.txt", mode = 'w'), flush=True)
        time.sleep(0.5)
        super().reset()