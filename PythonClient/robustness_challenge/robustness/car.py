import threading
import time

from robustness import airsim
from .tools.car_controller import CarController
from .sim_object import SimObject
from .detection_modules import DetectionSystem

class Car(SimObject):
    def __init__(self, name='car', detection_model=None, enable_api_control=True):
        super().__init__(name)

        ## Car controller init
        self.car_controller = CarController()
        self.car_controller.initialize(enable_api_control)

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
        self.detection.reset()
