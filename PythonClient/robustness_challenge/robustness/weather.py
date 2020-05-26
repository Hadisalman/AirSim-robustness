import threading
import time

from robustness import airsim
from .sim_object import SimObject

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