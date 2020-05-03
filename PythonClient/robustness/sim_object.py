from robustness import airsim
import time

class SimObject(object):
    def __init__(self, name):
        self.name = name

        self.client = airsim.Client()
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
