"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("CubicSpline/")
import time as time

import airsim

try:
    import cubic_spline_planner
except:
    raise

import sys, signal

dt = 1.0

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class CarController():
    def __init__(self):
        self.cx = []
        self.cy = []
        self.cyaw = []
        self.last_idx = 0
        self.ck = []
        self.ax = []
        self.ay = []
        self.offx = 0
        self.offy = 0
                
        self.k = 5.0                        # Path control gain
        self.Kp = 1.0                       # Speed proportional gain
        self.dt = 1.0                       # [s] time step
        self.wheel_base = 4.0               # [m] Wheel base of vehicle
        self.max_steer = np.radians(60.0)   # [rad] max steering angle
        self.target_speed = 30.0            # [m/s] Max. target velocity = 60 mph

        self.brake_override = False

        #self.ax = [0.0, 25.0, 40.0, 45.0, 48.0, 50.0, 52.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0,  55.0,  55.0,  55.0,  55.0,  55.0,  55.0,  53.0,  50.0,  40.0,  25.0,   0.0, -40.0, -80.0, -150.0, -170.0, -180.0, -185.0, -188.0, -190.0, -192.0, -197.0, -197.0, -197.0, -197.0, -197.0, -197.0, -197.0, -197.0, -190.0, -185.0, -180.0, -90.0, -20.0, 0.0]
        #self.ay = [0.0, 0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  5.0, 10.0, 20.0, 50.0, 70.0, 100.0, 110.0, 115.0, 117.0, 119.0, 121.0, 126.0, 126.0, 126.0, 126.0, 126.0, 126.0, 126.0,  126.0,  126.0,  126.0,  126.0,  126.0,  126.0,  126.0,  124.0,  121.0,  111.0,   91.0,   41.0,   10.0,    5.0,    0.0,    0.0,    0.0,    0.0,   0.0,   0.0, 0.0]

        self.ax = [0.0, 25.0, 40.0, 45.0, 48.0, 50.0, 52.0, 56.0, 57.0, 57.0, 57.0, 57.0,  57.0,  57.0,  57.0,  57.0, 56.0, 54.0,  50.0, 40.0,  25.0,   0.0, -50.0, -61.0,   -69.0,  -69.0,  -69.0,  -69.0,  -69.0,  -69.0,  -69.0,  -65.0,  -60.0,  -52.0,  -42.0, -20.0, 0.0]
        self.ay = [0.0, 0.0,   0.0,  0.0,  0.0,  0.0,  0.0, 5.0, 10.0, 20.0, 50.0, 70.0, 100.0, 110.0, 117.0, 120.0, 122.0, 124.0, 126.0, 126.0, 126.0, 126.0, 126.0, 126.0,  121.0,  116.0,   96.0,   46.0,   11.0,    9.0,    7.0,   0.0,    0.0,    0.0,    0.0,   0.0, 0.0]

        self.client = None
        self.controls = airsim.CarControls()
        self.state = State()

        self.plot = False

    def setupClient(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
    
    def setSpeed(self, speed):
        self.client.setCarSpeed(speed)

    def sendCommands(self, throttle=0.0, steering=0.0, brake=0.0):
        self.controls.throttle = throttle
        self.controls.brake = brake
        self.controls.steering = steering
        self.client.setCarControls(self.controls)

    def updateState(self):
        car_state = self.client.getCarState()
        self.state.x = car_state.kinematics_estimated.position.x_val - self.offx
        self.state.y = car_state.kinematics_estimated.position.y_val - self.offy

        self.state.v = np.linalg.norm(car_state.kinematics_estimated.linear_velocity.to_numpy_array())
        p, r, ye = airsim.to_eularian_angles(car_state.kinematics_estimated.orientation)
        self.state.yaw = ye

    def fitSpline(self):
        self.cx, self.cy, self.cyaw, self.ck, s = cubic_spline_planner.calc_spline_course(self.ax, self.ay, ds=1)
    
    def saveSpline(self, filename):
        with open(filename+'.txt', 'w') as spline_file:
            for idx in range(len(self.cx)):
                line = ",".join([str(self.cx[idx]), str(self.cy[idx]), str(self.cyaw[idx]), str(self.ck[idx])])
                spline_file.write(line + '\n')

    def readSpline(self, filename):
        with open(filename+'.txt', 'r') as spline_file:
            for line in spline_file:
                x, y, yaw, k = line.split(',')
                self.cx.append(float(x))
                self.cy.append(float(y))
                self.cyaw.append(float(yaw))
                self.ck.append(float(k))

    def pid_control(self, target, current):
        """
        Proportional control for the speed.

        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        return self.Kp * (target - current)

    def stanley_control(self, state):
        """
        Stanley steering control.

        :param state: (State object)
        :param self.cx: ([float])
        :param self.cy: ([float])
        :param self.cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = self.calc_target_index(state)

        #if last_target_idx >= current_target_idx:
        #    current_target_idx = last_target_idx

        # theta_e corrects the heading error
        theta_e = normalize_angle(self.cyaw[current_target_idx] - state.yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, state.v)
        # Steering control
        delta = theta_e + theta_d

        return delta, current_target_idx

    def calc_target_index(self, state):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param self.cx: [float]
        :param self.cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fx = state.x + self.wheel_base * np.cos(state.yaw)
        fy = state.y + self.wheel_base * np.sin(state.yaw)

        # Search nearest point index
        dx = [fx - icx for icx in self.cx]
        dy = [fy - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                        -np.sin(state.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle

    def stop_controller(self, signal, frame):
        print("\nCtrl+C received. Stopping car controller...")
        self.sendCommands(0.0, 0.0, 1.0)
        print("Done.")
        sys.exit(0)

    def stop(self):
        '''
        Apply brake and override car controller
        '''
        self.brake_override = True

    def resume(self):
        '''
        Resume normal path tracking
        '''
        self.brake_override = False

    def initialize(self):
        print("Initializing car controller...", end=' ')
        #car = CarController()
        self.setupClient()

        path = "small"
        #path = "large"
        
        if path == "small":
            filename = "NH_path_small"
        elif path == "large":
            filename = "NH_path_large"
        
        self.fitSpline()
        self.saveSpline(filename)
        #self.readSpline(filename)
        
        target_speed = 1.0  # [m/s]
        #max_simulation_time = 1000.0

        car_pose = self.client.simGetObjectPose('PlayerState_0')
        #self.state.x = car_state.kinematics_estimated.position.x_val
        #self.state.y = car_state.kinematics_estimated.position.y_val

        self.offx = car_pose.position.x_val + 72.0
        self.offy = car_pose.position.y_val - 15.0

        # Initial state
        self.state = State(x=self.offx, y=self.offy, yaw=np.radians(0.0), v=0.0)

        self.last_idx = len(self.cx) - 1

        const_throttle = 0.8
        brake = 0.0       

        print("Done.")

    def run(self):
        print("Running...")
        signal.signal(signal.SIGINT, self.stop_controller)

        x = []
        y = []
        yaw = []
        v = []

        while True:
            steering, target_idx = self.stanley_control(self.state)
            if not self.brake_override:
                speed = self.target_speed

                # Use adaptive lookahead to figure out how far we are from turn
                lookahead_window = int(speed)
                if target_idx + lookahead_window < len(self.cyaw):
                    lookahead_idx = target_idx + lookahead_window
                else:
                    lookahead_idx = (target_idx + lookahead_window) - len(self.cyaw)

                # Slow down to 7 m/s if yaw difference is over a threshold, indicating upcoming turn
                if abs((abs(self.cyaw[lookahead_idx]) - abs(self.cyaw[target_idx]))*180/np.pi) > 15.0:
                    speed = 7.0
                    
                self.setSpeed(speed)
                self.sendCommands(0.0, steering, 0.0)        
                self.updateState()        

                # Restart path tracking when close to end
                if target_idx > self.last_idx - 5:
                    target_idx = 0

                x.append(self.state.x)
                y.append(self.state.y)
                yaw.append(self.state.yaw)
                v.append(self.state.v)       

            else:
                self.sendCommands(0.0, steering, 1.0)

            if self.plot:  # pragma: no cover
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(self.cx, self.cy, ".r", label="course")
                plt.plot(x, y, "-b", label="trajectory")
                plt.plot(self.cx[target_idx], self.cy[target_idx], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("Speed[km/h]:" + str(self.state.v))
                plt.pause(0.001)

        # Stop
        assert last_idx >= target_idx, "Cannot reach goal"


if __name__ == '__main__':
    car = CarController()
    car.initialize()
    car.run()
