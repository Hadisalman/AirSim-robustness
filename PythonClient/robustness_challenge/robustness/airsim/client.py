from __future__ import print_function

from .utils import *
from .types import *

import msgpackrpc #install as admin: pip install msgpack-rpc-python
import numpy as np #pip install numpy
import msgpack
import time
import math
import logging

class Client:
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        if (ip == ""):
            ip = "127.0.0.1"
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value, pack_encoding = 'utf-8', unpack_encoding = 'utf-8')
        
    # -----------------------------------  Common vehicle APIs ---------------------------------------------
    def reset(self):
        self.client.call('reset')

    def ping(self):
        return self.client.call('ping')

    def getClientVersion(self):
        return 1 # sync with C++ client

    def getServerVersion(self):
        return self.client.call('getServerVersion')

    def getMinRequiredServerVersion(self):
        return 1 # sync with C++ client

    def getMinRequiredClientVersion(self):
        return self.client.call('getMinRequiredClientVersion')

    # basic flight control
    def enableApiControl(self, is_enabled, vehicle_name = ''):
        return self.client.call('enableApiControl', is_enabled, vehicle_name)

    def isApiControlEnabled(self, vehicle_name = ''):
        return self.client.call('isApiControlEnabled', vehicle_name)

    def simPause(self, is_paused):
        self.client.call('simPause', is_paused)

    def simIsPause(self):
        return self.client.call("simIsPaused")

    def simContinueForTime(self, seconds):
        self.client.call('simContinueForTime', seconds)

    def getHomeGeoPoint(self, vehicle_name = ''):
        return GeoPoint.from_msgpack(self.client.call('getHomeGeoPoint', vehicle_name))

    def confirmConnection(self):
        if self.ping():
            print("Connected!")
        else:
             print("Ping returned false!")
        server_ver = self.getServerVersion()
        client_ver = self.getClientVersion()
        server_min_ver = self.getMinRequiredServerVersion()
        client_min_ver = self.getMinRequiredClientVersion()
    
        ver_info = "Client Ver:" + str(client_ver) + " (Min Req: " + str(client_min_ver) + \
              "), Server Ver:" + str(server_ver) + " (Min Req: " + str(server_min_ver) + ")"

        if server_ver < server_min_ver:
            print(ver_info, file=sys.stderr)
            print("AirSim server is of older version and not supported by this client. Please upgrade!")
        elif client_ver < client_min_ver:
            print(ver_info, file=sys.stderr)
            print("AirSim client is of older version and not supported by this server. Please upgrade!")
        else:
            print(ver_info)
        print('')

    def simSwapTextures(self, tags, tex_id = 0, component_id = 0, material_id = 0):
        return self.client.call("simSwapTextures", tags, tex_id, component_id, material_id)

    # time-of-day control
    def simSetTimeOfDay(self, is_enabled, start_datetime = "", is_start_datetime_dst = False, celestial_clock_speed = 1, update_interval_secs = 60, move_sun = True):
        return self.client.call('simSetTimeOfDay', is_enabled, start_datetime, is_start_datetime_dst, celestial_clock_speed, update_interval_secs, move_sun)

    # weather
    def simEnableWeather(self, enable):
        return self.client.call('simEnableWeather', enable)

    def simSetWeatherParameter(self, param, val):
        return self.client.call('simSetWeatherParameter', param, val)

    #pedestrians
    
    def simPedestrianIsMoving(self, pedestrian_name):
        return self.client.call('simPedestrianIsMoving', pedestrian_name)

    def simPedestrianIsInCollision(self, pedestrian_name):
        return self.client.call('simPedestrianIsInCollision', pedestrian_name)

    def simPedestrianHasCollided(self, pedestrian_name):
        return self.client.call('simPedestrianHasCollided', pedestrian_name)
        
    def simGetPedestrianSpeed(self, pedestrian_name):
        return self.client.call('simGetPedestrianSpeed', pedestrian_name)
        
    def simStopPedestrian(self, pedestrian_name):
        return self.client.call('simStopPedestrian', pedestrian_name)
        
    def simMovePedestrianToGoal(self, pedestrian_name, pose, speed):
        return self.client.call('simMovePedestrianToGoal', pedestrian_name, pose, speed)

    
    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the ImageType members
    def simGetImage(self, camera_name, image_type, vehicle_name = ''):
        # todo: in future remove below, it's only for compatibility to pre v1.2
        camera_name = str(camera_name)

        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        result = self.client.call('simGetImage', camera_name, image_type, vehicle_name)
        if (result == "" or result == "\0"):
            return None
        return result

    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the ImageType members
    def simGetImages(self, requests, vehicle_name = ''):
        responses_raw = self.client.call('simGetImages', requests, vehicle_name)
        return [ImageResponse.from_msgpack(response_raw) for response_raw in responses_raw]

    # gets the static meshes in the unreal scene
    def simGetMeshPositionVertexBuffers(self):
        responses_raw = self.client.call('simGetMeshPositionVertexBuffers')
        return [MeshPositionVertexBuffersResponse.from_msgpack(response_raw) for response_raw in responses_raw]

    def simGetCollisionInfo(self, vehicle_name = ''):
        return CollisionInfo.from_msgpack(self.client.call('simGetCollisionInfo', vehicle_name))

    def simSetVehiclePose(self, pose, ignore_collison, vehicle_name = ''):
        self.client.call('simSetVehiclePose', pose, ignore_collison, vehicle_name)

    def simGetVehiclePose(self, vehicle_name = ''):
        pose = self.client.call('simGetVehiclePose', vehicle_name)
        return Pose.from_msgpack(pose)

    def simGetObjectPose(self, object_name):
        pose = self.client.call('simGetObjectPose', object_name)
        return Pose.from_msgpack(pose)

    def simSetObjectPose(self, object_name, pose, teleport = True):
        return self.client.call('simSetObjectPose', object_name, pose, teleport)

    def simListSceneObjects(self, name_regex = '.*'):
        return self.client.call('simListSceneObjects', name_regex)

    def simSetTextureFromUrl(self, object_name, url):
        return self.client.call('simSetTextureFromUrl', object_name, url)

    def simSetSegmentationObjectID(self, mesh_name, object_id, is_name_regex = False):
        return self.client.call('simSetSegmentationObjectID', mesh_name, object_id, is_name_regex)

    def simGetSegmentationObjectID(self, mesh_name):
        return self.client.call('simGetSegmentationObjectID', mesh_name)

    def simPrintLogMessage(self, message, message_param = "", severity = 0):
        return self.client.call('simPrintLogMessage', message, message_param, severity)
        
    def simSetDistortionParam(self, scenecap_name, param_name, value):
        self.client.call('simSetDistortionParam', scenecap_name, param_name, value)

    def simGetCameraInfo(self, camera_name, vehicle_name = ''):
        # TODO: below str() conversion is only needed for legacy reason and should be removed in future
        return CameraInfo.from_msgpack(self.client.call('simGetCameraInfo', str(camera_name), vehicle_name))

    def simSetCameraOrientation(self, camera_name, orientation, vehicle_name = ''):
        """
        - Control the orientation of a selected camera

        Args:
            camera_name (str): Name of the camera to be controlled
            orientation (airsim.Quaternion()): Quaternion representing the desired orientation of the camera
            vehicle_name (str, optional): Name of vehicle which the camera corresponds to
        """
        # TODO: below str() conversion is only needed for legacy reason and should be removed in future
        self.client.call('simSetCameraOrientation', str(camera_name), orientation, vehicle_name)
        
    def simSetCameraFov(self, camera_name, fov_degrees, vehicle_name = ''):
        """
        - Control the field of view of a selected camera

        Args:
            camera_name (str): Name of the camera to be controlled
            fov_degrees (float): Value of field of view in degrees
            vehicle_name (str, optional): Name of vehicle which the camera corresponds to
        """
        # TODO: below str() conversion is only needed for legacy reason and should be removed in future
        return self.client.call('simSetCameraFov', str(camera_name), fov_degrees, vehicle_name)

    def simGetGroundTruthKinematics(self, vehicle_name = ''):
        kinematics_state = self.client.call('simGetGroundTruthKinematics', vehicle_name)
        return KinematicsState.from_msgpack(kinematics_state)
    simGetGroundTruthKinematics.__annotations__ = {'return': KinematicsState}

    def simGetGroundTruthEnvironment(self, vehicle_name = ''):
        env_state = self.client.call('simGetGroundTruthEnvironment', vehicle_name)
        return EnvironmentState.from_msgpack(env_state)
    simGetGroundTruthEnvironment.__annotations__ = {'return': EnvironmentState}

    # sensor APIs
    def getImuData(self, imu_name = '', vehicle_name = ''):
        return ImuData.from_msgpack(self.client.call('getImuData', imu_name, vehicle_name))

    def getBarometerData(self, barometer_name = '', vehicle_name = ''):
        return BarometerData.from_msgpack(self.client.call('getBarometerData', barometer_name, vehicle_name))

    def getMagnetometerData(self, magnetometer_name = '', vehicle_name = ''):
        return MagnetometerData.from_msgpack(self.client.call('getMagnetometerData', magnetometer_name, vehicle_name))

    def getGpsData(self, gps_name = '', vehicle_name = ''):
        return GpsData.from_msgpack(self.client.call('getGpsData', gps_name, vehicle_name))

    def getDistanceSensorData(self, distance_sensor_name = '', vehicle_name = ''):
        return DistanceSensorData.from_msgpack(self.client.call('getDistanceSensorData', distance_sensor_name, vehicle_name))

    def getLidarData(self, lidar_name = '', vehicle_name = ''):
        return LidarData.from_msgpack(self.client.call('getLidarData', lidar_name, vehicle_name))
        
    def simGetLidarSegmentation(self, lidar_name = '', vehicle_name = ''):
        return self.client.call('simGetLidarSegmentation', lidar_name, vehicle_name)

    #  Plotting APIs
    def simFlushPersistentMarkers(self):
        """
        Clear any persistent markers - those plotted with setting is_persistent=True in the APIs below
        """
        self.client.call('simFlushPersistentMarkers')

    def simPlotPoints(self, points, color_rgba=[1.0, 0.0, 0.0, 1.0], size = 10.0, duration = -1.0, is_persistent = False):
        """
        Plot a list of 3D points in World NED frame
        
        Args:
            points (list[Vector3r]): List of Vector3r objects 
            color_rgba (list, optional): desired RGBA values from 0.0 to 1.0
            size (float, optional): Size of plotted point
            duration (float, optional): Duration (seconds) to plot for
            is_persistent (bool, optional): If set to True, the desired object will be plotted for infinite time.
        """
        self.client.call('simPlotPoints', points, color_rgba, size, duration, is_persistent)

    def simPlotLineStrip(self, points, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness = 5.0, duration = -1.0, is_persistent = False):
        """
        Plots a line strip in World NED frame, defined from points[0] to points[1], points[1] to points[2], ... , points[n-2] to points[n-1]
        
        Args:
            points (list[Vector3r]): List of 3D locations of line start and end points, specified as Vector3r objects
            color_rgba (list, optional): desired RGBA values from 0.0 to 1.0
            thickness (float, optional): Thickness of line
            duration (float, optional): Duration (seconds) to plot for
            is_persistent (bool, optional): If set to True, the desired object will be plotted for infinite time.
        """
        self.client.call('simPlotLineStrip', points, color_rgba, thickness, duration, is_persistent)

    def simPlotLineList(self, points, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness = 5.0, duration = -1.0, is_persistent = False):
        """
        Plots a line strip in World NED frame, defined from points[0] to points[1], points[2] to points[3], ... , points[n-2] to points[n-1]
        
        Args:
            points (list[Vector3r]): List of 3D locations of line start and end points, specified as Vector3r objects. Must be even
            color_rgba (list, optional): desired RGBA values from 0.0 to 1.0
            thickness (float, optional): Thickness of line
            duration (float, optional): Duration (seconds) to plot for
            is_persistent (bool, optional): If set to True, the desired object will be plotted for infinite time.
        """
        self.client.call('simPlotLineList', points, color_rgba, thickness, duration, is_persistent)

    def simPlotArrows(self, points_start, points_end, color_rgba=[1.0, 0.0, 0.0, 1.0], thickness = 5.0, arrow_size = 2.0, duration = -1.0, is_persistent = False):
        """
        Plots a list of arrows in World NED frame, defined from points_start[0] to points_end[0], points_start[1] to points_end[1], ... , points_start[n-1] to points_end[n-1]

        Args:
            points_start (list[Vector3r]): List of 3D start positions of arrow start positions, specified as Vector3r objects
            points_end (list[Vector3r]): List of 3D end positions of arrow start positions, specified as Vector3r objects
            color_rgba (list, optional): desired RGBA values from 0.0 to 1.0
            thickness (float, optional): Thickness of line
            arrow_size (float, optional): Size of arrow head
            duration (float, optional): Duration (seconds) to plot for
            is_persistent (bool, optional): If set to True, the desired object will be plotted for infinite time.
        """
        self.client.call('simPlotArrows', points_start, points_end, color_rgba, thickness, arrow_size, duration, is_persistent)


    def simPlotStrings(self, strings, positions, scale = 5, color_rgba=[1.0, 0.0, 0.0, 1.0], duration = -1.0):
        """
        Plots a list of strings at desired positions in World NED frame. 

        Args:
            strings (list[String], optional): List of strings to plot
            positions (list[Vector3r]): List of positions where the strings should be plotted. Should be in one-to-one correspondence with the strings' list
            scale (float, optional): Font scale of transform name
            color_rgba (list, optional): desired RGBA values from 0.0 to 1.0
            duration (float, optional): Duration (seconds) to plot for
        """
        self.client.call('simPlotStrings', strings, positions, scale, color_rgba, duration)

    def simPlotTransforms(self, poses, scale = 5.0, thickness = 5.0, duration = -1.0, is_persistent = False):
        """
        Plots a list of transforms in World NED frame. 

        Args:
            poses (list[Pose]): List of Pose objects representing the transforms to plot
            scale (float, optional): Length of transforms' axes
            thickness (float, optional): Thickness of transforms' axes 
            duration (float, optional): Duration (seconds) to plot for
            is_persistent (bool, optional): If set to True, the desired object will be plotted for infinite time.
        """
        self.client.call('simPlotTransforms', poses, scale, thickness, duration, is_persistent)

    def simPlotTransformsWithNames(self, poses, names, tf_scale = 5.0, tf_thickness = 5.0, text_scale = 10.0, text_color_rgba = [1.0, 0.0, 0.0, 1.0], duration = -1.0):
        """
        Plots a list of transforms with their names in World NED frame. 
        
        Args:
            poses (list[Pose]): List of Pose objects representing the transforms to plot
            names (list[string]): List of strings with one-to-one correspondence to list of poses
            tf_scale (float, optional): Length of transforms' axes
            tf_thickness (float, optional): Thickness of transforms' axes 
            text_scale (float, optional): Font scale of transform name
            text_color_rgba (list, optional): desired RGBA values from 0.0 to 1.0 for the transform name
            duration (float, optional): Duration (seconds) to plot for
        """
        self.client.call('simPlotTransformsWithNames', poses, names, tf_scale, tf_thickness, text_scale, text_color_rgba, duration)

    def cancelLastTask(self, vehicle_name = ''):
        self.client.call('cancelLastTask', vehicle_name)
    def waitOnLastTask(self, timeout_sec = float('nan')):
        return self.client.call('waitOnLastTask', timeout_sec)


# -----------------------------------  Car APIs ---------------------------------------------
class CarClient(Client, object):
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        super(CarClient, self).__init__(ip, port, timeout_value)

    def setCarControls(self, controls, vehicle_name = ''):
        self.client.call('setCarControls', controls, vehicle_name)

    def setCarTargetSpeed(self, speed, vehicle_name = ''):
        self.client.call('setCarTargetSpeed', speed, vehicle_name)

    def getCarState(self, vehicle_name = ''):
        state_raw = self.client.call('getCarState', vehicle_name)
        return CarState.from_msgpack(state_raw)

    def getCarControls(self, vehicle_name=''):
        controls_raw = self.client.call('getCarControls', vehicle_name)
        return CarControls.from_msgpack(controls_raw)