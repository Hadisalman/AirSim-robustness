import asyncio
from robustness import airsim
from IPython import embed
import copy
import numpy as np
import time

client = airsim.CarClient()
client.confirmConnection()

ped_object_name = 'Adv_Ped2'
scene_objs = client.simListSceneObjects()
print(f'Pedestrian in the scene? {ped_object_name in scene_objs}')
assert ped_object_name in scene_objs

# while True:
#     print(f'Is colliding: {client.simPedestrianIsInCollision(ped_object_name)}')
#     print(f'Has collided: {client.simPedestrianHasCollided(ped_object_name)}')
#     time.sleep(1)


# client.simPedestrianIsMoving(ped_object_name)
client.simGetPedestrianSpeed(ped_object_name)


client.simPedestrianIsMoving(ped_object_name)
client.simStopPedestrian(ped_object_name)

pose = client.simGetObjectPose(ped_object_name)

for i in range(10):
    if i%2==0:
        goal_pose = airsim.Pose(copy.copy(pose.position))
        goal_pose.position.x_val -= 4 + np.random.rand()/1000
        goal_pose.position.y_val += np.random.rand()/1000

        client.simMovePedestrianToGoal(ped_object_name, goal_pose, 3)
    else:
        goal_pose = airsim.Pose(copy.copy(pose.position))
        goal_pose.position.x_val -= np.random.rand()/1000
        goal_pose.position.y_val += np.random.rand()/1000

        client.simMovePedestrianToGoal(ped_object_name, goal_pose, 3)
    time.sleep(5)


# async def main():
#     print('hello')    
#     await asyncio.sleep(1)
#     print('world')

# asyncio.run(main())


#  async def simQueuePedestrianGoals(self, pedestrian_name, goals_and_speeds):
#         while len(goals_and_speeds) > 0:
#             next = goals_and_speeds.pop(0)
#             simMovePedestrianToGoal(pedestrian_name, next[0], next[1], next[2], next[3])
#             while simPedestrianIsMoving(pedestrian_name):
#                 await asyncio.sleep(0.1)