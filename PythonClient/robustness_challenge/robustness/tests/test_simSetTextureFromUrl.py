import asyncio
from robustness import airsim
from IPython import embed
import copy
import numpy as np
import time

client = airsim.Client()
client.confirmConnection()

object_name = 'Adv_Plate'
# object_name = 'Adv_Plate_Blueprint'
scene_objs = client.simListSceneObjects()
print(f'Object in the scene? {object_name in scene_objs}')
assert object_name in scene_objs

url = 'https://www.dropbox.com/s/yfbc6xo9m69wwfu/dog.jpg?raw=1'
# url = 'https://s01.sgp1.cdn.digitaloceanspaces.com/article/131928-mxiccwtarv-1575034997.jpg'
# url = 'https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/48905773692_7980d433f2_o.jpg'
client.simSetTextureFromUrl(object_name, url)