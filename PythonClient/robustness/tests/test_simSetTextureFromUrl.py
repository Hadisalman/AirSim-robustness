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


# url = 'https://www.google.com/imgres?imgurl=http%3A%2F%2Fs3.amazonaws.com%2Fimages.seroundtable.com%2Fgoogle-replace-old-1547730448.gif&imgrefurl=https%3A%2F%2Fwww.seroundtable.com%2Fgoogle-fetch-as-google-with-url-inspection-tool-26972.html&tbnid=dcstw_I4SPjjyM&vet=12ahUKEwj90dKx0pbpAhU0JH0KHdkrC7QQMygCegUIARCDAg..i&docid=3LJ5N7tXp8m4GM&w=640&h=300&q=image%20with%20url&ved=2ahUKEwj90dKx0pbpAhU0JH0KHdkrC7QQMygCegUIARCDAg'
url = 'https://www.nasa.gov/sites/default/files/styles/full_width_feature/public/thumbnails/image/48905773692_7980d433f2_o.jpg'
client.simSetTextureFromUrl(object_name, url)