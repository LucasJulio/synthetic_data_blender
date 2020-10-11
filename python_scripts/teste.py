import os
import numpy as np
from bpy import ops, context, data

ENVIRONMENT_ROT_DIVISIONS=20
PROB_SUPP_LIGHT1=0.2
PROB_SUPP_LIGHT2=0.2
PROB_HIDE_OBJ=0.1
PROB_HIDE_TEXT=0.6

MAX_ANGLE_PITCH_CAMERA=26
MAX_DELTA_X_CAMERA=0.01
MAX_DELTA_Y_CAMERA=0.01

def position_camera(delta_x=None, delta_y=None):
    handler=data.objects['Handler camera']

    if delta_x is None:
        delta_x=(np.random.random()*2*MAX_DELTA_X_CAMERA)-MAX_DELTA_X_CAMERA

    if delta_y is None:
        delta_y=(np.random.random()*2*MAX_DELTA_Y_CAMERA)-MAX_DELTA_Y_CAMERA



    handler.delta_location[0]=delta_x  #X or Pich
    handler.delta_location[1]=delta_y  #Z or Yaw

position_camera()
