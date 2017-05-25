from easydict import EasyDict as edict
import numpy as np

config = edict()

config.IMG_HEIGHT = 1242
config.IMG_WIDTH = 375

# TODO(shizehao): infer fea shape in run time
config.FEA_HEIGHT = 39
config.FEA_WIDTH = 12

config.EPSILON = 1e-8

config.LOSS_COEF_BBOX = 5.0
config.LOSS_COEF_CONF_POS = 75.0
config.LOSS_COEF_CONF_NEG = 100.0
config.LOSS_COEF_CLASS = 1.0

config.EXP_THRESH = 1.0


def set_anchor_shape(anchor_shape):
  anchor_shape[:][0] = anchor_shape[:][0] / config.IMG_HEIGHT
  anchor_shape[:][1] = anchor_shape[:][1] / config.IMG_WIDTH
  return anchor_shape


config.ANCHOR_SHAPE = set_anchor_shape(np.array([[36., 37.], [366., 174.], [115., 59.],
                                                 [162., 87.], [38., 90.], [258., 173.],
                                                 [224., 108.], [78., 170.], [72., 43.]], dtype=np.float32))

config.NUM_ANCHORS = len(config.ANCHOR_SHAPE)
config.NUM_CLASSES = 3
config.ANCHORS = config.NUM_ANCHORS * config.FEA_HEIGHT * config.FEA_WIDTH



