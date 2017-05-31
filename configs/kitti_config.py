from easydict import EasyDict as edict
import numpy as np

config = edict()

config.IMG_HEIGHT = 375
config.IMG_WIDTH = 1242

# TODO(shizehao): infer fea shape in run time
config.FEA_HEIGHT = 12
config.FEA_WIDTH = 39

config.EPSILON = 1e-16

config.LOSS_COEF_BBOX = 5.0
config.LOSS_COEF_CONF_POS = 75.0
config.LOSS_COEF_CONF_NEG = 100.0
config.LOSS_COEF_CLASS = 1.0

config.EXP_THRESH = 1.0

config.RBG_MEANS = np.array([[[ 123.68, 116.779, 103.939]]])


def set_anchors(H, W):
  B = 9
  shape = np.array(
          [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
           [ 162.,  87.], [  38.,  90.], [ 258., 173.],
           [ 224., 108.], [  78., 170.], [  72.,  43.]])

  # # scale
  # shape[:, 0] = shape[:, 0] / config.IMG_HEIGHT
  # shape[:, 1] = shape[:, 1] / config.IMG_WIDTH

  anchor_shapes = np.reshape(
      [shape] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(config.IMG_WIDTH)/(W+1)]*H*B),
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(config.IMG_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors

config.ANCHOR_SHAPE = set_anchors(config.FEA_HEIGHT, config.FEA_WIDTH)

config.NUM_ANCHORS = 9
config.NUM_CLASSES = 3
config.ANCHORS = config.NUM_ANCHORS * config.FEA_HEIGHT * config.FEA_WIDTH

config.PLOT_PROB_THRESH = 0.4
config.NMS_THRESH = 0.4
config.PROB_THRESH = 0.005
config.TOP_N_DETECTION = 64






