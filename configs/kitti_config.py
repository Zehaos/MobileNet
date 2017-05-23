from easydict import EasyDict as edict

# templete
config = edict()

config.IMG_HEIGHT = 375
config.IMG_WIDTH = 1242

config.FEA_HEIGHT = 12
config.FEA_WIDTH = 12

config.BATCH_SIZE = 1

config.EPSILON = 1e-8
config.LOSS_COEF_CLASS = 1
config.LOSS_COEF_CONF_POS = 1
config.LOSS_COEF_CONF_NEG = 1
config.LOSS_COEF_BBOX = 1




# config.ANCHOR_SHAPE = [[36., 37.], [366., 174.], [115., 59.],
#                       [162., 87.], [38., 90.], [258., 173.],
#                       [224., 108.], [78., 170.], [72., 43.]]
config.ANCHOR_SHAPE = [[100./config.IMG_WIDTH,100./config.IMG_HEIGHT],
                       [200./config.IMG_WIDTH,200./config.IMG_HEIGHT]]

config.NUM_ANCHORS = len(config.ANCHOR_SHAPE)
config.NUM_CLASSES = 3


