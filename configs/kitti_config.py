from easydict import EasyDict as edict

# templete

configs = edict()
configs.EPSILON = 1e-8
configs.LOSS_COEF_CLASS = 1
configs.LOSS_COEF_CONF_POS = 1
configs.LOSS_COEF_CONF_NEG = 1
configs.LOSS_COEF_BBOX = 1
configs.BATCH_SIZE = 1
configs.ANCHORS = 1