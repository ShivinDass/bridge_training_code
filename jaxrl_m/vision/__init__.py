from jaxrl_m.vision.resnet_v1 import resnetv1_configs
from jaxrl_m.vision.resnet_dec import resnetdec_configs

encoders = dict()
encoders.update(resnetv1_configs)

decoders = dict()
decoders.update(resnetdec_configs)