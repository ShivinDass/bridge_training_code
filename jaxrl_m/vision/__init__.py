from jaxrl_m.vision.resnet_v1 import resnetv1_configs
from jaxrl_m.vision.resnet_dec import resnetdec_configs
from jaxrl_m.vision.pretrained_resnet.resnet import pretrained_resnet_configs

encoders = dict()
encoders.update(resnetv1_configs)
encoders.update(pretrained_resnet_configs)

decoders = dict()
decoders.update(resnetdec_configs)