import jax
import jax.numpy as jnp
import flax.linen as nn
import functools
from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
import h5py
import warnings
from . import ops

###############################################################################

from tqdm import tqdm
import requests
import os
import tempfile


def download(ckpt_dir, url):
    name = url[url.rfind('/') + 1 : url.rfind('?')]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'flaxmodels')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file

###############################################################################


URLS = {'resnet18': 'https://www.dropbox.com/s/wx3vt76s5gpdcw5/resnet18_weights.h5?dl=1',
        'resnet34': 'https://www.dropbox.com/s/rnqn2x6trnztg4c/resnet34_weights.h5?dl=1',
        'resnet50': 'https://www.dropbox.com/s/fcc8iii38ezvqog/resnet50_weights.h5?dl=1',
        'resnet101': 'https://www.dropbox.com/s/hgtnk586pnz0xug/resnet101_weights.h5?dl=1',
        'resnet152': 'https://www.dropbox.com/s/tvi28uwiy54mcfr/resnet152_weights.h5?dl=1'}

LAYERS = {'resnet18': [2, 2, 2, 2],
          'resnet34': [3, 4, 6, 3],
          'resnet50': [3, 4, 6, 3],
          'resnet101': [3, 4, 23, 3],
          'resnet152': [3, 8, 36, 3]}


class BasicBlock(nn.Module):
    """
    Basic Block.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: h5py.Group=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    dtype: str='float32'
    freezed_BN: bool=False

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        if self.freezed_BN:
            bn = ops.freezed_batch_norm
        else:
            bn = ops.batch_norm

        residual = x 

        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(2, 2) if self.downsample else (1, 1),
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = bn(x,
               train=train,
               epsilon=1e-05,
               momentum=0.1,
               params=None if self.param_dict is None else self.param_dict['bn1'],
               dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = bn(x,
               train=train,
               epsilon=1e-05,
               momentum=0.1,
               params=None if self.param_dict is None else self.param_dict['bn2'],
               dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features, 
                               kernel_size=(1, 1), 
                               strides=(2, 2), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = bn(residual,
                          train=train,
                          epsilon=1e-05,
                          momentum=0.1,
                          params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                          dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        expansion (int): Factor to multiply number of output channels with.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: Any=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    expansion: int=4
    dtype: str='float32'
    freezed_BN: bool=False

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Bottleneck.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        if self.freezed_BN:
            bn = ops.freezed_batch_norm
        else:
            bn = ops.batch_norm

        residual = x 
        
        x = nn.Conv(features=self.features, 
                    kernel_size=(1, 1), 
                    strides=(1, 1),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = bn(x,
               train=train,
               epsilon=1e-05,
               momentum=0.1,
               params=None if self.param_dict is None else self.param_dict['bn1'],
               dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=(3, 3), 
                    strides=(2, 2) if self.downsample and self.stride else (1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)
        
        x = bn(x,
               train=train,
               epsilon=1e-05,
               momentum=0.1,
               params=None if self.param_dict is None else self.param_dict['bn2'],
               dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features * self.expansion, 
                    kernel_size=(1, 1), 
                    strides=(1, 1), 
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv3']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = bn(x,
               train=train,
               epsilon=1e-05,
               momentum=0.1,
               params=None if self.param_dict is None else self.param_dict['bn3'],
               dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features * self.expansion, 
                               kernel_size=(1, 1), 
                               strides=(2, 2) if self.stride else (1, 1), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = bn(residual,
                          train=train,
                          epsilon=1e-05,
                          momentum=0.1,
                          params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                          dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class ResNet(nn.Module):
    """
    ResNet.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str): 
            Which ResNet model to use:
                - 'resnet18'
                - 'resnet34'
                - 'resnet50'
                - 'resnet101'
                - 'resnet152'
        num_classes (int):
            Number of classes.
        block (nn.Module):
            Type of residual block:
                - BasicBlock
                - Bottleneck
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """
    output: str='softmax'
    pretrained: str='imagenet'
    normalize: bool=True
    architecture: str='resnet18'
    num_classes: int=1000
    block: nn.Module=BasicBlock
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    ckpt_dir: str=None
    dtype: str='float32'
    freezed_BN: bool=False

    def setup(self):

        self.param_dict = None
        if self.pretrained == 'imagenet':
            ckpt_file = download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(ckpt_file, 'r')

    # NOTE: train=False because we're freezing BN running stats
    @nn.compact
    def __call__(self, x, train=False):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            If output == 'logits' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """

        x = x.astype(jnp.float32)
        if len(x.shape) == 4:
            N, H, W, C = x.shape
            if H != 256:
                x = jax.image.resize(x, (N, 256, 256, C), method='bilinear', antialias=False)
        elif len(x.shape) == 3:
            H, W, C = x.shape
            if H != 256:
                x = jax.image.resize(x, (256, 256, C), method='bilinear', antialias=False)


        if self.normalize:
            x = x / 255.0
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(x.dtype)
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(x.dtype)
            x = (x - mean) / std

        if self.pretrained == 'imagenet':
            if self.num_classes != 1000:
                warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} '
                                'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ', UserWarning)
            num_classes = 1000
        else:
            num_classes = self.num_classes
 
        act = {}

        x = nn.Conv(features=64, 
                    kernel_size=(7, 7),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']),
                    strides=(2, 2), 
                    padding=((3, 3), (3, 3)),
                    use_bias=False,
                    dtype=self.dtype)(x)
        act['conv1'] = x

        if self.freezed_BN:
            x = ops.freezed_batch_norm(x,
                                       epsilon=1e-05,
                                       momentum=0.1,
                                       params=None if self.param_dict is None else self.param_dict['bn1'],
                                       dtype=self.dtype)
        else:
            x = ops.batch_norm(x,
                               train=train,
                               epsilon=1e-05,
                               momentum=0.1,
                               params=None if self.param_dict is None else self.param_dict['bn1'],
                               dtype=self.dtype)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer 1
        down = self.block.__name__ == 'Bottleneck'
        for i in range(LAYERS[self.architecture][0]):
            params = None if self.param_dict is None else self.param_dict['layer1'][f'block{i}']
            x = self.block(features=64,
                           kernel_size=(3, 3),
                           downsample=i == 0 and down,
                           stride=i != 0,
                           param_dict=params,
                           block_name=f'block1_{i}',
                           dtype=self.dtype,
                           freezed_BN=self.freezed_BN)(x, act, train)
        
        # Layer 2
        for i in range(LAYERS[self.architecture][1]):
            params = None if self.param_dict is None else self.param_dict['layer2'][f'block{i}']
            x = self.block(features=128,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block2_{i}',
                           dtype=self.dtype,
                           freezed_BN=self.freezed_BN)(x, act, train)

        # Layer 3
        for i in range(LAYERS[self.architecture][2]):
            params = None if self.param_dict is None else self.param_dict['layer3'][f'block{i}']
            x = self.block(features=256,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block3_{i}',
                           dtype=self.dtype,
                           freezed_BN=self.freezed_BN)(x, act, train)

        # Layer 4
        for i in range(LAYERS[self.architecture][3]):
            params = None if self.param_dict is None else self.param_dict['layer4'][f'block{i}']
            x = self.block(features=512,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block4_{i}',
                           dtype=self.dtype,
                           freezed_BN=self.freezed_BN)(x, act, train)

        # Classifier
        x = jnp.mean(x, axis=(-3, -2))
        if self.output == 'embs':
            return x
        
        # NOTE: should not be using these for our model
        # x = nn.Dense(features=num_classes,
        #              kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['fc']['weight']), 
        #              bias_init=self.bias_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['fc']['bias']),
        #              dtype=self.dtype)(x)
        # act['fc'] = x
        
        # if self.output == 'softmax':
        #     return nn.softmax(x)
        # if self.output == 'log_softmax':
        #     return nn.log_softmax(x)
        # if self.output == 'activations':
        #     return act
        # return x


def ResNet18(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32',
             freezed_BN=False):
    """
    Implementation of the ResNet18 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet18',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype,
                  freezed_BN=freezed_BN)


def ResNet34(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32',
             freezed_BN=False):
    """
    Implementation of the ResNet34 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet34',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype,
                  freezed_BN=freezed_BN)


def ResNet50(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32',
             freezed_BN=False):
    """
    Implementation of the ResNet50 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet50',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype,
                  freezed_BN=freezed_BN)


# NOTE: this ckpt_dir is just to accomodate docker
pretrained_resnet_configs = {
    "pretrained_resnet18": functools.partial(
        ResNet18,
        output='embs',
        ckpt_dir='./pretrained_resnet'
    ),
    "pretrained_resnet34": functools.partial(
        ResNet34,
        output='embs',
        ckpt_dir='./pretrained_resnet'
    ),
}
