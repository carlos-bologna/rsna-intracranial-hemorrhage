import torch
import torch.nn as nn
from torchvision.models import resnet50
from common.blocks import ConvBlock, LinearAttentionBlock, ProjectorBlock

class ResNet50Attention(nn.Module):
    
    def __init__(self, num_classes, attention=True, pretrained=True, normalize_attn=True):
        super(ResNet50Attention, self).__init__()
        self.attention = attention
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        resnet50_model = resnet50(pretrained=pretrained)
        
        layers = [l for l in resnet50_model.children()]
        
        self.conv1 = layers[0]
        self.bn1 = layers[1]
        self.relu = layers[2]
        self.maxpool = layers[3]

        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.avgpool = layers[8]

        if self.attention:
            self.fc = nn.Linear(in_features=1792, out_features=self.num_classes, bias=True)
        else:
            self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
        
        if self.attention:
            self.projector1 = ProjectorBlock(2048, 256)
            self.projector2 = ProjectorBlock(2048, 512)
            self.projector3 = ProjectorBlock(2048, 1024)
            self.attn1 = LinearAttentionBlock(in_features=256, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=1024, normalize_attn=normalize_attn)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.layer4(l3)
        g = self.avgpool(x)
        #x = torch.flatten(g, 1)
        
        # pay attention
        if self.attention:
            p1 = self.projector1(g)
            c1, g1 = self.attn1(l1, p1)
            
            p2 = self.projector2(g)
            c2, g2 = self.attn2(l2, p2)
            
            p3 = self.projector3(g)
            c3, g3 = self.attn3(l3, p3)
            
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            x = self.fc(g) # batch_sizexnum_classes
            
        else:
        
            x = self.fc(g)

        return x
    
class ResNet50AttentionMultiTask(nn.Module):
    
    def __init__(self, num_classes, attention=True, pretrained=True, normalize_attn=True):
        super(ResNet50AttentionMultiTask, self).__init__()
        self.attention = attention
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        resnet50_model = resnet50(pretrained=pretrained)
        
        layers = [l for l in resnet50_model.children()]
        
        self.conv1 = layers[0]
        self.bn1 = layers[1]
        self.relu = layers[2]
        self.maxpool = layers[3]

        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.avgpool = layers[8]

        if self.attention:
            self.fc = nn.Linear(in_features=1792, out_features=self.num_classes, bias=True)
            self.fc_z = nn.Linear(in_features=1792, out_features=1, bias=True)
        else:
            self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
            self.fc_z = nn.Linear(in_features=2048, out_features=1, bias=True)

        if self.attention:
            self.projector1 = ProjectorBlock(2048, 256)
            self.projector2 = ProjectorBlock(2048, 512)
            self.projector3 = ProjectorBlock(2048, 1024)
            self.attn1 = LinearAttentionBlock(in_features=256, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=1024, normalize_attn=normalize_attn)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.layer4(l3)
        g = self.avgpool(x)
        #x = torch.flatten(g, 1)
        
        # pay attention
        if self.attention:
            p1 = self.projector1(g)
            c1, g1 = self.attn1(l1, p1)
            
            p2 = self.projector2(g)
            c2, g2 = self.attn2(l2, p2)
            
            p3 = self.projector3(g)
            c3, g3 = self.attn3(l3, p3)
            
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            x = self.fc(g) # batch_sizexnum_classes
            z = self.fc_z(g)
            
        else:
            x = self.fc(g)
            z = self.fc_z(g)


        return (x, z)
    
class ResNet50AttentionMultiTaskV2(nn.Module):
    
    def __init__(self, num_classes, attention=True, pretrained=True, normalize_attn=True):
        super(ResNet50AttentionMultiTaskV2, self).__init__()
        self.attention = attention
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        resnet50_model = resnet50(pretrained=pretrained)
        
        layers = [l for l in resnet50_model.children()]
        
        self.conv1 = layers[0]
        self.bn1 = layers[1]
        self.relu = layers[2]
        self.maxpool = layers[3]

        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.avgpool = layers[8]
        self.dropout = nn.Dropout(p=0.4)

        if self.attention:
            self.fc = nn.Linear(in_features=1792, out_features=self.num_classes, bias=True)
            self.fc_z = nn.Linear(in_features=1792, out_features=1, bias=True)
        else:
            self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
            self.fc_z = nn.Linear(in_features=2048, out_features=1, bias=True)

        if self.attention:
            self.projector1 = ProjectorBlock(2048, 256)
            self.projector2 = ProjectorBlock(2048, 512)
            self.projector3 = ProjectorBlock(2048, 1024)
            self.attn1 = LinearAttentionBlock(in_features=256, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=1024, normalize_attn=normalize_attn)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.layer4(l3)
        g = self.avgpool(x)
        #x = torch.flatten(g, 1)
        
        # pay attention
        if self.attention:
            p1 = self.projector1(g)
            c1, g1 = self.attn1(l1, p1)
            
            p2 = self.projector2(g)
            c2, g2 = self.attn2(l2, p2)
            
            p3 = self.projector3(g)
            c3, g3 = self.attn3(l3, p3)
            
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            o = self.relu(g)
            o = self.relu(o)
            o = self.dropout(o)
            o = self.relu(o)
            x = self.fc(o) # batch_sizexnum_classes
            z = self.fc_z(o)
            
        else:
        
            o = self.relu(g)
            o = self.relu(o)
            o = self.dropout(o)
            o = self.relu(o)
            x = self.fc(o)
            z = self.fc_z(o)


        return (x, z)
    
## Multitask DenseNet121
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from .utils import load_state_dict_from_url


__all__ = ['MultiTaskDenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    state_dict['aux_classifier.weight'] = state_dict['classifier.weight']
    state_dict['aux_classifier.bias'] = state_dict['classifier.bias']
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = MultiTaskDenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121multitask(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from with multitask learning
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)

class MultiTaskDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(MultiTaskDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.aux_classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = F.relu(out, inplace=True)
        out1 = self.classifier(out)
        out2 = self.aux_classifier(out)
        return (out1, out2)
    
class MultiTaskDenseNetV2(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(MultiTaskDenseNetV2, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.aux_classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = F.relu(out, inplace=True)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, p=0.4, training=True, inplace=True)
        out = F.relu(out, inplace=True)
        out1 = self.classifier(out)
        out2 = self.aux_classifier(out)
        return (out1, out2)

def _densenetV2(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = MultiTaskDenseNetV2(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121multitaskV2(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from with multitask learning
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenetV2('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)