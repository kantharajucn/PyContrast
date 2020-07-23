import torch.nn as nn
import torch


dependencies = ['torch']

def _wrap_module(model):
    class WrappedModel(nn.Module):
        def __init__(self):
            super(WrappedModel, self).__init__()
            self.module = model  # that I actually define.

        def forward(self, x):
            return self.module(x)
    return WrappedModel


def INsDis(pretrained=False, **kwargs):
    """
    Unsupervised Feature Learning via Non-parameteric Instance Discrimination
    :param pretrained:
    :param kwargs:
    :return:
    """

    from networks.build_backbone import RGBSingleHead
    model = _wrap_module(RGBSingleHead)
    model = model(**kwargs)
    url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth?dl=1'
    state_dict = torch.hub.load_state_dict_from_url(url)
    if pretrained:
        model.load_state_dict(state_dict["modal"])
    return model

def CMC(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """

    from networks.build_backbone import CMCSingleHead
    model = _wrap_module(CMCSingleHead)
    model = model(**kwargs)
    url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACYcqgM-lcG3__QIbxuM2Koa/CMC.pth?dl=1'
    state_dict = torch.hub.load_state_dict_from_url(url)
    if pretrained:
        model.load_state_dict(state_dict["modal"])
    return model

def MoCo(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """

    from networks.build_backbone import RGBSingleHead
    model = _wrap_module(RGBSingleHead)
    model = model(**kwargs)
    url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAB53yJAYuCrOFluygBsVKOOa/MoCo.pth?dl=1'
    state_dict = torch.hub.load_state_dict_from_url(url)
    if pretrained:
        model.load_state_dict(state_dict["modal"])
    return model

def MoCo(pretrained=False, **kwargs):
    """
    Momentum Contrast for Unsupervised Visual Representation Learning
    :param pretrained:
    :param kwargs:
    :return:
    """

    from networks.build_backbone import RGBSingleHead
    model = _wrap_module(RGBSingleHead)
    model = model(**kwargs)
    url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAB53yJAYuCrOFluygBsVKOOa/MoCo.pth?dl=1'
    state_dict = torch.hub.load_state_dict_from_url(url)
    if pretrained:
        model.load_state_dict(state_dict["modal"])
    return model

def MoCoV2(pretrained=False, **kwargs):
    """
    Improved Baselines with Momentum Contrastive Learning
    :param pretrained:
    :param kwargs:
    :return:
    """

    from networks.build_backbone import RGBSingleHead
    model = _wrap_module(RGBSingleHead)
    model = model(**kwargs)
    url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AABaYuKiiZFYowa31yKeGGOQa/MoCov2.pth?dl=2'
    state_dict = torch.hub.load_state_dict_from_url(url)
    if pretrained:
        model.load_state_dict(state_dict["modal"])
    return model

def PIRL(pretrained=False, **kwargs):
    """
    Self-Supervised Learning of Pretext-Invariant Representations
    :param pretrained:
    :param kwargs:
    :return:
    """

    from networks.build_backbone import RGBMultiHeads
    model = _wrap_module(RGBMultiHeads)
    model = model(**kwargs)
    url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth?dl=1'
    state_dict = torch.hub.load_state_dict_from_url(url)
    if pretrained:
        model.load_state_dict(state_dict["modal"])
    return model

def InfoMin(pretrained=False, **kwargs):
    """
    What Makes for Good Views for Contrastive Learning?
    :param pretrained:
    :param kwargs:
    :return:
    """

    from networks.build_backbone import RGBMultiHeads
    model = _wrap_module(RGBMultiHeads)
    model = model(**kwargs)
    url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth?dl=0'
    state_dict = torch.hub.load_state_dict_from_url(url)
    if pretrained:
        model.load_state_dict(state_dict["modal"])
    return model

