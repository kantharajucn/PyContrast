import torch.nn as nn
import torch


dependencies = ['torch']



def InsDis(pretrained=False, **kwargs):
    """
    Unsupervised Feature Learning via Non-parameteric Instance Discrimination
    :param pretrained:
    :param kwargs:
    :return:
    """

    from pycontrast.networks.build_backbone import RGBSingleHead
    model = RGBSingleHead(*kwargs)

    if pretrained:
        model = nn.DataParallel(model)
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth?dl=1'
        state_dict = torch.hub.load_state_dict_from_url(url)
        print(state_dict.keys())
        model.load_state_dict(state_dict["model"])
    return model

def CMC(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """

    from pycontrast.networks.build_backbone import CMCSingleHead
    model = CMCSingleHead(**kwargs)

    if pretrained:
        model = nn.DataParallel(model)
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACYcqgM-lcG3__QIbxuM2Koa/CMC.pth?dl=1'
        state_dict = torch.hub.load_state_dict_from_url(url)
        print(state_dict.keys())
        model.load_state_dict(state_dict["model"])
    return model

def MoCo(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """

    from pycontrast.networks.build_backbone import RGBSingleHead
    model = RGBSingleHead(**kwargs)

    if pretrained:
        model = nn.DataParallel(model)
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAB53yJAYuCrOFluygBsVKOOa/MoCo.pth?dl=1'
        state_dict = torch.hub.load_state_dict_from_url(url)
        print(state_dict.keys())
        model.load_state_dict(state_dict["model"])
    return model


def MoCoV2(pretrained=False, **kwargs):
    """
    Improved Baselines with Momentum Contrastive Learning
    :param pretrained:
    :param kwargs:
    :return:
    """

    from pycontrast.networks.build_backbone import RGBSingleHead
    model = RGBSingleHead(**kwargs)

    if pretrained:
        model = nn.DataParallel(model)
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AABaYuKiiZFYowa31yKeGGOQa/MoCov2.pth?dl=1'
        state_dict = torch.hub.load_state_dict_from_url(url)
        print(state_dict.keys())
        model.load_state_dict(state_dict["model_ema"])
    return model

def PIRL(pretrained=False, **kwargs):
    """
    Self-Supervised Learning of Pretext-Invariant Representations
    :param pretrained:
    :param kwargs:
    :return:
    """

    from pycontrast.networks.build_backbone import RGBSingleHead
    model = RGBSingleHead(**kwargs)

    if pretrained:
        model = nn.DataParallel(model)
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth?dl=1'
        state_dict = torch.hub.load_state_dict_from_url(url)
        print(state_dict.keys())
        model.load_state_dict(state_dict["model"])
    return model

def InfoMin(pretrained=False, **kwargs):
    """
    What Makes for Good Views for Contrastive Learning?
    :param pretrained:
    :param kwargs:
    :return:
    """
    from pycontrast.networks.build_backbone import RGBSingleHead
    model = RGBSingleHead(**kwargs)

    if pretrained:
        model = nn.DataParallel(model)
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth?dl=1'
        state_dict = torch.hub.load_state_dict_from_url(url)
        print(state_dict.keys())
        model.load_state_dict(state_dict["model"])
    return model

