from collections import OrderedDict

import torch.nn as nn
import torch


dependencies = ['torch']


def _load_encoder_weights(model, modal, state):
    """load pre-trained weights for encoder

    Args:
      model: pretrained encoder, should be frozen
    """
    if modal == 'RGB':
        # Unimodal (RGB) case
        encoder_state_dict = OrderedDict()
        for k, v in state.items():
            k = k.replace('module.', '')
            if 'encoder' in k:
                k = k.replace('encoder.', '')
                encoder_state_dict[k] = v
        model.encoder.load_state_dict(encoder_state_dict)
    else:
        # Multimodal (CMC) case
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        for k, v in state.items():
            k = k.replace('module.', '')
            if 'encoder1' in k:
                k = k.replace('encoder1.', '')
                encoder1_state_dict[k] = v
            if 'encoder2' in k:
                k = k.replace('encoder2.', '')
                encoder2_state_dict[k] = v
        model.encoder1.load_state_dict(encoder1_state_dict)
        model.encoder2.load_state_dict(encoder2_state_dict)
    print('Pre-trained weights loaded!')

    return model


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
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth?dl=1'
        device = torch.device("cpu")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
        model = _load_encoder_weights(model, modal="RGB", state=state_dict["model"])
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
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACYcqgM-lcG3__QIbxuM2Koa/CMC.pth?dl=1'
        device = torch.device("cpu")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
        model = _load_encoder_weights(model, modal="CMC", state=state_dict["model"])
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
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAB53yJAYuCrOFluygBsVKOOa/MoCo.pth?dl=1'
        device = torch.device("cpu")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
        model = _load_encoder_weights(model, modal="RGB", state=state_dict["model"])
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
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AABaYuKiiZFYowa31yKeGGOQa/MoCov2.pth?dl=1'
        device = torch.device("cpu")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
        model = _load_encoder_weights(model, modal="RGB", state=state_dict["model"])
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
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth?dl=1'
        device = torch.device("cpu")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
        model = _load_encoder_weights(model, modal="RGB", state=state_dict["model"])
    return model

def InfoMin(pretrained=False, **kwargs):
    """
    What Makes for Good Views for Contrastive Learning?
    :param pretrained:
    :param kwargs:
    :return:
    """
    from pycontrast.networks.build_backbone import RGBMultiHeads
    model = RGBMultiHeads(**kwargs)

    if pretrained:
        url = 'https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth?dl=1'
        device = torch.device("cpu")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
        model = _load_encoder_weights(model, modal="RGB", state=state_dict["model"])
    return model

if __name__ == "__main__":
    InfoMin(pretrained=True)