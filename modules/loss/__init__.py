from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy, KLDivLoss, CrossEntropyLabelSmoothFilterNoise
from .extra import AALS, PGLR, UET, RegLoss, UET2
from .partavgtriplet import PartAveragedTripletLoss
from .center_triplet import CenterTripletLoss
from .faceloss import CosFace, AdaFace, ArcFace

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'KLDivLoss',
    'CrossEntropyLabelSmoothFilterNoise',
    "AALS",
    'PGLR',
    "UET",
    "UET2",
    "RegLoss",
    "PartAveragedTripletLoss",
    "CenterTripletLoss",

    "CosFace",
    "AdaFace",
    "ArcFace"
    
]

__factory = {
    "CosFace" : CosFace,
    "AdaFace" : AdaFace,
    "ArcFace" : ArcFace,

    "": None
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)