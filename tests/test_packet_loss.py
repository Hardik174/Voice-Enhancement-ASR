import torch
from src.augment.packet_loss import gilbert_elliott_loss

def test_shape():
    x=torch.randn(1,16000); y=gilbert_elliott_loss(x,.1); assert y.shape==x.shape
