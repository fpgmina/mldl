import numpy as np
import torch
from models.cnn import CustomNet


def test_custom_net():
   torch.random.manual_seed(42)
   model = CustomNet()
   out = model(torch.randn(32, 3, 224, 224))
   out_mean = out.mean()
   assert np.allclose(out_mean.detach().numpy(), -0.01339262)