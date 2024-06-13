from utils.networkHelper import *


class DiffusionModel(nn.Module):
    def __init__(self,
                 var_schedule="linear_beta_schedule",
                 timestep=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None
                 ):
        super().__init__()
        self.net = denoise_model
        