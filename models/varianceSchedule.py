from utils.networkHelper import *

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008, **kwargs):


class VarianceSchedule(nn.Module):
    def __init__(self,
                 var_schedule="linear_beta_schedule",
                 beta_start=None,
                 beta_end=None):
        super().__init__()
        self.schedule_name = var_schedule
        var_schedule_dict = {'linear_beta_schedule': linear_beta_schedule}

        if var_schedule in var_schedule_dict:
            self.var_schedule = var_schedule_dict[var_schedule]
        else:
            raise ValueError('Function not found in dictionary')

        if beta_end and beta_start is None and var_schedule != "cosine_beta_schedule":
            self.beta_start = 0.0001
            self.beta_end = 0.02
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end

    def forward(self, timesteps):
        return self.var_schedule(timesteps=timesteps) if self.schedule_name == "cosine_beta_schedule" \
            else self.var_schedule(timesteps=timesteps, beta_start=self.beta_start, beta_end=self.beta_end)