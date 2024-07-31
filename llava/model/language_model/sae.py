import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers import PretrainedConfig, PreTrainedModel
from torch.nn import L1Loss, MSELoss, LayerNorm, SmoothL1Loss

class SAEConfig(PretrainedConfig):

    model_type = "sae"

    def __init__(
        self,
        hidden_size=4096,
        dictionary_size=11008,
        hidden_act="relu",
        is_training=False,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.dictionary_size = dictionary_size
        self.hidden_act = hidden_act
        self.is_training = is_training

        super().__init__(
                    **kwargs
        )


class SAE(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.dictionary_size = config.dictionary_size

        self.proj_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.up_proj = nn.Linear(self.hidden_size, self.dictionary_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.dictionary = torch.load("/home/D/mj/model/final_dictionary.pt").to(torch.float32).to("cuda")

        # con_param = torch.load("/home/mj/model/final_dictionary.pt")

        # con_param = con_param.contiguous().T

        # self.dictionary.weight = nn.Parameter(con_param)
        self.norm = nn.LayerNorm(self.hidden_size)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, input_ids: torch.Tensor = None, labels: torch.Tensor = None, eval: bool = False):
        # up_proj = self.act_fn(self.gate_proj(self.proj(x))) * self.up_proj(self.proj(x))

        proj = self.proj_value(self.act_fn(self.proj_key(input_ids)))

        up_proj = self.act_fn(self.up_proj(proj))

        logits = torch.matmul(up_proj, self.dictionary)
        logits = self.norm(logits)

        if labels is not None:

            l2_loss = MSELoss(reduction="mean")

            # l1_penalty = 1e-3 * sum([p.abs().sum() for p in self.up_proj.parameters()])
            l1_parameters = []
            for parameter in self.up_proj.parameters():
                l1_parameters.append(parameter.view(-1))

            l1_weight = 1e-5
            l1_penalty = l1_weight * self.compute_l1_loss(torch.cat(l1_parameters))

            loss = l2_loss(logits, labels) + l1_penalty
            # print(loss)
            return {"loss": loss, "logits": logits}

        if eval:
            return {"up_proj": up_proj, "logits": logits}

        return {"logits": logits}

