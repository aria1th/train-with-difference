"""
LoRA

class LoRAModule

class LoRANetwork

class LoRANetworkManager
    Handles the LoRA networks
    - LoRANetworkManager(networks)

forward(*orinal_params, **original_kwargs, network_args:Dict[str, Dict[Any, Any]])
    Calls forward on the network with the given args
    network_args is a dictionary of network names to a dictionary of arguments to pass to that network
"""
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

LORA_PREFIX_UNET = "lora_unet"
UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
    "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
]  # locon, 3clier

DEFAULT_TARGET_REPLACE = [
    "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
]

class LoRAModule(nn.Module):
    """
    Offers forward method to be merged with original forward result.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Linear" or "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d" or "Conv" in org_module.__class__.__name__:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            raise NotImplementedError(
                f"LoRAModule only supports Linear and Conv2d, but got {org_module.__class__.__name__}"
            )

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def forward(self, x, *args, **kwargs):
        """
        Returns partial forward result of the original module, which is to be merged with original forward result.
        """
        return (
            self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    """
    LoRA network.
    Single LoRA Network contains multiple LoRAModules which is used for saving differences for each module weights.
    Bias is not trained.
    
    LoRANetwork should have LoRANetworkManager which handles forward for each UNet's original forward.
    
    """
    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha

        # LoRAのみ
        self.module = LoRAModule

        # unetのloraを作る
        self.unet_loras = self.create_modules(
            LORA_PREFIX_UNET,
            unet,
            DEFAULT_TARGET_REPLACE,
            self.lora_dim,
            self.multiplier,
        )
        assert len(self.unet_loras) > 0, "no lora modules created."
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion 名前の被りがないか確認しているようだ
        lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        # 適用する
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
    ) -> list:
        """
        Creates LoRA modules for the given root module and returns a list of them.
        """
        loras = []

        for name, module in root_module.named_modules():
            #print(f"create LoRA for {module.__class__.__name__}: {name}")
            if module.__class__.__name__ in target_replace_modules or 'LoRACompatible' in module.__class__.__name__:
                for child_name, child_module in module.named_modules():
                    #print(f"create LoRA for {child_module.__class__.__name__}: {name}.{child_name}")
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"] or 'LoRACompatible' in child_module.__class__.__name__:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        #print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha
                        )
                        loras.append(lora)

        return loras

    def prepare_optimizer_params(self):
        """
        Returns a list of dictionaries of parameters to be passed to the optimizer.
        """
        all_params = []
        if self.unet_loras:  # 実質これしかない
            params = []
            for lora in self.unet_loras:
                params.extend(lora.parameters())
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        """
        Saves the weights of the network to the given file.
        """
        state = self.state_dict()
        if state is None:
            raise RuntimeError("no state dict found.")
        assert isinstance(state, dict), "state dict is not a dictionary."

        if dtype is not None:
            for key in list(state.keys()):
                v = state[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state[key] = v

        for key in list(state.keys()):
            if not key.startswith("lora"):
                # lora以外除外
                del state[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state, file, metadata)
        else:
            torch.save(state, file)

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0

class LoRANetworkManager:
    """
    Handles the LoRA networks
    
    This is not a torch.nn.Module.
    The LoRANetworkManager class hijacks the forward method of the original network and replaces it.
    The hijacked forward will accept a dictionary of arguments to pass to each LoRA network.
    
    At default, without arguments, the original forward is called.
    See SingleLoraNetworkManager to use default LoRA Network to be called every time.
    """
    NotImplemented