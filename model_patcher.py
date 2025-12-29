"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""

import copy
from comfy.model_patcher import ModelPatcher


class NunchakuModelPatcher(ModelPatcher):
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "is_injected"):
            self.is_injected = False
        if not hasattr(self, "skip_injection"):
            self.skip_injection = False
        if not hasattr(self, "pinned"):
            self.pinned = set()

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """
        Load the diffusion model onto the specified device.

        Parameters
        ----------
        device_to : torch.device or str, optional
            The device to which the diffusion model should be moved.
        lowvram_model_memory : int, optional
            Not used in this implementation.
        force_patch_weights : bool, optional
            Not used in this implementation.
        full_load : bool, optional
            Not used in this implementation.
        """
        with self.use_ejected():
            self.model.diffusion_model.to_safely(device_to)

    def detach(self, unpatch_all: bool = True):
        """
        Detach the model and move it to the offload device.

        Parameters
        ----------
        unpatch_all : bool, optional
            If True, unpatch all model components (default is True).
        """
        self.eject_model()
        self.model.diffusion_model.to_safely(self.offload_device)

    def clone(self):
        n = NunchakuModelPatcher(self.model, self.load_device, self.offload_device, self.model_size(), weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.pinned = self.pinned
        n.is_injected = self.is_injected
        n.skip_injection = self.skip_injection
        
        return n

    def pin_weight_to_device(self, key):
        pass

    def unpin_weight(self, key):
        pass

    def unpin_all_weights(self):
        pass
