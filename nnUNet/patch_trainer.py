with open("/root/HCMA-UNet/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py", "r") as f:
    lines = f.read()

lines = lines.replace("self.enable_deep_supervision = True", "self._enable_deep_supervision = True")
lines = lines.replace("self.enable_deep_supervision =", "self._enable_deep_supervision =")
lines = lines.replace("self.enable_deep_supervision", "self._enable_deep_supervision")

prop = """
    @property
    def enable_deep_supervision(self):
        return getattr(self, '_enable_deep_supervision', True)

    @enable_deep_supervision.setter
    def enable_deep_supervision(self, value):
        if value:
            import traceback
            print("SETTING DEEP SUPERVISION TO TRUE! Traceback:")
            traceback.print_stack()
        self._enable_deep_supervision = value
"""

# Insert it inside nnUNetTrainer class
lines = lines.replace("class nnUNetTrainer(object):", "class nnUNetTrainer(object):" + prop)

with open("/root/HCMA-UNet/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py", "w") as f:
    f.write(lines)
