"""STC-LoRA: synaptic tagging & capture for continual learning in frozen LLMs.

    from stc_lora import STCLoRA, STCLoRAConfig
    adapter = STCLoRA(hf_model, STCLoRAConfig(rank=8, perm_lr_ratio=0.05))
    adapter.learn(input_ids, neuromod=0.8)   # surprise-gated update+decay+capture

Paper: "Synaptic Tagging and Capture for Continual Learning in Frozen Language
Models" (paper/latex/main.tex). Every table and figure reproduces from
experiments/ against the artifacts in outputs/.
"""

from stc_lora.adapter import STCLoRA, STCLoRAConfig, STCLoRALinear

__version__ = "0.1.0"
__all__ = ["STCLoRA", "STCLoRAConfig", "STCLoRALinear"]
