import os

from vllm.lora.resolver import LoRAResolverRegistry

def register():
    """Register the filesytem LoRA Resolver with vLLM"""

    lora_cache_dir = os.environ["VLLM_PLUGIN_LORA_CACHE_DIR"]
    if not lora_cache_dir or not os.path_exists(lora_cache_dir):
        raise ValueError("VLLM_PLUGIN_LORA_CACHE_DIR must be set to a valid directory for Filesystem Resolver plugin to function")
    fs_resolver = FileSystemResolver(lora_cache_dir)
    LoRAResolverResgistry.register_resolver("Filesystem Resolver", fs_resolver)
