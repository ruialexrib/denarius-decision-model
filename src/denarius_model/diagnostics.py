import importlib.metadata
import platform
import shutil

def diagnose():
    result={'python':platform.python_version(),'ollama':bool(shutil.which('ollama'))}
    for p in ('torch','transformers','peft','accelerate','datasets','bitsandbytes'):
        try: result[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: result[p]=None
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda)
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info(); result.update(gpu=torch.cuda.get_device_name(),free_vram=free,total_vram=total,bf16=torch.cuda.is_bf16_supported())
    except ImportError: pass
    return result
