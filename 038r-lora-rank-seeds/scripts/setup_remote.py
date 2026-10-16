import subprocess
subprocess.run(['pip','uninstall','-y','torchao'],check=False)
subprocess.run(['pip','install','-q','transformers','peft','datasets','accelerate'],check=True)
import torch
print(torch.cuda.get_device_name(), flush=True)
