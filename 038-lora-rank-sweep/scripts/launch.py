# run via `colab exec -s tig038 -f launch.py`: starts sweep.py in the background on the VM
import subprocess
subprocess.Popen("cd /content && nohup python sweep.py > /content/sweep_stdout.log 2>&1 &", shell=True)
print("launched")
