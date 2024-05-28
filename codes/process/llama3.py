import subprocess
import os
import subprocess
import os
# Define the two Bash commands you want to run concurrently
bash_command_1 = "/share/ollama/ollama serve"
bash_command_2 = "/share/ollama/ollama run llama3:70b < /root/LLM_project/codes/process/input.txt > /root/LLM_project/codes/process/output.txt"

print("Starting the first command")
process_1 = subprocess.Popen(bash_command_1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
pid = process_1.pid
print(process_1.pid)
print("Starting the second command")
process_2 = subprocess.Popen(bash_command_2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(process_2.pid)
stdout_2, stderr_2 = process_2.communicate()

return_code_2 = process_2.returncode

print(f"Command 2 return code: {return_code_2}")
print(f"Command 2 output: {stdout_2.decode()}")
print(f"Command 2 error: {stderr_2.decode()}")
process_1.send_signal(subprocess.signal.SIGINT)
os.system("ps -ef | grep -- '--n-gpu-layers' | awk '{print $2}' | head -n 1 | xargs kill -9")