import subprocess
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

return_code_1 = process_1.returncode
return_code_2 = process_2.returncode

print(f"Command 2 return code: {return_code_2}")
print(f"Command 2 output: {stdout_2.decode()}")
print("1")
while (1):
    continue
# exit(1)
# process_1.wait()
# os.kill(pid, signal.SIGKILL)
# print(f"Command 2 error: {stderr_2.decode()}")
# torch.cuda.empty_cache()
# if (print(f"Command 2 output: {stdout_2.decode()}") == None):
#     print("Terminating the first command")
#     os.kill(pid, signal.SIGKILL)

# Ensure process_1 is fully terminated and collect its outputs and errors
# print("Waiting for the first command to terminate")
# stdout_1, stderr_1 = process_1.communicate()

# # Check the return code of process_1
# print("Collecting the output and error of the first command")
# return_code_1 = process_1.returncode
# print(f"Command 1 return code: {return_code_1}")
# print(f"Command 1 output: {stdout_1.decode()}")
# print(f"Command 1 error: {stderr_1.decode()}")
