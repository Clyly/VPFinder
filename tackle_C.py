import os

def tackle_c(org, repo, issue, file, line):
    cpp_executable = "./clang_ast"
    source_file = f"/home/build/repo/{org}/{repo}/{issue}/{file}"
    target_line = line
    output_file = f"/home/build/set/{org}/{repo}/{issue}/final_result.txt"
    output_path = f"/home/build/set/{org}/{repo}/{issue}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # result = subprocess.run([cpp_executable, source_file, target_line, output_file], capture_output=True, text=True)
    # print(result.returncode)
    # print(result.stdout)
    with open("build/slice.sh", 'a') as f:
        f.write(f"{cpp_executable} {source_file} {target_line} {output_file}")
        f.write('\n')
        f.close()

if __name__ == '__main__':
    tackle_c("acassen", "keepalived", "277", "vrrp.c", "801")