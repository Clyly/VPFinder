import os

def merge_code():
    org_dir_list = os.listdir("set")
    for org in org_dir_list:
        repo_dir_list = os.listdir(f"set/{org}")
        for repo in repo_dir_list:
            issue_dir_list = os.listdir(f"set/{org}/{repo}")
            for issue in issue_dir_list:
                if os.path.exists(f"set/{org}/{repo}/{issue}/final_result.txt"):
                    print(f"Now processing {org}/{repo}/{issue}")
                    os.rename(f"set/{org}/{repo}/{issue}/final_result.txt", f"set/{org}/{repo}/{issue}/final_results.txt")
                    try:
                        with open(f"set/{org}/{repo}/{issue}/final_results.txt", 'r') as f, open(
                                f"set/{org}/{repo}/{issue}/my_temp.txt", 'w') as o:
                            for line in f.readlines():
                                if line.startswith("Function:"):
                                    continue
                                o.write(line)
                    except:
                        continue

                    with open(f"set/{org}/{repo}/{issue}/my_temp.txt", 'r') as f:
                        content = f.read()
                        f.close()

                    lines = content.split('\n')
                    unique_lines = []
                    for line in lines:
                        # line = line.strip()
                        if line not in unique_lines:
                            unique_lines.append(line)
                    result = '\n'.join(unique_lines)
                    with open(f"set/{org}/{repo}/{issue}/final_result.txt", 'w') as f:
                        f.write(result)
                        f.close()
                else:
                    with open(f"set/{org}/{repo}/{issue}/final_result.txt", 'w') as o:
                        files_dir_list = os.listdir(f"set/{org}/{repo}/{issue}")
                        for file in files_dir_list:
                            with open(f"set/{org}/{repo}/{issue}/{file}", 'r') as f:
                                print(f"Now processing {org}/{repo}/{issue}/{file}")
                                try:
                                    code = f.read()
                                except:
                                    continue
                                o.write(code)
                                f.close()

if __name__ == '__main__':
    merge_code()