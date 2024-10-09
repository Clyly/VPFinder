import os
import sys

import chardet


def to_patch():
    org_dir_list = os.listdir("before")
    for org in org_dir_list:
        repo_dir_list = os.listdir(f"before/{org}")
        for repo in repo_dir_list:
            issue_dir_list = os.listdir(f"before/{org}/{repo}")
            for issue in issue_dir_list:
                issue_patch_add = ""
                issue_patch_del = ""
                issue_patch_add_flag = False
                issue_patch_del_flag = False
                files_dir_list = os.listdir(f"before/{org}/{repo}/{issue}")
                print(files_dir_list)
                for file in files_dir_list:
                    file_patch = ""
                    file_error_flag = False
                    try:
                        with open(f"before/{org}/{repo}/{issue}/{file}", 'r') as f:
                            print(f"Now processing {org}/{repo}/{issue}/{file}")
                            code = f.read()
                            f.close()
                    except:
                        file_error_flag = True
                        with open(f"before/{org}/{repo}/{issue}/{file}", 'rb') as error_file:
                            print(f"Now processing {org}/{repo}/{issue}/{file}")
                            content = error_file.read()
                            detect = chardet.detect(content)
                            encoding = detect['encoding']
                            error_file.close()

                    if not file_error_flag:
                        encoding = 'utf-8'
                    file_path = f"before/{org}/{repo}/{issue}/{file}"
                    file_path = os.path.normpath(file_path)
                    with open(file_path, 'r', encoding=encoding) as f:
                        i = 1
                        try:
                            for line in f.readlines():
                                if line.startswith('@@'):
                                    file_patch += f"patch {i}:\n"
                                    i += 1
                                elif line.startswith('-'):
                                    issue_patch_del_flag = True
                                    file_patch += line
                                    file_patch += '\n'
                                    issue_patch_del += line.lstrip('-')
                                    issue_patch_del += '\n'
                                elif line.startswith('+'):
                                    issue_patch_add_flag = True
                                    file_patch += line
                                    file_patch += '\n'
                                    issue_patch_add += line.lstrip('+')
                                    issue_patch_add += '\n'
                        except:
                            continue

                    if not os.path.exists(f"patch/{org}/{repo}/{issue}"):
                        os.makedirs(f"patch/{org}/{repo}/{issue}")
                    with open(f"patch/{org}/{repo}/{issue}/{file}", 'w', encoding='utf-8') as o:
                        o.write(file_patch.rstrip('\n'))

                if not os.path.exists(f"patch/{org}/{repo}/{issue}"):
                    os.makedirs(f"patch/{org}/{repo}/{issue}")
                if issue_patch_add_flag:
                    with open(f"patch/{org}/{repo}/{issue}/add.txt", 'w', encoding='utf-8') as o:
                        o.write(issue_patch_add.rstrip('\n'))
                else:
                    with open(f"patch/{org}/{repo}/{issue}/add.txt", 'w', encoding='utf-8') as o:
                        o.write("No add.")
                if issue_patch_del_flag:
                    with open(f"patch/{org}/{repo}/{issue}/del.txt", 'w', encoding='utf-8') as o:
                        o.write(issue_patch_del.rstrip('\n'))
                else:
                    with open(f"patch/{org}/{repo}/{issue}/del.txt", 'w', encoding='utf-8') as o:
                        o.write("No del.")


if __name__ == "__main__":
    sys.setrecursionlimit(4000)
    to_patch()