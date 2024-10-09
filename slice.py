import os
import ast
import subprocess
import javalang
import sys
from tackle_C import tackle_c

def slice():
    org_dir_list = os.listdir("before")
    for org in org_dir_list:
        repo_dir_list = os.listdir(f"before/{org}")
        for repo in repo_dir_list:
            issue_dir_list = os.listdir(f"before/{org}/{repo}")
            for issue in issue_dir_list:
                files_dir_list = os.listdir(f"before/{org}/{repo}/{issue}")
                for file in files_dir_list:
                    c_flag = False
                    if file.endswith('.py') or file.endswith('.java') or file.endswith('.c') or file.endswith('.cpp') or file.endswith('.cc'):
                        with open(f"before/{org}/{repo}/{issue}/{file}", 'r') as f:
                            print(f"Now processing {org}/{repo}/{issue}/{file}")
                            start_num = 0
                            number = 0
                            commit = []
                            flag = False
                            real = 0
                            try:
                                code = f.read()
                                f.close()
                            except:
                                print("Exception when first reading! Continue!")
                                continue
                        with open(f"before/{org}/{repo}/{issue}/{file}", 'r') as f:
                            for line in f.readlines():
                                if line.startswith('@@'):
                                    n = line.split(' ')[1]
                                    n = n.split(',')[0]
                                    start_num = abs(int(n))
                                    flag = True
                                    number = 0
                                else:
                                    if line.startswith('-'):
                                        number += 1
                                        number += real
                                        if flag:
                                            if start_num + number - 1 not in commit:
                                                commit.append(start_num + number - 1)
                                                flag = False
                                        else:
                                            if start_num + number not in commit:
                                                commit.append(start_num + number)
                                    elif line.startswith('+'):
                                        number += real
                                        if flag:
                                            if start_num + number + 1 not in commit:
                                                commit.append(start_num + number + 1)
                                                flag = False
                                        else:
                                            if start_num + number not in commit:
                                                commit.append(start_num + number)
                                        number -= 1
                                        if number < 0:
                                            number = 0
                                            real += 1
                                    else:
                                        number += 1
                                        number += real
                            f.close()
                            commit.sort()
                            print("commit message:")
                            print(commit)
                        if not os.path.exists(f"repo/{org}/{repo}/{issue}/{file}"):
                            print("No such file! Continue!")
                            continue
                        file_flag = False
                        if file.endswith(".py"):
                            file_flag = True
                            with open(f"repo/{org}/{repo}/{issue}/{file}", 'r') as f:
                                try:
                                    code = f.read()
                                except:
                                    print("Exception when second reading! Continue!")
                                    continue
                                try:
                                    ast_tree = ast.parse(code)
                                except:
                                    print("Exception when second reading! Continue!")
                                    continue
                                function_objects = []
                                for node in ast_tree.body:
                                    if isinstance(node, ast.ClassDef):
                                        if node.lineno not in function_objects:
                                            function_objects.append(node.lineno)
                                    elif isinstance(node, ast.FunctionDef):
                                        if node.lineno not in function_objects:
                                            function_objects.append(node.lineno)
                                f.close()
                            print("The ast:")
                            print(function_objects)
                        elif file.endswith(".java"):
                            file_flag = True
                            with open(f"repo/{org}/{repo}/{issue}/{file}", 'r') as f:
                                try:
                                    code = f.read()
                                except:
                                    print("Exception when second reading! Continue!")
                                    continue
                                try:
                                    ast_tree = javalang.parse.parse(code)
                                except:
                                    print("Exception when second reading! Continue!")
                                    continue
                                function_objects = []
                                try:
                                    for path, node in ast_tree:
                                        if isinstance(node, javalang.tree.ClassDeclaration):
                                            if node.position.line not in function_objects:
                                                function_objects.append(node.position.line)
                                        if isinstance(node, javalang.tree.MethodDeclaration):
                                            if node.position.line not in function_objects:
                                                function_objects.append(node.position.line)
                                except:
                                    print("Exception when parseing java ast! Continue!")
                                    continue
                                # for path, node in ast_tree:
                                #     if isinstance(node, javalang.tree.ClassDeclaration):
                                #         if node.position.line not in function_objects:
                                #             function_objects.append(node.position.line)
                                #     if isinstance(node, javalang.tree.MethodDeclaration):
                                #         if node.position.line not in function_objects:
                                #             function_objects.append(node.position.line)
                                f.close()
                            print("The ast:")
                            print(function_objects)
                        elif file.endswith(".c") or file.endswith(".cpp") or file.endswith(".cc"):
                            c_flag = True
                            for i in commit:
                                tackle_c(org, repo, issue, file, i)   # a992
                        if not file_flag:
                            continue

                        if c_flag:
                            continue
                        results = []
                        point = 0
                        stay = 0
                        for i in commit:
                            if point == len(function_objects):
                                if len(function_objects) == 0:
                                    results.append((0, 'all'))
                                else:
                                    if stay == function_objects[point - 1]:
                                        if (stay, 'all') not in results:
                                            results.append((stay, 'all'))
                                break
                            if i < function_objects[point]:
                                if (stay, function_objects[point]) not in results:
                                    results.append((stay, function_objects[point]))
                            elif i == function_objects[point]:
                                stay = i
                            else:
                                point += 1

                        results = []
                        point = 0
                        stay = 0
                        stop_flag = False
                        flag = False
                        if len(function_objects) == 0:
                            flag = True
                            results.append((0, 'all'))
                        for i in commit:
                            if flag:
                                break
                            #print(f"Now processing a: {i}")
                            #print(f"i = {i}")
                            if point >= len(function_objects) - 1:
                                #print(f"The point {point} == {len(function_objects)}")
                                if commit[len(commit) - 1] >= function_objects[len(function_objects) - 1]:
                                    if (function_objects[len(function_objects) - 1], 'all') not in results:
                                        results.append((function_objects[len(function_objects) - 1], 'all'))
                                break
                            if i < stay:
                                #print(f"i < stay: {i} < {stay}, next")
                                continue
                            if i < function_objects[point]:
                                #print(f"i < b[point]: {i} < {function_objects[point]}")
                                #print("append")
                                if (stay, function_objects[point]) not in results:
                                    results.append((stay, function_objects[point]))
                            elif i == function_objects[point]:
                                #print(f"i == b[point]: {i} == {function_objects[point]}")
                                stay = i
                            else:
                                #print(f"i > b[point]: {i} > {function_objects[point]}")
                                stay = function_objects[point]
                                #print(f"stay = {stay}")
                                for j in function_objects[point + 1:]:
                                    #print(f"j = {j}")
                                    if i > j:
                                        #print(f"i > j: {i} > {j}")
                                        stay = j
                                        #print(f"stay = {stay}")
                                        point += 1
                                        #print(f"point = {point}")
                                    elif i < j:
                                        #print(f"i < j: {i} < {j}")
                                        if (stay, j) not in results:
                                            results.append((stay, j))
                                        #print("append")
                                        stay = j
                                        point += 1
                                        #print(f"stay = {stay}")
                                        break
                                    else:
                                        #print(f"i == j: {i} == {j}")
                                        stay = j
                                        point = function_objects.index(stay) + 1
                                        if point >= len(function_objects):
                                            stop_flag = True
                                            if (stay, 'all') not in results:
                                                results.append((stay, 'all'))
                                            break
                                        if (stay, function_objects[point]) not in results:
                                            results.append((stay, function_objects[point]))
                                        stay = function_objects[point]
                                        point += 1
                                        break
                                if point == len(function_objects) - 1:
                                    stop_flag = True
                                    results.append((stay, 'all'))
                            #print(f"point = {point}")
                            if stop_flag:
                                break
                        if point < len(function_objects) - 1:
                            results.append((stay, function_objects[point + 1]))

                        print(results)

                        if not os.path.exists(f"set/{org}/{repo}/{issue}"):
                            os.makedirs(f"set/{org}/{repo}/{issue}")
                        b = 1
                        temp = 0
                        with open(f"repo/{org}/{repo}/{issue}/{file}", 'r') as f:
                            for code in f.readlines():
                                if len(results) == 0:
                                    with open(f"set/{org}/{repo}/{issue}/{file}", 'a') as o:
                                        o.write(code)
                                    continue
                                if b >= results[temp][0] and results[temp][1] == "all":
                                    with open(f"set/{org}/{repo}/{issue}/{file}", 'a') as o:
                                        o.write(code)
                                elif b >= results[temp][0] and b < results[temp][1]:
                                    with open(f"set/{org}/{repo}/{issue}/{file}", 'a') as o:
                                        o.write(code)
                                # elif results[temp][1] == "all":
                                elif b < results[temp][0]:
                                    b += 1
                                    continue
                                elif b >= results[temp][1]:
                                    temp += 1
                                    if temp == len(results):
                                        break
                                    else:
                                        if b == results[temp][0]:
                                            with open(f"set/{org}/{repo}/{issue}/{file}", 'a') as o:
                                                o.write(code)
                                b += 1

def trace_extraction(file, line):
    command = ['./clang_ast', file, line]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error executing command!")
        return None
    return result.stdout

def is_info_empty(info):
    for line in info.splitlines():
        if line.strip():
            return False
    return True

if __name__ == '__main__':
    sys.setrecursionlimit(4000)
    slice()