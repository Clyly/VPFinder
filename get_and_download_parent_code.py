import os
import requests
import time
import json

# Github
api_key = "Your Token Here"

headers = {
    "Authorization": "Token " + api_key,
    "Accept": "application/vnd.github.v3+json"
}

def code_return_both():
    i = 0
    with open("data/output.txt", 'r') as f, open("data/failed.txt", 'w') as x:
        for url in f.readlines():
            print(i)
            i += 1

            org, repo, issue, sha = url.split("/")[0], url.split("/")[1], url.split("/")[2], url.split("/")[3]
            sha = sha.rstrip('\n')

            url_api = f"https://api.github.com/repos/{org}/{repo}/commits/{sha}"
            print(url_api)

            count_1 = 0
            flag_1 = False
            while count_1 < 10:
                try:
                    response = requests.get(url_api, headers=headers)
                    break
                except:
                    count_1 += 1
                    time.sleep(count_1 * 3)
                    if count_1 == 10:
                        print("Exception when first requesting!")
                        flag_1 = True

            if flag_1:
                x.write(f"{org}/{repo}/{issue}/{sha}\n")
                continue

            if response.status_code == 200:
                events = json.loads(response.text)
            else:
                print("Exception with status code!")
                x.write(f"{org}/{repo}/{issue}/{sha}\n")
                time.sleep(5)
                continue

            if events["parents"] is not None:
                parent_sha = events["parents"][0]["sha"]
            else:
                print("No parents")
                x.write(f"{org}/{repo}/{issue}/{sha}\n")
                time.sleep(2)
                continue

            files = None
            if events["files"] is not None:
                files = events["files"]
            if files is None:
                print("No files!")
                x.write(f"{org}/{repo}/{issue}/{sha}\n")
                time.sleep(5)
                continue
            else:
                flag = False
                for file in files:
                    print(file["filename"])
                    file_name = file["filename"]
                    filr_url = f"https://raw.githubusercontent.com/{org}/{repo}/{parent_sha}/{file_name}"
                    count_2 = 0
                    print("now trying download...")
                    flag_2 = False
                    while count_2 < 10:
                        try:
                            print(f"now {count_2 + 1} try")
                            response1 = requests.get(filr_url, headers=headers)
                            break
                        except:
                            count_2 += 1
                            time.sleep(count_2 * 3)
                            if count_2 == 10:
                                print("Max try failed!")
                                flag_2 = True
                    if flag_2:
                        x.write(f"{org}/{repo}/{issue}/{sha}\n")
                        continue
                    if not os.path.exists(f"repo/{org}/{repo}/{issue}"):
                        os.makedirs(f"repo/{org}/{repo}/{issue}")
                    file_n = file_name.split("/")[-1]
                    with open(f"repo/{org}/{repo}/{issue}/{file_n}", 'wb') as o:
                        o.write(response1.content)
                    print("file download")
                    time.sleep(8)

                    print("now trying patch...")
                    try:
                        patch = file["patch"]
                        flag = True
                    except:
                        continue
                    if not os.path.exists(f"before/{org}/{repo}/{issue}"):
                        os.makedirs(f"before/{org}/{repo}/{issue}")
                    try:
                        with open(f"before/{org}/{repo}/{issue}/{file_n}", 'w') as o:
                            o.write(patch)
                            flag = True
                            print("patch download")
                    except:
                        print("Exception when writing patch!")
                if not flag:
                    x.write(f"{org}/{repo}/{issue}/{sha}\n")

if __name__ == '__main__':
    code_return_both()