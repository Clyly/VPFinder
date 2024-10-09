import os
import json
import time
import requests
import pandas as pd

# Github
api_key = "Your Token Here"

headers = {
            "Authorization": "Token " + api_key,
            "Accept": "application/vnd.github.v3+json"
        }

def get_dataset_commits():
    i = 1
    dataset = pd.read_csv("data/dataset_description.csv")
    for index, row in dataset.iterrows():
        print(i)
        i += 1
        url = row['Issue_Url']
        org, repo_name, issue_number = url.split("/")[-4], url.split("/")[-3], url.split("/")[-1]
        issue_number = issue_number.rstrip('\n')
        print(url)

        flag = True
        with open("data/output.txt", 'r') as f:
            for content in f.readlines():
                o_org, o_repo, o_issue, sha = content.split("/")[0], content.split("/")[1], content.split("/")[2], content.split("/")[3]
                sha = sha.rstrip('\n')
                if org == o_org and repo_name == o_repo and issue_number == o_issue:
                    flag = False
                    break
        if flag:
            print("Not in output.")
            continue

        url_api = f"https://api.github.com/repos/{org}/{repo_name}/commits/{sha}"
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
                    with open("data/commit_failed.txt", 'a') as f:
                        f.write(url + '\n')
                    flag_1 = True

        if flag_1:
            continue

        if response.status_code == 200:
            events = json.loads(response.text)
        else:
            print("Exception with status code!")
            with open("data/commit_failed.txt", 'a') as f:
                f.write(url + '\n')
            time.sleep(5)
            continue

        files = None
        if events["files"] is not None:
            files = events["files"]

        if files is None:
            print("No files!")
            with open("data/commit_failed.txt", 'a') as f:
                f.write(url + '\n')
            time.sleep(5)
            continue
        else:
            message = None
            if events["commit"]["message"] is not None:
                message = events["commit"]["message"]
            else:
                message = " No message."
            print(message)
            if not os.path.exists(f"message/{org}/{repo_name}/{issue_number}"):
                os.makedirs(f"message/{org}/{repo_name}/{issue_number}")
            with open(f"message/{org}/{repo_name}/{issue_number}/message.txt", 'w', encoding='utf-8') as f:
                f.write(message)


            flag = False
            for file in files:
                print(file["filename"])
                file_name = file["filename"]
                filr_url = f"https://raw.githubusercontent.com/{org}/{repo_name}/{sha}/{file_name}"
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
                    continue
                if not os.path.exists(f"commit/{org}/{repo_name}/{issue_number}"):
                    os.makedirs(f"commit/{org}/{repo_name}/{issue_number}")
                file_n = file_name.split("/")[-1]
                with open(f"commit/{org}/{repo_name}/{issue_number}/{file_n}", 'wb') as o:
                    o.write(response1.content)
                print("file download")
                time.sleep(8)

if __name__ == '__main__':
    get_dataset_commits()