import json
import time
import requests
import pandas as pd

# Github
api_key = "Your Token Here"

def get_comments():
    i = 1
    dataset = pd.read_csv("data/dataset_tc.csv")
    all_contents = []
    for index, row in dataset.iterrows():
        print(i)
        i += 1
        url = row['Issue_Url']
        org, repo_name, issue_number = url.split("/")[-4], url.split("/")[-3], url.split("/")[-1]
        issue_number = issue_number.rstrip('\n')
        print(url)
        comments_api = f"https://api.github.com/repos/{org}/{repo_name}/issues/{issue_number}/comments"

        headers = {
            "Authorization": "Token " + api_key,
            "Accept": "application/vnd.github.v3+json"
        }

        count = 0
        flag = False
        while count < 10:
            try:
                response = requests.get(comments_api, headers=headers)
                break
            except:
                count += 1
                time.sleep(count * 2)
                if count == 10:
                    flag = True

        if flag:
            print("exception when first requesting comments")
            with open("data/fail_comments.txt", 'a') as f:
                f.write(f"{url}\n")
            all_contents.append("No comments.")
            time.sleep(2)
            continue

        if response.status_code == 200:
            comments = json.loads(response.text)
        else:
            print("exception when first requesting comments")
            with open("data/fail_comments.txt", 'a') as f:
                f.write(f"{url}\n")
            all_contents.append("No comments.")
            time.sleep(2)
            continue

        contents = ""
        for comment in comments:
            comment_body = comment['body']
            contents = contents + comment_body + '\n'
        contents = contents.rstrip('\n')
        all_contents.append(contents)

    print(len(all_contents))

    dataset.insert(4, "Comment_bodys", all_contents)
    dataset.to_csv("data/dataset_tdc.csv")

if __name__ == '__main__':
    get_comments()