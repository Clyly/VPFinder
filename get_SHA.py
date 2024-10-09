import requests
import json
import time
import re

# Github
api_key = "Your Token Here"

def get_code():
    i = 1
    with open("data/url.txt", 'r') as f:
        for url in f.readlines():
            print(i)
            i += 1

            # Extract the organization name, repo name, and issue number from the URL
            org, repo_name, issue_number = url.split("/")[-4], url.split("/")[-3], url.split("/")[-1]
            issue_number = issue_number.rstrip('\n')

            url_api = f"https://api.github.com/repos/{org}/{repo_name}/issues/{issue_number.strip()}/events"
            print(url_api)
            comments_api = f"https://api.github.com/repos/{org}/{repo_name}/issues/{issue_number}/comments"

            headers = {
                "Authorization": "Token " + api_key,
                "Accept": "application/vnd.github.v3+json"
            }

            count_1 = 0
            print("now trying...")
            flag_1 = False
            while count_1 < 10:
                try:
                    print(f"now {count_1 + 1}th try")
                    response = requests.get(url_api, headers=headers)
                    break
                except:
                    count_1 += 1
                    time.sleep(count_1 * 2)
                    if count_1 == 10:
                        print("max try")
                        flag_1 = True
            if flag_1:
                print("exception when first requesting")
                continue

            if response.status_code == 200:
                events = json.loads(response.text)
            else:
                print("exception when first requesting")
                time.sleep(2)
                comment_commit = get_from_comments(comments_api)
                if comment_commit is not None:
                    with open("data/output.txt", 'a') as o:
                        o.write(f"{org}/{repo_name}/{issue_number}/{comment_commit}\n")
                continue

            commit = None
            for event in events:
                if event["commit_url"] != 'null':
                    commit = event["commit_url"]
                    break

            if commit is None:
                print("No commits")
                time.sleep(2)
                comment_commit = get_from_comments(comments_api)
                if comment_commit is not None:
                    with open("data/output.txt", 'a') as o:
                        o.write(f"{org}/{repo_name}/{issue_number}/{comment_commit}\n")
                continue
            else:
                count_2 = 0
                flag_2 = False
                while count_2 < 10:
                    try:
                        response1 = requests.get(commit, headers=headers)
                        break
                    except:
                        count_2 += 1
                        time.sleep(count_2 * 2)
                        if count_2 == 10:
                            flag_2 = True
            if flag_2:
                print("exception when second requesting")
                continue

            if response1.status_code == 200:
                comm = json.loads(response1.text)
            else:
                print("exception when second requesting")
                time.sleep(2)
                continue

            if comm["sha"] is None:
                print(f"{org}/{repo_name}/{issue_number}: no sha")
                time.sleep(2)
                comment_commit = get_from_comments(comments_api)
                if comment_commit is not None:
                    with open("data/output.txt", 'a') as o:
                        o.write(f"{org}/{repo_name}/{issue_number}/{comment_commit}\n")
                continue
            else:
                a = comm["sha"]
                print(f"{org}/{repo_name}/{issue_number}/{a}")
                with open("data/output.txt", 'a') as o:
                    o.write(f"{org}/{repo_name}/{issue_number}/{a}\n")
            time.sleep(2)

def get_from_comments(url):
    # https://api.github.com/repos/org/repo_name/issues/issue_number/comments
    print("Now trying comments...")
    time.sleep(2)

    org, repo_name, issue_number = url.split("/")[-5], url.split("/")[-4], url.split("/")[-2]

    headers = {
        "Authorization": "Token " + api_key,
        "Accept": "application/vnd.github.v3+json"
    }

    count = 0
    flag = False
    while count < 10:
        try:
            response = requests.get(url, headers=headers)
            break
        except:
            count += 1
            time.sleep(count * 2)
            if count == 10:
                flag = True

    if flag:
        print("exception when first requesting comments")
        return None

    if response.status_code == 200:
        comments = json.loads(response.text)
    else:
        print("exception when first requesting comments")
        return None

    for comment in comments:
        comment_body = comment['body']
        pattern = r"https://github.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/commit/[a-zA-Z0-9]+"
        matches = re.findall(pattern, comment_body)

        for match in matches:
            m_org, m_repo_name, commit = match.split("/")[-4], match.split("/")[-3], match.split("/")[-1]
            if org == m_org and repo_name == m_repo_name:
                return commit

    return None

if __name__ == '__main__':
    get_code()