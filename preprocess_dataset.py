import pandas as pd
import numpy as np
import re
import os
from MemVul.util import replace_tokens_simple_1

def preprocess_dataset_csv():
    # preprocess the dataset
    df = pd.read_csv("data/all_samples.csv", low_memory=False)
    # print(len(df))
    # print(len(df[(df.Issue_Title.isnull()) | (df.Issue_Body.isnull())]))
    print(len(df[(df.Issue_Title.isnull()) & (df.Issue_Body.isnull())]))  # 1

    # remove issue report with missing title and body
    df = df[~(df.Issue_Title.isnull() & df.Issue_Body.isnull())]
    # df = df[~(df.Issue_Title.isnull() | df.Issue_Body.isnull())]
    print(len(df))
    print(len(df[df.Security_Issue_Full == 1]))
    print(len(df[df.Security_Issue_Full == 0]))

    # remove issues that created after the disclosure of correspoding CVE
    df.fillna("", inplace=True)
    df["project"] = df["Issue_Url"].map(extract_project)
    # print(len(set(df["project"].tolist())))

    # remove CIR created after the official disclosure of corresponding CVE
    df["Issue_Created_At"] = df["Issue_Created_At"].map(fix_time)
    print(len(df[(df.Security_Issue_Full == 1) & (df.Issue_Created_At > df.Published_Date)]))  # 53
    df = df[(df.Security_Issue_Full == 0) | (df.Issue_Created_At < df.Published_Date)]

    # remove projects without CIRs (since we remove some CIRs in the last step)
    df["valid"] = df.groupby('project')['Security_Issue_Full'].transform('sum')  # 0-invalid >=1-valid
    print(len(df[df.valid == 0]))  # 26421

    df = df[df.valid != 0]
    # print(len(df))
    # print(len(set(df["project"].tolist())))
    # print(len(df[df.Security_Issue_Full == 1]))
    # print(len(df[df.Security_Issue_Full == 0]))

    # my work
    number = 0
    df['code'] = 'code'
    org_dir_list = os.listdir("set")
    for org in org_dir_list:
        repo_dir_list = os.listdir(f"set/{org}")
        for repo in repo_dir_list:
            issue_dir_list = os.listdir(f"set/{org}/{repo}")
            for issue in issue_dir_list:
                number += 1
                try:
                    with open(f"set/{org}/{repo}/{issue}/final_result.txt", 'r') as f:
                        code = f.read()
                        url = f"https://github.com/{org}/{repo}/issues/{issue}"
                        index = df.index[df["Issue_Url"] == url].tolist()[0]
                        df.loc[index, 'code'] = code
                        f.close()
                except:
                    continue
    print(number)
    df.dropna(axis=0, subset=["code"])

    print("begin process...")
    df["Issue_Title"] = df["Issue_Title"].map(replace_tokens_simple_1)
    df["Issue_Body"] = df["Issue_Body"].map(replace_tokens_simple_1)
    print("finish process...")
    df.to_csv("data/all_samples_processed.csv")

def extract_project(url):
    tmp = url.split('/')
    if len(tmp) != 7:
        print("ERROR" + url)
        return "ERROR"
    return f"{tmp[3]}/{tmp[4]}"

def fix_time(t):
    t = t.strip()
    t = re.sub(r'\sUTC', 'Z', t)
    # t = t.strip()
    t = re.sub(r'\s', 'T', t)
    return t

def exam():
    dataset = pd.read_csv("data/all_samples_processed.csv", low_memory=False)
    print(len(dataset["code"]))

if __name__ == '__main__':
    # exam()
    # generate_dataset_mlm('/validation_project.json')
    preprocess_dataset_csv()
