import re
import random
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import date, datetime
from sklearn.utils import shuffle

from MemVul.util import replace_tokens_simple
from MemVul.util import replace_tokens_simple_1


def generate_dataset_mlm(file):
    # generate the trainset for run_mlm_wwm.py (further pretraining BERT)
    # each line corresponds to a issue reprot
    datasets = json.load(open(DATA_PATH + '/' + file, 'r'))
    print(len(datasets))
    descriptions = [f"{_['Issue_Title']}. {_['Issue_Body']}" for _ in datasets]
    new_file = DATA_PATH + '/' + f"test_project_mlm.txt"
    with open(new_file, 'w', encoding="utf-8") as f:
        f.write('\n'.join(descriptions))

def fix_time(t):
    t = t.strip()
    t = re.sub(r'\sUTC', 'Z', t)
    # t = t.strip()
    t = re.sub(r'\s', 'T', t)
    return t


def check_text(s):
    ss = s
    ss = re.sub(r'[^\u0000-\u007F]', ' ', ss)
    num_token = len(re.findall(r'[a-z]+', ss, re.I))
    # print(num_token)
    if num_token < 10:
        return False
    else:
        return True

def check_text_1(s):
    if re.search(r'[\u4E00–\u9FFF]', s):
        return False
    else:
        return True


def preprocess_dataset_csv(f):
    # preprocess the dataset
    df = pd.read_csv(DATA_PATH + "/" + f"{f}.csv", low_memory=False)

    print(len(df[df.Security_Issue_Full == 1]))   # 1111
    print(len(df[df.Security_Issue_Full == 0]))   # 3701

    # print(len(df))
    # print(len(df[(df.Issue_Title.isnull()) | (df.Issue_Body.isnull())]))
    print(len(df[(df.Issue_Title.isnull()) & (df.Issue_Body.isnull())]))  # 0

    # remove issue report with missing title and body
    # df = df[~(df.Issue_Title.isnull() & df.Issue_Body.isnull())]
    # df = df[~(df.Issue_Title.isnull() | df.Issue_Body.isnull())]
    # print(len(df))
    # print(len(df[df.Security_Issue_Full == 1]))
    # print(len(df[df.Security_Issue_Full == 0]))

    # remove issues that created after the disclosure of correspoding CVE
    df.fillna("", inplace=True)
    df["project"] = df["Issue_Url"].map(extract_project)
    # print(len(set(df["project"].tolist())))
    
    # remove CIR created after the official disclosure of corresponding CVE
    df["Issue_Created_At"] = df["Issue_Created_At"].map(fix_time)
    print(len(df[(df.Security_Issue_Full == 1) & (df.Issue_Created_At > df.Published_Date)]))  # 0
    df = df[(df.Security_Issue_Full == 0) | (df.Issue_Created_At < df.Published_Date)]

    # remove projects without CIRs (since we remove some CIRs in the last step)
    df["valid"] = df.groupby('project')['Security_Issue_Full'].transform('sum')  # 0-invalid >=1-valid
    print(len(df[df.valid == 0]))  # 1030

    df = df[df.valid != 0]
    # print(len(df))
    # print(len(set(df["project"].tolist())))
    print(len(df[df.Security_Issue_Full == 1]))   # 1111
    print(len(df[df.Security_Issue_Full == 0]))   # 2671

    print("begin process...")
    df["Issue_Title"] = df["Issue_Title"].map(replace_tokens_simple_1)
    df["Issue_Body"] = df["Issue_Body"].map(replace_tokens_simple_1)
    print("finish process...")
    df.to_csv(DATA_PATH + '/' + f"{f}_processed.csv")


def extract_project(url):
    tmp = url.split('/')
    if len(tmp) != 7:
        print("ERROR" + url)
        return "ERROR"
    return f"{tmp[3]}/{tmp[4]}"


def divide_dataset_project_csv(f):
    # divide the train set and test set
    # divide the train set and validation set
    df = pd.read_csv(DATA_PATH + '/' + f"{f}.csv", low_memory=False)

    df.fillna("", inplace=True)
    df["project"] = df["Issue_Url"].map(extract_project)
    project = list(set(df["project"].tolist()))
    project.sort()
    print(len(df))   # 3782
    print(len(project))   # 358

    project = shuffle(project)

    project_selected = random.sample(project, k=int(len(project)*0.1))
    print(project_selected)
    print(len(project))
    df_train = df[~df.project.isin(project_selected)]

    print(len(df_train))   # 3323
    print(len(set(df_train["project"].tolist())))   # 323
    print(len(df_train[df_train.Security_Issue_Full == 1]))   # 1026
    print(len(df_train[df_train.Security_Issue_Full == 0]))   # 2297
    del df_train["project"]

    # print("ratio:", len(df_train[df_train.Security_Issue_Full == 1]) / len(df_train))

    df_test = df[df.project.isin(project_selected)]
    print(len(df_test))   # 459
    print(len(set(df_test["project"].tolist())))   # 35
    print(len(df_test[df_test.Security_Issue_Full == 1]))   # 85
    print(len(df_test[df_test.Security_Issue_Full == 0]))   # 374
    del df_test["project"]

    # print("ratio:", len(df_test[df_test.Security_Issue_Full == 1]) / len(df_test))

    # divide into train and test
    df_train, df_test = shuffle(df_train), shuffle(df_test)
    df_train.to_csv(DATA_PATH + '/' + "train_project.csv", index=False)
    df_test.to_csv(DATA_PATH + '/' + "test_project.csv", index=False)
    df_test.to_csv(DATA_PATH + '/' + "validation_project.csv", index=False)


def build_CWE_tree():
    CWE = json.load(open(DATA_PATH + '/' + 'CWE_info.json', 'r'))
    CWE_tree = dict()
    for cwe in CWE:
        cwe_id = int(cwe["CWE-ID"])
        CWE_tree[cwe_id] = cwe
        CWE_tree[cwe_id]["father"] = list()
        CWE_tree[cwe_id]["children"] = list()
        CWE_tree[cwe_id]["peer"] = list()
        CWE_tree[cwe_id]["relate"] = list()
    
    for cwe_id, cwe in CWE_tree.items():
        relations = cwe['Related Weaknesses'].split("::")
        for r in relations:
            if "VIEW ID:1000" in r:
                rr = r.split(":")
                target_id = int(rr[3])
                if "ChildOf" in rr:
                    cwe["father"].append(target_id)
                    CWE_tree[target_id]["children"].append(cwe_id)
                elif "PeerOf" in rr or "CanAlsoBe" in rr:
                    cwe["peer"].append(target_id)
                    CWE_tree[target_id]["peer"].append(cwe_id)
                elif "CanPrecede" in rr or "Requires" in rr:
                    cwe["relate"].append(target_id)
                    CWE_tree[target_id]["relate"].append(cwe_id)
    
    with open(DATA_PATH + '/' + "CWE_tree.json", 'w') as f:
        json.dump(CWE_tree, f, indent=4)


def get_pos_sample():
    # extract all the positive samples and link them to the correspoding CWE
    dataset = pd.read_csv(DATA_PATH + '/' + "dataset.csv", low_memory=False)
    dataset = dataset.fillna("")
    pos = dataset[dataset.Security_Issue_Full == 1]
    print(len(pos))   # 1111
    print(len(dataset))   # 4812

    pos = rm_unnamed_columns(pos)
    pos = pos.to_dict(orient="records")

    CVE = json.load(open(DATA_PATH + '/' + "CVE_dict.json", 'r'))
    for sample in pos:
        cve_id = sample["CVE_ID"]  # CVE_ID of all samples are valid
        sample['CWE_ID'] = CVE[cve_id]['CWE_ID']
        sample['CVE_Description'] = CVE[cve_id]['CVE_Description']
    
    with open(DATA_PATH + '/' + "pos_info.json", 'w') as f:
        json.dump(pos, f, indent=4)


def get_pos_sample_train():
    # extract all the positive samples and link them to the correspoding CWE
    dataset = pd.read_csv(DATA_PATH + '/' + "train_project.csv", low_memory=False)
    dataset = dataset.fillna("")
    pos = dataset[dataset.Security_Issue_Full == 1]
    print(len(pos))  # 1111
    print(len(dataset))  # 4812

    pos = rm_unnamed_columns(pos)
    pos = pos.to_dict(orient="records")

    CVE = json.load(open(DATA_PATH + '/' + "CVE_dict.json", 'r'))
    for sample in pos:
        cve_id = sample["CVE_ID"]  # CVE_ID of all samples are valid
        sample['CWE_ID'] = CVE[cve_id]['CWE_ID']
        sample['CVE_Description'] = CVE[cve_id]['CVE_Description']

    with open(DATA_PATH + '/' + "pos_info_train.json", 'w') as f:
        json.dump(pos, f, indent=4)


def pos_distribution():
    count_missing_CWE = 0
    POS = json.load(open(DATA_PATH + '/' + "pos_info.json", 'r'))
    CWE = json.load(open(DATA_PATH + '/' + "CWE_tree.json", 'r'))
    CWE_distribution = dict()
    for pos in POS:
        cve_id = pos['CVE_ID']
        cwe_id = pos['CWE_ID'] or "null"  # cwe_id = None
        if not CWE_distribution.get(cwe_id):
            CWE_distribution[cwe_id] = {'abstraction': None, '#issue report': 0, '#CVE': 0, 'CVE_distribution': dict()}
            # three special CWE categories: NVD-CWE-noinfo, NVD-CWE-Other, null(CVE's CWE value is missing) 
            if cwe_id not in ["NVD-CWE-noinfo", "NVD-CWE-Other", "null"]:
                id_ = cwe_id.split('-')[1]
                if CWE.get(id_):
                    CWE_distribution[cwe_id]['abstraction'] = CWE[id_]["Weakness Abstraction"]
                else:
                    count_missing_CWE += 1
                    print(cwe_id)

        CWE_distribution[cwe_id]['#issue report'] += 1
        if not CWE_distribution[cwe_id]['CVE_distribution'].get(cve_id):
            CWE_distribution[cwe_id]['CVE_distribution'][cve_id] = 0
            CWE_distribution[cwe_id]['#CVE'] += 1
        
        CWE_distribution[cwe_id]['CVE_distribution'][cve_id] += 1
    
    print(count_missing_CWE)  # 6
    with open(DATA_PATH + '/' + "CWE_distribution.json", 'w') as f:
        json.dump(CWE_distribution, f, indent=4)


def pos_distribution_train():
    count_missing_CWE = 0
    POS = json.load(open(DATA_PATH + '/' + "pos_info_train.json", 'r'))
    CWE = json.load(open(DATA_PATH + '/' + "CWE_tree.json", 'r'))
    CWE_distribution = dict()
    for pos in POS:
        cve_id = pos['CVE_ID']
        cwe_id = pos['CWE_ID'] or "null"  # cwe_id = None
        if not CWE_distribution.get(cwe_id):
            CWE_distribution[cwe_id] = {'abstraction': None, '#issue report': 0, '#CVE': 0, 'CVE_distribution': dict()}
            # three special CWE categories: NVD-CWE-noinfo, NVD-CWE-Other, null(CVE's CWE value is missing)
            if cwe_id not in ["NVD-CWE-noinfo", "NVD-CWE-Other", "null"]:
                id_ = cwe_id.split('-')[1]
                if CWE.get(id_):
                    CWE_distribution[cwe_id]['abstraction'] = CWE[id_]["Weakness Abstraction"]
                else:
                    count_missing_CWE += 1
                    print(cwe_id)

        CWE_distribution[cwe_id]['#issue report'] += 1
        if not CWE_distribution[cwe_id]['CVE_distribution'].get(cve_id):
            CWE_distribution[cwe_id]['CVE_distribution'][cve_id] = 0
            CWE_distribution[cwe_id]['#CVE'] += 1

        CWE_distribution[cwe_id]['CVE_distribution'][cve_id] += 1

    print(count_missing_CWE)  # 6
    with open(DATA_PATH + '/' + "CWE_distribution_train.json", 'w') as f:
        json.dump(CWE_distribution, f, indent=4)


def BFS(cwe_id, CWE_tree, level=1):
    level += 1
    sub_tree = list()
    queue = [cwe_id, -1]
    while level != 0 and len(queue) != 0:
        node = str(queue.pop(0))
        if node == "-1":
            level -= 1
            if len(queue) != 0:
                queue.append(-1)
            continue
        sub_tree.append(node) 
        queue.extend(CWE_tree[node]["children"] + CWE_tree[node]["peer"] + CWE_tree[node]["relate"])
    
    return sub_tree


def remove_repeat(original_list):
    no_repeat_list = [original_list[0]]
    for sample in original_list:
        if sample not in no_repeat_list:
            no_repeat_list.append(sample)
    return no_repeat_list


def add_end_seperator(s):
    # add seperators at the end of the input sentence(s) during merge
    s = s.strip()
    if s == "":
        return s
    # if re.match(r'[a-z0-9]', s[-1], re.I) is not None:
    if re.match(r'\.', s[-1]) is None:
        # not end with dot
        s += '.'
    s += ' '
    return s


def generate_description(cwe_id, CWE_tree, CVE_dict=None, CWE_distribution=None, num_cve_per_anchor=5):
    # generate descriptions for each anchor
    description = ""
    if cwe_id not in CWE_tree:
        cwe = CWE_distribution[f"CWE-{cwe_id}"]
        cve_belong_to_cwe = list(cwe['CVE_distribution'].keys())
        # for cve_id in random.choices(cve_belong_to_cwe, k = 3*num_cve_per_anchor):
        for cve_id in random.sample(cve_belong_to_cwe, k=min(3*num_cve_per_anchor, len(cve_belong_to_cwe))):
            description += add_end_seperator(replace_tokens_simple(CVE_dict[cve_id]["CVE_Description"]))  # preprocess the CVE description
    else:
        description += add_end_seperator(CWE_tree[cwe_id]['Name'])
        description += add_end_seperator(CWE_tree[cwe_id]["Description"])
        for item in CWE_tree[cwe_id]['Common Consequences'].split("::"):
            if "SCOPE" in item:
                flag = False
                for element in item.split(':'):
                    if flag and element not in ['IMPACT', 'NOTE']:
                        description += add_end_seperator(element)
                    if element == 'IMPACT':
                        flag = True
        description += add_end_seperator(CWE_tree[cwe_id]["Extended Description"])

        # items = CWE_tree[cwe_id]["Observed Examples"].split("::")
        # if items[0] == "null":
        #     pass
        # else:
        #     examples = [_[_.find("DESCRIPTION")+12: _.find("LINK")-1] for _ in items]
        #     # for exa in random.choices(examples, k = num_cve_per_anchor):
        #     for exa in random.sample(examples, k=min(num_cve_per_anchor, len(examples))):
        #         description += add_end_seperator(exa)

    return description


def build_anchor(level=1, num_cve_per_anchor=5):
    # build the external memory
    abstr_level = {"Pillar": 1, "Class": 2, "Base": 2.5, "Variant": 3, "Compound": 3}
    
    CWE_distribution_train = json.load(open(DATA_PATH + '/' + "CWE_distribution_train.json", 'r'))  # only use the train set
    CWE_tree = json.load(open(DATA_PATH + '/' + "CWE_tree.json", 'r'))  # dict
    CVE_dict = json.load(open(DATA_PATH + '/' + "CVE_dict.json", 'r'))  # dict
    
    CWE_anchor = dict()
    for id_, cwe in CWE_distribution_train.items():
        description = ""
        if id_ == "null":
            # corresponding CVE record miss CWE value. considered as dirty data
            continue

        cwe_id = id_.split("-")[1]
        cve_belong_to_cwe = list(cwe['CVE_distribution'].keys())  # randomness
        num_cve = len(cve_belong_to_cwe)
        if cwe_id not in CWE_tree:
            # for CWE not in the Research View, only using the CVE description (11 nodes + NVD-CWE-noinfo + NVD-CWE-Other).
            # for cve_id in random.choices(cve_belong_to_cwe, k=3*num_cve_per_anchor):
            for cve_id in random.sample(cve_belong_to_cwe, k=min(3*num_cve_per_anchor, num_cve)):
                description += add_end_seperator(replace_tokens_simple(CVE_dict[cve_id]["CVE_Description"]))
        else:
            related_cwe = BFS(cwe_id, CWE_tree, level)  # level: use BFS to search how many levels of the subtree rooted at cwe_id
            related_cwe = remove_repeat(related_cwe)
            related_cwe = [(_, abstr_level[CWE_tree[_]["Weakness Abstraction"]]) for _ in related_cwe]
            related_cwe.sort(key=lambda x:x[1])  # high-level nodes first and then low-level nodes
            for _ in related_cwe:
                description += generate_description(_[0], CWE_tree, num_cve_per_anchor = num_cve_per_anchor)

            # for cve_id in random.choices(cve_belong_to_cwe, k=num_cve_per_anchor):
            for cve_id in random.sample(cve_belong_to_cwe, k=min(num_cve_per_anchor, num_cve)):
                description += add_end_seperator(replace_tokens_simple(CVE_dict[cve_id]["CVE_Description"]))
        
        CWE_anchor[id_] = description.strip()
    
    print(len(CWE_anchor))  # 129
    
    with open(DATA_PATH + '/' + "CWE_anchor_golden_project.json", 'w') as f:
        json.dump(CWE_anchor, f, indent=4)


def rm_unnamed_columns(df):
    # used before writing a csv file (make sure we do not write Unamed columns)
    print(df.columns.tolist())
    valid_columns = list()
    for col in df.columns.tolist():
        if 'Unnamed' in col:
            continue
        valid_columns.append(col)
    
    df = df[valid_columns]
    print(df.columns.tolist())
    return df


def csv_to_json(f):
    # convert csv file to json
    # df = pd.read_csv(DATA_PATH + "1000.csv", header=0, index_col=False)  # for converting CWE(Research View)
    df = pd.read_csv(DATA_PATH + '/' + f"{f}.csv", low_memory=False)
    df.fillna("", inplace=True)
    
    # exclude unamed columns(index)
    # explicitly specify colums or use rm_unnamed_colums
    # df = df[["Issue_Url", "Issue_Created_At", "Issue_Title", "Issue_Body", "CVE_ID", "Published_Date", "Security_Issue_Full"]]
    # df = rm_unnamed_columns(df)

    records = df.to_dict(orient="records")

    with open(DATA_PATH + '/' + f"{f}.json", 'w') as ff:
        json.dump(records, ff, indent=4)


def rm_project_without_pos(f):
    df = pd.read_csv(DATA_PATH + f"{f}.csv", low_memory=False)
    df.fillna("", inplace=True)

    print(df["Security_Issue_Full"].value_counts())

    df["project"] = df["Issue_Url"].map(extract_project)
    project = list(set(df["project"].tolist()))
    # print(project)
    print(len(project))
    print(len(df[df.Security_Issue_Full == 1]))
    print(len(df[df.Security_Issue_Full == 0]))
    print(len(df))

    df["valid"] = df.groupby('project')['Security_Issue_Full'].transform('sum') 
    print(len(df[df.valid == 0]))
    print(len(df[df.valid == 0].drop_duplicates(subset=["project"])))

    # for p in project:
    #     if len(df[(df.project == p) & (df.Security_Issue_Full == 1)]) == 0:
    #         print("ERROR")

    df = df[df.valid != 0]
    print(len(set(df.project.to_list())))
    print(len(df[df.Security_Issue_Full == 1]))
    print(len(df[df.Security_Issue_Full == 0]))
    print(len(df))

    df[["Issue_Url", "Issue_Created_At", "Issue_Title", "Issue_Body", "CVE_ID", "Published_Date", "Security_Issue_Full"]].to_csv(DATA_PATH + f"{f}.csv")


def repo_info_stat(file):
    # stars and forks of the projects
    df = pd.read_csv(DATA_PATH + f"{file}.csv", low_memory=False)
    df.fillna("", inplace=True)
    df["project"] = df["Issue_Url"].map(extract_project)
    projects = set(df["project"].to_list())
    print(len(projects))

    repo_info = json.load(open(DATA_PATH + "repo_info.json", 'r'))  # stats of all the 1390 projects (8 are unable to retrieve)
    retrieved_projects = set(repo_info.keys())

    print("project unable to retrieve", len(projects - retrieved_projects))
    print(projects - retrieved_projects)

    projects = projects & retrieved_projects  # intersection

    star = [repo_info[p]["stargazers_count"] for p in projects]
    watch = [repo_info[p]["watchers_count"] for p in projects]
    fork = [repo_info[p]["forks_count"] for p in projects]
    subscribe = [repo_info[p]["subscribers_count"] for p in projects]

    print("star", np.median(star), np.average(star))
    print("watch", np.median(watch), np.average(watch))
    print("fork", np.median(fork), np.average(fork))
    print("subscribe", np.median(subscribe), np.average(subscribe))


def match_keyword(des):
    sec_related_word = r"(?i)(denial.of.service|\bxxe\b|remote.code.execution|\bopen.redirect|osvdb|\bvuln|\bcve\b|\bxss\b|\bredos\b|\bnvd\b|malicious|x−frame−options|attack|cross.site|exploit|directory.traversal|\brce\b|\bdos\b|\bxsrf\b|clickjack|session.fixation|hijack|advisory|insecure|security|\bcross−origin\b|unauthori[z|s]ed|infinite.loop|authenticat(e|ion)|bruteforce|bypass|constant.time|crack|credential|\bdos\b|expos(e|ing)|hack|harden|injection|lockout|overflow|password|\bpoc\b|proof.of.concept|poison|privelage|\b(in)?secur(e|ity)|(de)?serializ|spoof|timing|traversal)"
    if re.search(sec_related_word, des):
        return True
    return False


def match_steps_to_reproduce(x, keyword):
    # keyword is a regex
    x = x or ""
    if re.search(keyword, x, flags=re.I):
        return True
    return False


DATA_PATH = "data"

if __name__ == '__main__':
    # preprocess_dataset_csv("dataset")
    divide_dataset_project_csv("dataset")
    csv_to_json("train_project")
    csv_to_json("validation_project")
    csv_to_json("test_project")
    generate_dataset_mlm('train_project.json')
    generate_dataset_mlm('test_project.json')
    get_pos_sample()
    pos_distribution()
    get_pos_sample_train()
    pos_distribution_train()
    build_anchor()