import json
import pandas as pd

def tree_parser(node, relation_list):
    for k, v in node.items():
        if len(v) > 0:
            tree_parser(v, relation_list)
        else:
            relation_list.append(k)

def project_to_csv_with_relation_layer_final():
    data1 = pd.read_csv("data/train_project.csv")
    data2 = pd.read_csv("data/test_project.csv")
    data = pd.concat([data1, data2])
    index = []
    text = []
    comment = []
    del_patch = []
    add_patch = []
    code = []
    message = []
    CVE_ID = []
    CWE_ID = []

    with open("data/CVE_dict.json", 'r') as f:
        content = f.read()
    table = json.loads(content)

    no_vul = 0
    for i in range(len(data)):
        if data.iloc[i]["Security_Issue_Full"] == 1:
            index.append("positive")
            CVE_ID.append(data.iloc[i]["CVE_ID"])
            if data.iloc[i]["CVE_ID"] in table.keys():
                now = table[data.iloc[i]["CVE_ID"]]["CWE_ID"]
                if now is None:
                    if CVE_ID[-1] == "CVE-2020-35579":
                        now = "CWE-404"
                    else:
                        print(table[data.iloc[i]["CVE_ID"]])
                if now == 'NVD-CWE-noinfo' or now == 'NVD-CWE-Other':
                    now = 'CWE-1000'
                CWE_ID.append(now)
            else:
                print(CVE_ID)
                raise ValueError
        else:
            index.append("negative")
            CVE_ID.append("CVE-0")
            CWE_ID.append("CWE-0")
        text.append(str(data.iloc[i]["Issue_Title"]) + " " + str(data.iloc[i]["Issue_Body"]))
        comment.append(str(data.iloc[i]["Comment_bodys"]))
        del_patch.append(data.iloc[i]["Patch_del"])
        add_patch.append(data.iloc[i]["Patch_add"])
        message.append(data.iloc[i]["Message"])
        code.append(data.iloc[i]["code"])

    # data2 = pd.DataFrame({"flag": index, "text": text, "code": code, "CVE_ID": CVE_ID, "CWE_ID": CWE_ID})
    # data2.to_csv("data/dataset_all_with_cwe.csv")

    CWE_ID = [element.replace('CWE-', "") for element in CWE_ID]
    unique_cwe_id = set(CWE_ID)
    num_classes = len(unique_cwe_id)

    data2 = pd.DataFrame({"flag": index, "text": text, "comment": comment, "message": message, "del_patch": del_patch, "add_patch": add_patch, "code": code, "CVE_ID": CVE_ID, "CWE_ID": CWE_ID})
    # data2.to_csv("data/dataset_all.csv")

    df = data2
    df['CWE_ID'] = df['CWE_ID'].astype(str)

    with open("data/my_CWE_tree.json", 'r') as f:
        content = f.read()
    cwe_tree = json.loads(content)

    cwe_relationship = {}

    for index in ['284', '435', '664', '682', '691', '693', '697', '703', '707', '710']:
        relation_list = []
        for k, v in cwe_tree['1000'][index].items():
            relation_list.append(k)
            if len(v) > 0:
                tree_parser(v, relation_list)
        cwe_relationship[index] = relation_list

    df.loc[df['CWE_ID'].isin(cwe_relationship['284']), 'CWE_ID'] = '284'
    df.loc[df['CWE_ID'].isin(cwe_relationship['435']), 'CWE_ID'] = '435'
    df.loc[df['CWE_ID'].isin(cwe_relationship['664']), 'CWE_ID'] = '664'
    df.loc[df['CWE_ID'].isin(cwe_relationship['682']), 'CWE_ID'] = '682'
    df.loc[df['CWE_ID'].isin(cwe_relationship['691']), 'CWE_ID'] = '691'
    df.loc[df['CWE_ID'].isin(cwe_relationship['693']), 'CWE_ID'] = '693'
    df.loc[df['CWE_ID'].isin(cwe_relationship['697']), 'CWE_ID'] = '697'
    df.loc[df['CWE_ID'].isin(cwe_relationship['703']), 'CWE_ID'] = '703'
    df.loc[df['CWE_ID'].isin(cwe_relationship['707']), 'CWE_ID'] = '707'
    df.loc[df['CWE_ID'].isin(cwe_relationship['710']), 'CWE_ID'] = '710'

    # CWE-664
    # df.loc[df['CWE_ID'].isin(['125', '119', '787', '772', '200', '770', '22', '908', '732', '94', '59', '681', '212', '763', '189', '522', '915', '1187', '672']), 'CWE_ID'] = '664'
    df.loc[df['CWE_ID'].isin(['125', '119', '787']), 'CWE_ID'] = '118'
    df.loc[df['CWE_ID'].isin(['772', '763']), 'CWE_ID'] = '404'
    df.loc[df['CWE_ID'].isin(['200', '22', '732', '522']), 'CWE_ID'] = '668'
    df.loc[df['CWE_ID'].isin(['770', '908', '1187']), 'CWE_ID'] = '665'
    df.loc[df['CWE_ID'].isin(['94', '915']), 'CWE_ID'] = '913'
    df.loc[df['CWE_ID'].isin(['59']), 'CWE_ID'] = '706'
    df.loc[df['CWE_ID'].isin(['681']), 'CWE_ID'] = '704'
    df.loc[df['CWE_ID'].isin(['212']), 'CWE_ID'] = '669'
    df.loc[df['CWE_ID'].isin(['672']), 'CWE_ID'] = '666'
   
    df.loc[df['CWE_ID'].isin(['913', '706', '704', '669', '666', '665']), 'CWE_ID'] = '1'

    df.loc[df['CWE_ID'].isin(['667']), 'CWE_ID'] = '691'

    df.loc[df['CWE_ID'].isin(['79', '89', '93', '77']), 'CWE_ID'] = '707'
    df.loc[df['CWE_ID'].isin(['674']), 'CWE_ID'] = '691'
    df.loc[df['CWE_ID'].isin(['295', '862', '863', '552', '254', '306', '320']), 'CWE_ID'] = '284'
    df.loc[df['CWE_ID'].isin(['331']), 'CWE_ID'] = '693'
    df.loc[df['CWE_ID'].isin(['252']), 'CWE_ID'] = '703'
    df.loc[df['CWE_ID'].isin(['399', '264', '189']), 'CWE_ID'] = '682'

    class_counts = df['CWE_ID'].value_counts()
    print(class_counts)

    threshold1 = 25
    single_sample_classes = class_counts[class_counts <= threshold1].index
    print(single_sample_classes)
    df.loc[df['CWE_ID'].isin(single_sample_classes), 'CWE_ID'] = '2'   # 693 435 697 703 417

    class_counts = df['CWE_ID'].value_counts()
    print(class_counts)

    label_dict = {}
    label_dict[0] = '0'
    label_dict[1] = '664'
    label_dict[2] = '707'
    label_dict[3] = '710'
    label_dict[4] = '682'
    label_dict[5] = '691'
    label_dict[6] = '2'
    label_dict[7] = '284'
    label_dict[8] = '1000'
    label_dict[9] = '118'
    label_dict[10] = '404'
    label_dict[11] = '668'
    label_dict[12] = '1'

    print(label_dict)
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    df['CWE_ID'] = df['CWE_ID'].map(reverse_label_dict)

    df.to_csv("data/dataset_project_relation_layer.csv")

if __name__ == "__main__":
    project_to_csv_with_relation_layer_final()
