"""
数据来源 wos核心数据集
数据样本在wos_data_core_ref_txt_orgexample.txt
"""
import os
import re
import yaml
import pandas as pd

#genarate df_wos
# WHAT IS ALL FUNCTION NEED IS A MAIN_PATH WITH ALL TXT FROM WOS_CORE_DATASET
# main_path = r'D:\实验室电脑\SME1018\SME\data\demo'

def deal(list_ori,p):   
    #将txt文件分为['KEY VALUE','KEY VALUE']
    list_new=[]
    list_short=[]
    for i in list_ori:
        if i!=p:
            list_short.append(i)
        else:
            list_new.append(list_short)
            list_short=[]
    list_new.append(list_short)
    return list_new



def extract_doi_kvlist_from_txt(txt_folder_path,histcite = False,histcite_folder_path = None, use_df_doi = False,df_doi = None,use_dict_to_yaml = True,yaml_path = 'wos.yaml'):
    """
    将一个txt处理为{doi:['KEY VALUE','KEY VALUE']}
    txt_folder_path:从wos下载的txt统一存入同一文件夹中分析 该参数为文件夹的位置
    histcite = False 是否进行histcite分析
    histcite_folder_path = None 用于储存histcite分析的TXT文件夹路径
    use_df_doi 是否使用预筛选的doi列表
    df_doi doi列表
    use_dict_to_yaml 是否储存生成的字典
    yaml_path 字典路径
    """
    dict_to_yaml = {}
    DOI = 'DI '
    path = [os.path.join(txt_folder_path,i) for i in os.listdir(txt_folder_path)]
    for txt_id,txt_path in enumerate(path):
        with open(txt_path,'r',encoding = 'utf-8') as f:
            data = f.readlines()
        data1 = deal(data[2:],'\n')
        df_dict = {}
        for index,paper in enumerate(data1):
            dict_key = 'nodoi_'+str(index)
            for i in paper:
                if re.search(DOI,i):
                    dict_key = i
            if dict_key != 'nodoi_'+str(index):     
                df_dict[dict_key]=paper

        
        datadoi = [i[3:-1] for i in df_dict.keys()]


        for i in datadoi:
            if use_df_doi:
                if i in df_doi and i not in dict_to_yaml.keys():
                    try:
                        dict_to_yaml[i] = df_dict['DI '+i+'\n']
                    except:
                        continue
            else:
                if i not in dict_to_yaml.keys():
                    try:
                        dict_to_yaml[i] = df_dict['DI '+i+'\n']
                    except:
                        continue
        ##histcite##
        if histcite:
            al_doi = []
            with open(os.path.join(histcite_folder_path,str(txt_id)+'.txt'),'w',encoding = 'utf-8') as f:
                for i in data[:2]:
                    f.write(i)
                for i in datadoi:
                    if use_df_doi:
                        if i in df_doi and i not in al_doi:
                            try:
                                al_doi.append(i)
                                l = df_dict['DI '+i+'\n']
                                for j in l:
                                    f.write(j)
                                f.write('\n')
                            except:
                                continue
                    else:
                        if i not in al_doi:
                            try:
                                al_doi.append(i)
                                l = df_dict['DI '+i+'\n']
                                for j in l:
                                    f.write(j)
                                f.write('\n')
                            except:
                                continue
                f.write('EF')
    ##histcite##
    if use_dict_to_yaml:
        fp = open(yaml_path, 'w')
        fp.write(yaml.dump(dict_to_yaml))
        fp.close()
        
    print(f'Totle {len(dict_to_yaml.keys())} doi in wos')
    return dict_to_yaml



def convert_dict_to_df(wos_dict,output = None):
    df_list = []
    for i in list(wos_dict.values()):
        msg_kv = {}
        label = 'PT'
        content = ['J']
        for j in i[1:]:
            if j[0]!=' ' and j!='ER\n':
                msg_kv[label] = content
                label = j[:2]
                content = [j[3:-1]]
            else:
                content.append(j[3:-1])
        df_list.append(msg_kv)
    
    df = pd.DataFrame(df_list)
    org_columns = df.columns.tolist()

    def i_0(x):
        try:
            return x[0]
        except:
            return None
    df['date'] = df['PY'].apply(i_0)
    df['ref_id'] = df['UT'].apply(i_0)
    df['source'] = df['SO'].apply(i_0)
    df['pubtype'] = df['DT'].apply(i_0)
    df['author_email'] = df['EM'].apply(i_0)
    df['page_count'] = df['PG'].apply(i_0)
    df['abstract'] = df['AB'].apply(i_0)
    df['doctype'] = df['DT'].apply(i_0)
    df['doi'] = df['DI'].apply(i_0)
    df['abstract'] = df['AB'].apply(i_0)

    df['author_name'] = df['AF']
    df['author_address'] = df['C1']
    df['co_author_address'] = df['RP']
    df['author_name'] = df['AF']
    df['reference'] = df['CR']


    df['title'] = df['TI'].apply(lambda x:' '.join(x))
    df['author_last'] = df['AF'].apply(lambda x:x[-1])

    def i_0_split(x):
        try:
            return x[0].split(';')
        except:
            return None
    df['cat_subject'] = df['SC'].apply(i_0_split)

    def joini_0_split(x):
        try:
            return ' '.join(x).split(';')
        except:
            return None
    df['fund_text'] = df['FU'].apply(joini_0_split)

    df = df.drop(columns = org_columns)
    if output:
        df.to_excel(output)
    else:
        return df


