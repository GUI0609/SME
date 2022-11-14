#wos_html_core_ss.py
import pandas as pd
from vespid.data.crossref import *
from vespid.data.make_dataset import *
from semanticscholar import SemanticScholar
import crossref_commons
from collections import Counter
import numpy as np
from crossref_commons import retrieval
from numpy import column_stack

api_key = 'H6LFRtg5Ar55rlCiBkr1k7YH2nr6DCvpa538wF5C'

def process_html_excel(email = None,htmldata_path=None):
    #对于没有doi的文献，通过crossref检索标题添加doi号
    df = pd.read_excel(htmldata_path)
    df = df.reset_index().drop(columns = ['index'])
    error_doi=[]
    for (i,j) in Counter(df['Doi'].tolist()).items():
        if j>1:
            error_doi.append(i)
    for index in range(len(df)):
        if str(df.loc[index,'Doi']) in error_doi:
            df.loc[index,'Doi'] = np.nan
    
    df.columns = ['title','source','author_names','abstract','Citation_Index','date','tt_ab','exp_theory_label','DOI','pdf']
    df = add_publication_dois(df,email,score_difference_threshold=0.20,n_jobs=None)
    df = df[pd.notnull(df['DOI'])].reset_index().drop(columns = ['index'])
    return df

def concat_html_core(html_df,core_df):
    # html_df.columns = ['title', 'date', 'DOI', 'abstract', 'source', 'author_names','Citation_Index', 'tt_ab', 'exp_theory_label', 'pdf']
    df = pd.merge(html_df,core_df,how = 'left',on = 'DOI',suffixes = ['_html','_core'])
    column_to_drop = ['title_core','source_core','abstract_core','date_core']
    df = df.drop(columns = column_to_drop).rename(columns = {'title_html':'title','source_html':'source','abstract_html':'abstract','date_html':'date','page_count_html':'page_count','author_names_html':'author_names'})
    return df

def add_ss_to_wos(df,api_key):
    #将ss的信息加入到df中
    output = add_semantic_scholar_to_wos(
        df = df, 
        api_key = api_key,
        max_concurrent_requests=10,
        n_jobs=-1
    ).sort_index()
    output = output.rename(columns = {'authors':'authors_ss'})
    return output

def add_crossref_ref_to_ss_wos(df,api_key,doicolumns = 'DOI'):
    def get_ss_ref(doi):
        try:
            sch = SemanticScholar(api_key=api_key)
            paper = sch.get_paper(doi)
            if 'references' in paper.keys() and paper.references!=[]:
                ref_ss = paper.references
            elif 'citations' in paper.keys() and paper.citations!=[]:
                ref_ss = paper.citations
            else:
                ref_ss = None
            return ref_ss
        except:
            return None
    df['ref_ss'] = df[doicolumns].apply(get_ss_ref)

    return df


#综合修正函数
def correct_missing_msg(df,api_key):
    #对没有出版年份的文献进行修正
    def _correct_without_date(df):
        cor_date = df[df['date']!=0].reset_index().drop(columns = ['index'])
        incor_date = df[df['date']==0].reset_index().drop(columns = ['index'])
        for index in range(len(incor_date)):
            result = crossref_commons.retrieval.get_publication_as_json(incor_date['DOI'].loc[index])
            incor_date['date'].loc[index] = result['created']['date-parts'][0][0]
        df2 = cor_date.append(incor_date).reset_index().drop(columns = ['index'])
        return df2
    #对没有引用数的文献进行修正
    def _correct_without_Citation_Index(df,api_key):
        cor_date = df[pd.notnull(df['Citation_Index'])].reset_index().drop(columns = ['index'])
        incor_date = df[pd.isnull(df['Citation_Index'])].reset_index().drop(columns = ['index'])
        print(len(incor_date))
        for index in range(len(incor_date)):
            sch = SemanticScholar(api_key=api_key)
            try:
                result = sch.get_paper(incor_date['DOI'].loc[index])
                incor_date['Citation_Index'].loc[index] = result['numCitedBy']
            except:
                incor_date['Citation_Index'].loc[index] = None
        df2 = cor_date.append(incor_date).reset_index().drop(columns = ['index'])
        return df2

    #对没有期刊的文献进行修正
    def _correct_without_source(df):
        cor_date = df[pd.notnull(df['source'])].reset_index().drop(columns = ['index'])
        incor_date = df[pd.isnull(df['source'])].reset_index().drop(columns = ['index'])
        print(len(incor_date))
        for index in range(len(incor_date)):
            try:
                result = crossref_commons.retrieval.get_publication_as_json(incor_date['DOI'].loc[index])
                incor_date['source'].loc[index] = result['container-title'][0].upper()
            except:
                incor_date['source'].loc[index] = None
        df2 = cor_date.append(incor_date).reset_index().drop(columns = ['index'])
        return df2

    #对没有tt_ab的文献进行修正
    def _correct_without_tt_ab(df):
        cor_date = df[pd.notnull(df['tt_ab'])].reset_index().drop(columns = ['index'])
        incor_date = df[pd.isnull(df['tt_ab'])].reset_index().drop(columns = ['index'])
        print(len(incor_date))
        for index in range(len(incor_date)):
            result = str(incor_date['title'].loc[index])+' '+str(incor_date['abstract'].loc[index])
            incor_date['tt_ab'].loc[index] = result
        df2 = cor_date.append(incor_date).reset_index().drop(columns = ['index'])
        return df2

    #对没有引用信息的文献进行修正
    def _correct_without_ref(df):
        cor_date = df[(pd.notnull(df['reference']))|(pd.notnull(df['ref_ss']))].reset_index().drop(columns = ['index'])
        incor_date = df[(pd.isnull(df['reference']))&(pd.isnull(df['ref_ss']))].reset_index().drop(columns = ['index'])
        print(len(incor_date))
        for index in range(len(incor_date)):
            try:
                result = crossref_commons.retrieval.get_publication_as_json(incor_date['DOI'].loc[index])['reference']
            except:
                result = None
            incor_date['crossref_reference'].loc[index] = result
          
        df2 = cor_date.append(incor_date).reset_index().drop(columns = ['index'])
        return df2
    def _correct_without_date2(df):
        cor_date = df[pd.notnull(df['date'])].reset_index().drop(columns = ['index'])
        incor_date = df[pd.isnull(df['date'])].reset_index().drop(columns = ['index'])
        print(len(incor_date))
        for index in range(len(incor_date)):
            try:
                result = crossref_commons.retrieval.get_publication_as_json(incor_date['DOI'].loc[index])
                incor_date['date'].loc[index] = result['published']['date-parts'][0][0]
            except:
                incor_date['date'].loc[index] = 0
        df2 = cor_date.append(incor_date).reset_index().drop(columns = ['index'])
        return df2



    df = _correct_without_date(df)
    df = _correct_without_Citation_Index(df,api_key)
    df = _correct_without_source(df)
    df = _correct_without_tt_ab(df)
    df = _correct_without_ref(df)
    df = _correct_without_date2(df)
    return df

if __name__ == '__main__':
    #从html文件开始
    html_df = process_html_excel(email = email,htmldata_path='/content/SME/data/demo/920.xlsx')
    #添加第一轮导出的高引用文献，并使用01排除是否为本领域文献
    export_1025_LHJ = pd.read_excel('/content/SME/data/demo/export_1025_LHJ.xlsx')
    export_1025_LHJ = export_1025_LHJ[export_1025_LHJ['TorF']==1].drop(columns = ['TorF'])
    export_1025_LHJ.columns = ['title','date','DOI','abstract']
    df_html_LHJ = pd.concat([html_df,export_1025_LHJ],sort=False).reset_index().drop(columns = ['index'])
    #合并wos_core文件
    df_core_wos = None#此处需要修改路径
    df = concat_html_core(df_html_LHJ,df_core_wos)[['title','abstract','DOI','date']]
    df = add_ss_to_wos(df,api_key)
    df = add_crossref_ref_to_ss_wos(df,api_key,doicolumns = 'DOI')
    df = correct_missing_msg(df,api_key)
    #df为1111-2.csv
    # df = pd.read_csv('1111-2.csv')