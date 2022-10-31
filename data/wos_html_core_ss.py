import pandas as pd
from vespid.data.crossref import *
from vespid.data.make_dataset import *
from semanticscholar import SemanticScholar
import crossref_commons
from collections import Counter
import numpy as np

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
    df = pd.merge(html_df,core_df,how = 'inner',on = 'DOI',suffixes = ['_html','_core'])
    column_to_drop = ['title_html','source_html','abstract_html','date_html']
    df = df.drop(columns = column_to_drop).rename(columns = {'title_core':'title','source_core':'source','abstract_core':'abstract','date_core':'date','page_count_html':'page_count','author_names_html':'author_names'})
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

#对没有出版年份的文献进行修正
def correct_without_year(df):
    cor_date = df[df['date']!=0].reset_index().drop(columns = ['index'])
    incor_date = df[df['date']==0].reset_index().drop(columns = ['index'])
    for index in range(len(incor_date)):
        result = crossref_commons.retrieval.get_publication_as_json(incor_date['DOI'].loc[index])
        incor_date['date'].loc[index] = result['created']['date-parts'][0][0]
    df2 = cor_date.append(incor_date).reset_index().drop(columns = ['index'])
    return df2