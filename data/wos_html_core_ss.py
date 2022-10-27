import pandas as pd
from vespid.data.crossref import *
from vespid.data.make_dataset import *
from semanticscholar import SemanticScholar

api_key = 'H6LFRtg5Ar55rlCiBkr1k7YH2nr6DCvpa538wF5C'

def process_html_excel(email = None,htmldata_path=None):
    df = pd.read_excel(htmldata_path)
    df = df.reset_index().drop(columns = ['index'])
    df.columns = ['title','source','author_name','abstract','Citation_Index','date','tt_ab','exp_theory_label','doi','pdf']
    #对于没有doi的文献，通过crossref检索标题添加doi号
    df = add_publication_dois(df,email,score_difference_threshold=0.20,n_jobs=None)
    df = df[pd.notnull(df['DOI'])].reset_index().drop(columns = ['index'])
    return df

def concat_html_core(html_df,core_df):
    html_df.columns = ['title','source','author_name','abstract','page_count','date','tt_ab','exp_theory_label','doi','pdf']
    df = pd.merge(html_df,core_df,how = 'inner',on = 'doi',suffixes = ['_html','_core'])
    column_to_drop = ['title_html','source_html','abstract_html','date_html']
    df = df.drop(columns = column_to_drop).rename(columns = {'title_core':'title','source_core':'source','abstract_core':'abstract','date_core':'date'})
    return df

def add_ss_to_wos(df,api_key):
    #将ss的信息加入到df中
    output = add_semantic_scholar_to_wos(
        df = df, 
        api_key = api_key,
        max_concurrent_requests=10,
        n_jobs=-1
    ).sort_index()
    output['authors'] = output['authors_ss']
    return output

def add_crossref_ref_to_ss_wos(df,api_key,doicolumns = 'doi'):
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




    



