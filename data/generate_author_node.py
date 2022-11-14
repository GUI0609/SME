import pandas as pd
from vespid.data.crossref import *
from vespid.data.make_dataset import *
from semanticscholar import SemanticScholar
import crossref_commons
from collections import Counter
import numpy as np
from crossref_commons import retrieval
from numpy import column_stack
from polyfuzz import PolyFuzz

from vespid.data.neo4j_tools import (
    Neo4jConnectionHandler, 
    test_graph_connectivity
)

from py2neo import Graph, Node, Relationship, NodeMatcher

graph = Neo4jConnectionHandler(
    db_ip='49.234.22.192',
            database='neo4j',
            db_username='neo4j',
            db_password='Ggl0609,',
            secure_connection=False)


api_key = 'H6LFRtg5Ar55rlCiBkr1k7YH2nr6DCvpa538wF5C'


df = pd.read_csv('/content/drive/MyDrive/数据文件/1111-2.csv').drop(columns = ['Unnamed: 0'])
author_msg = df[['DOI','author_names','author_names_core','authors_ss']]
author_msg_from_ss  = author_msg[pd.notnull(author_msg['authors_ss'])].reset_index().drop(columns = ['index'])[['DOI','authors_ss']]
author_msg_from_wos = author_msg[(pd.isnull(author_msg['authors_ss']))&(pd.notnull(author_msg['author_names']))].reset_index().drop(columns = ['index'])[['DOI','author_names']]
author_msg_from_core = author_msg[(pd.isnull(author_msg['authors_ss']))&(pd.isnull(author_msg['author_names']))&(pd.notnull(author_msg['author_names_core']))].reset_index().drop(columns = ['index'])[['DOI','author_names_core']]
print(len(author_msg_from_ss))
print(len(author_msg_from_wos))
print(len(author_msg_from_core))
print(len(author_msg[(pd.isnull(author_msg['authors_ss']))&(pd.isnull(author_msg['author_names']))&(pd.isnull(author_msg['author_names_core']))]))

#author_msg_from_ss
author_msg_from_ss['authors_ss'] = author_msg_from_ss['authors_ss'].apply(eval)
author_from_ss = get_unique_authors(author_msg_from_ss['authors_ss'],['authorId','name','url'])

#author_msg_from_wos
names = []
def convert_author_msg_from_wos(author_names_list):
  for author in author_names_list.split('; '):
    namelist = author.split('(')[-1].split(')')[0].split(' ')
    if len(namelist)==3:
        name = namelist[1]+' '+namelist[2][:-1]+' '+namelist[0][:-1]
        names.append(name)
    elif len(namelist)==2:
        name = namelist[1]+' '+namelist[0][:-1]
        names.append(name)
    elif len(namelist)==4:
        name = namelist[1]+' '+namelist[2]+' '+namelist[3]+' '+namelist[0][:-1]
        names.append(name)
    else:
        name = ' '.join(namelist)
        names.append(name)
for author_names_list in author_msg_from_wos['author_names'].tolist():
    convert_author_msg_from_wos(author_names_list)

no_match_names = [i for i in list(set(names)) if i not in author_from_ss['name'].tolist()]

def fuzzmatch(from_list,to_list):
    model = PolyFuzz("TF-IDF")
    model.match(from_list, to_list)
    match_result = model.get_matches()
    match_result = match_result[pd.notnull(match_result['To'])]
    match_result = match_result.loc[(match_result['From'].str[0]==match_result['To'].str[0])&(match_result['From'].str[-1]==match_result['To'].str[-1])].reset_index()[['From','To']]
    return match_result

from_list = names
to_list = author_from_ss['name'].tolist()
match_result = fuzzmatch(from_list,to_list)
wos_unique_author_name = [i for i in names if i not in match_result['From'].tolist()]
author_from_wos = pd.DataFrame({'name':wos_unique_author_name,'authorId':['wos_'+str(i) for i in range(len(wos_unique_author_name))]})

#author_msg_from_core
author_msg_from_core['author_names_core'] = author_msg_from_core['author_names_core'].apply(eval)
#因为只有一个，所以就直接[0],使用模糊匹配了，也没有
author_from_core = pd.DataFrame({'name':author_msg_from_core['author_names_core'][0],'authorId':['core_'+str(i) for i in range(len(author_msg_from_core['author_names_core'][0]))]})
author = pd.concat([author_from_ss,author_from_wos,author_from_core]).reset_index().drop(columns = ['index'])
author['authorId'] = author['authorId'].fillna('noid_0')


def make_author_nodes(
    df, 
    filepath=None,
    graph=None,
    batch_size=2_000
):
    '''
    Given data wherein each record is a publication, extract the unique
    authors and their metadata, then save to CSV.
    Parameters
    ----------
    df: pandas DataFrame that must contain, at least, the column 'authors',
        with data represented in that column as lists of dictionaries, one 
        dict per author on a given paper.
    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save node data.
    
    Returns
    -------
    Nodes object representing unique authors.
    '''

    columns_of_interest = [
        'authorId',
        'name',
        'url',
        #'aliases'
    ]

    properties = pd.DataFrame([
        ['name', 'name', np.nan],
        ['url', 'semanticScholarURL', np.nan],
        #['aliases', 'aliases', 'string[]']
    ], columns=['old', 'new', 'type'])
    
    if graph is not None:
        properties['type'] = np.nan

    author_nodes = Nodes(
        parent_label='Person', 
        additional_labels=['Author'],
        data=author, 
        id_column='authorId', 
        reference='author', 
        properties=properties
    )

    if filepath is not None:
        author_nodes.export_to_csv(filepath)
    
    elif graph is not None:
        # Check that constraint exists and create it if not
        logger.debug("Creating authors constraint if it doesn't exist...")
        query = "CREATE CONSTRAINT people IF NOT EXISTS ON (a:Person) ASSERT a.id IS UNIQUE"
        graph.cypher_query_to_dataframe(query, verbose=False)
        
        logger.info("Saving author nodes data to Neo4j...")
        author_nodes.export_to_neo4j(graph, batch_size=batch_size)
        
    return author_nodes

make_author_nodes(author,graph = graph)

#生成作者和paper的关系
# au_pp_r_from_ss
au_pp_r_from_ss = author_msg_from_ss.dropna().explode('authors_ss').dropna().reset_index().drop(columns = ['index'])
au_pp_r_from_ss['authors_ss'] = au_pp_r_from_ss['authors_ss'].apply(lambda x:x['authorId'])
au_pp_r_from_ss = au_pp_r_from_ss.dropna().reset_index().drop(columns = ['index'])
au_pp_r_from_ss = au_pp_r_from_ss.rename(columns = {'authors_ss':'authorId'})

# author_msg_from_wos
def convert_author_msg_from_wos_2(author):
  namelist = author.split('(')[-1].split(')')[0].split(' ')
  if len(namelist)==3:
    name = namelist[1]+' '+namelist[2][:-1]+' '+namelist[0][:-1]
  elif len(namelist)==2:
    name = namelist[1]+' '+namelist[0][:-1]
  elif len(namelist)==4:
    name = namelist[1]+' '+namelist[2]+' '+namelist[3]+' '+namelist[0][:-1]
  else:
    name = ' '.join(namelist)
  return name
author_msg_from_wos['author_names'] = author_msg_from_wos['author_names'].apply(lambda x:x.split('; '))

author_msg_from_wos = author_msg_from_wos.dropna().explode('author_names').dropna().reset_index().drop(columns = ['index'])
author_msg_from_wos['author_names'] = author_msg_from_wos['author_names'].apply(convert_author_msg_from_wos_2)
match_result_dict = match_result.set_index(['From'])['To'].to_dict()
# match_result_dict
author_msg_from_wos['author_names_match'] = author_msg_from_wos['author_names'].map(match_result_dict)
author_msg_from_wos = author_msg_from_wos[pd.notnull(author_msg_from_wos['author_names_match'])][['DOI','author_names_match']].rename(columns = {'author_names_match':'author_names'}).append(author_msg_from_wos[pd.isnull(author_msg_from_wos['author_names_match'])][['DOI','author_names']]).reset_index().drop(columns = ['index'])
# author
author_dict = author.set_index(['name'])['authorId'].to_dict()
# match_result_dict
author_msg_from_wos['authorId'] = author_msg_from_wos['author_names'].map(author_dict)
au_pp_r_from_wos = author_msg_from_wos[['DOI','authorId']]


#author_msg_from_core
author_msg_from_core['author_names_core'] = author_msg_from_core['author_names_core'].apply(eval)
au_pp_r_from_core = pd.DataFrame({'DOI':author_msg_from_core['DOI'][0],'authorId':['core_'+str(i) for i in range(len(author_msg_from_core['author_names_core'][0]))]})
au_pp_r = pd.concat([au_pp_r_from_ss,au_pp_r_from_wos,au_pp_r_from_core]).reset_index().drop(columns = ['index'])

paper_node_msg = pd.read_csv('1111-2.csv')
doiid_dict = paper_node_msg.set_index(['DOI'])['id'].to_dict()
au_pp_r['id'] = au_pp_r['DOI'].map(doiid_dict)
doidate_dict = paper_node_msg.set_index(['DOI'])['date'].to_dict()
au_pp_r['date'] = au_pp_r['DOI'].map(doidate_dict)

#FROM author node
columns_of_interest = [
        'authorId',
        'name',
        'url',
        #'aliases'
    ]

properties = pd.DataFrame([
    ['name', 'name', np.nan],
    ['url', 'semanticScholarURL', np.nan],
    #['aliases', 'aliases', 'string[]']
], columns=['old', 'new', 'type'])

author_nodes = Nodes(
    parent_label='Person', 
    additional_labels=['Author'],
    data=author, 
    id_column='authorId', 
    reference='author', 
    properties=properties
)


#TO paper node
properties = pd.DataFrame(
        [['date', 'publicationDate', np.nan],
        ['DOI', 'doi', np.nan],
        ['Citation_Index', 'Citation_Index', np.nan], 
        ['pdf', 'pdf', 'string[]'], 
        ['exp_theory_label', 'exp_theory_label', 'string[]'], 
        ['source', 'publicationName', np.nan],
        # ['authors_ss', 'authorNames', 'string[]'],
        ['title', 'title', np.nan],
        # ['fund_text', 'fundingText', np.nan],
        ['tt_ab', 'tt_ab', np.nan],
        ['abstract', 'abstract', np.nan],
        ['ref_id', 'ref_id', np.nan],
        ['page_count', 'page_count', np.nan],
        ['doctype', 'publicationDocumentTypes', 'string[]'],
        ['cat_subject', 'cat_subject', 'string[]'],
        ['id_ss', 'semanticScholarID', np.nan],
        ['url_ss', 'semanticScholarURL', np.nan],
        ['influentialCitationCount', 'influentialCitationCount', np.nan],
        ['author_email','author_email', 'float[]'],
        ['fieldsOfStudy', 'fieldsOfStudy', 'string[]']],columns=['old', 'new', 'type'])
SME_node = Nodes(
    parent_label='Publication', 
    data=paper_node_msg, 
    id_column='id', 
    reference='SME', 
    properties=properties
)



#开始生成作者关系
def make_authorship_relationships(
    df,
    author_nodes,
    paper_nodes,
    filepath=None,
    graph=None,
    batch_size=2_000
    ):
    '''
    Derives the relationship between an author of a publication and the 
    publication they wrote. Also saves the results in a format that Neo4j
    can ingest.
    Parameters
    ----------
    df: pandas DataFrame containing one record per author-publication 
        connection.
    author_nodes: Nodes object referring to the unique authors being ingested.
    paper_nodes: Nodes object referring to the publications being ingested.
    filepath: str. indicates where the CSV for neo4j ingest should be written.
        Should be of the form 'path/to/file.csv'. If None, ``graph`` must not
        be None.
        
    graph: Neo4jConnection object. If not None, indicates that a Neo4j
        graph should be used as the place to save relationship data.
    Returns
    -------
    Relationships object with each unique authorship.
    '''

    # Columns with data we don't need to retain for the authorship connections
    # Mostly used for earlier ETL steps or node creation

    properties = pd.DataFrame(
        [
            ['date', 'publicationDate', 'datetime'],
        ],
        columns=['old', 'new', 'type']
    )
    
    if graph is not None:
        properties['type'] = np.nan
        
    output = Relationships(
        type='WROTE',
        data=au_pp_r,
        start_node=author_nodes,
        id_column_start='authorId',
        id_column_end='id',
        end_node=paper_nodes,
        allow_unknown_nodes=False,
        properties=properties
    )

    if filepath is not None:
        logger.info(f"Saving {len(output)} relationships to disk...")
        output.export_to_csv(filepath)
        
    elif graph is not None:
        logger.info(f"Saving {len(output)} authorship relationships to Neo4j...")
        output.export_to_neo4j(graph, batch_size=batch_size)

    return output


    
make_authorship_relationships(au_pp_r,author_nodes,SME_node,graph = graph)