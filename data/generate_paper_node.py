import pandas as pd
from vespid.data.crossref import *
from vespid.data.make_dataset import *
from semanticscholar import SemanticScholar
import crossref_commons
from collections import Counter
import numpy as np
from crossref_commons import retrieval
from numpy import column_stack

from vespid.data.neo4j_tools import (
    Neo4jConnectionHandler, 
    test_graph_connectivity
)

from py2neo import Graph, Node, Relationship, NodeMatcher

graph = Neo4jConnectionHandler(
    db_ip='49.234.22.192',
            database='neo4j',
            db_username='neo4j',
            db_password='********',
            secure_connection=False)


api_key = 'H6LFRtg5Ar55rlCiBkr1k7YH2nr6DCvpa538wF5C'

wos_ss_core_path = '1111-2.csv'
df = pd.read_csv(wos_ss_core_path)
#为每篇SME的文献生成id
df = df.reset_index()
df['id'] = df['index'].apply(lambda x:'SME_'+str(x))
df = df.drop(columns = ['index'])
#生成文献节点
paper_node_msg = df[['title','Citation_Index','exp_theory_label','DOI','pdf','date', 'ref_id', 'source','page_count','doctype','id_ss','tt_ab',
       'url_ss','influentialCitationCount','author_email','abstract','cat_subject','fieldsOfStudy','id']]
#此处的文献既包括SME文献，也包括引用文献
ref_node_msg = df[['DOI','reference','ref_ss','crossref_reference']]

ref_from_ss  = ref_node_msg[pd.notnull(ref_node_msg['ref_ss'])].reset_index().drop(columns = ['index'])[['DOI','ref_ss']]
ref_from_wos = ref_node_msg[(pd.isnull(ref_node_msg['ref_ss']))&(pd.notnull(ref_node_msg['reference']))].reset_index().drop(columns = ['index'])[['DOI','reference']]
ref_from_crossref = ref_node_msg[(pd.isnull(ref_node_msg['ref_ss']))&(pd.notnull(ref_node_msg['crossref_reference']))&(pd.isnull(ref_node_msg['reference']))].reset_index().drop(columns = ['index'])[['DOI','crossref_reference']]
print(f'the count of reference from semantic scholar is: {len(ref_from_ss)}')
print(f'the count of reference from wos_html is: {len(ref_from_wos)}')
print(f'the count of reference from crossref is: {len(ref_from_crossref)}')
print('no reference')
print(len(ref_node_msg[(pd.isnull(ref_node_msg['reference']))&(pd.isnull(ref_node_msg['ref_ss']))&(pd.isnull(ref_node_msg['crossref_reference']))]))

# ref_from_ss
#如果来源于读取的文件，记得eval
try:
    ref_from_ss['ref_ss'] = ref_from_ss['ref_ss'].apply(eval)
except:
    pass
ref_from_ss_df = pd.DataFrame()
for i in range(len(ref_from_ss)):
  ref_df = pd.DataFrame(ref_from_ss['ref_ss'][i])
  ref_df['DOI']=[ref_from_ss['DOI'][i]]*len(ref_from_ss['ref_ss'][i])
  ref_from_ss_df = ref_from_ss_df.append(ref_df)
ref_from_ss_df = ref_from_ss_df.reset_index().drop(columns = ['index'])

#ref_from_wos
ref_from_wos_df = []
for i in range(len(ref_from_wos)):
  for j in [i.split('DOI ')[-1] for i in eval(ref_from_wos['reference'][i])]:
    if j.startswith('10.'):
      ref_from_wos_df.append((ref_from_wos['DOI'][i],j))
    elif j.startswith('[10'):
      ref_from_wos_df.append((ref_from_wos['DOI'][i],j.split(',')[0][1:]))
ref_from_wos_df = pd.DataFrame(ref_from_wos_df,columns = ['DOI','doi'])
ref_from_wos_df = ref_from_wos_df.dropna().reset_index().drop(columns = ['index'])
ref_from_wos_df2 = pd.merge(ref_from_wos_df,ref_from_ss_df,how='left',on = 'doi').drop_duplicates(subset = ['DOI_x','doi']).rename(columns = {'DOI_x':'DOI'}).reset_index().drop(columns = ['DOI_y','index'])
import tqdm
cor_ref_msg = ref_from_wos_df2[pd.notnull(ref_from_wos_df2['title'])].reset_index().drop(columns = ['index'])
incor_ref_msg = ref_from_wos_df2[pd.isnull(ref_from_wos_df2['title'])].reset_index().drop(columns = ['index'])
for index in tqdm.tqdm(range(len(incor_ref_msg))):
    try:
        sch = SemanticScholar(api_key=api_key)
        result = sch.get_paper(incor_ref_msg['doi'].loc[index])
        incor_ref_msg['venue'].loc[ ] = result['venue']
        incor_ref_msg['title'].loc[index] = result['title']
        incor_ref_msg['year'].loc[index] = result['year']
        incor_ref_msg['authors'].loc[index] = result['authors']
    except:
        continue
ref_from_wos_df2 = cor_ref_msg.append(incor_ref_msg).reset_index().drop(columns = ['index'])

#整合生成ref_node_and_relation
ref_node_and_relation = ref_from_ss_df.append(ref_from_wos_df2)

#ref_from_crossref
ref_from_crossref_df = pd.DataFrame()
for i in range(len(ref_from_crossref)):
    ref_df = pd.DataFrame(eval(ref_from_crossref['crossref_reference'][i]))
    ref_df['paper_DOI']=[ref_from_crossref['DOI'][i]]*len(eval(ref_from_crossref['crossref_reference'][i]))
    ref_from_crossref_df = ref_from_crossref_df.append(ref_df)
ref_from_crossref_df = ref_from_crossref_df[['DOI','paper_DOI']].rename(columns = {'DOI':'doi','paper_DOI':'DOI'}).dropna().reset_index().drop(columns = ['index'])
ref_from_crossref_df2 = pd.merge(ref_from_crossref_df,ref_from_ss_df,how='left',on = 'doi').drop_duplicates(subset = ['DOI_x','doi']).rename(columns = {'DOI_x':'DOI'}).reset_index().drop(columns = ['DOI_y','index'])
import tqdm
cor_ref_msg = ref_from_crossref_df2[pd.notnull(ref_from_crossref_df2['title'])].reset_index().drop(columns = ['index'])
incor_ref_msg = ref_from_crossref_df2[pd.isnull(ref_from_crossref_df2['title'])].reset_index().drop(columns = ['index'])
for index in tqdm.tqdm(range(len(incor_ref_msg))):
    try:
        sch = SemanticScholar(api_key=api_key)
        result = sch.get_paper(incor_ref_msg['doi'].loc[index])
        incor_ref_msg['venue'].loc[index] = result['venue']
        incor_ref_msg['title'].loc[index] = result['title']
        incor_ref_msg['year'].loc[index] = result['year']
        incor_ref_msg['authors'].loc[index] = result['authors']
    except:
        continue
ref_from_crossref_df2 = cor_ref_msg.append(incor_ref_msg).reset_index().drop(columns = ['index'])
#整合crossref
ref_node_and_relation = ref_node_and_relation.append(ref_from_crossref_df2)
ref_node_and_relation = ref_node_and_relation.reset_index().drop(columns = ['index'])

#生成所有的引用文献节点，包括SME中的文献
reference_node_and_SME = ref_node_and_relation.drop_duplicates(subset = ['doi','title']).reset_index()
reference_node_and_SME['id_ref'] = reference_node_and_SME['index'].apply(lambda x:'ref_'+str(x))
#把在SME中的文献挑出来
reference_node_and_SME2 = pd.merge(reference_node_and_SME,paper_node_msg[['DOI','id']],how = 'left',left_on = 'doi',right_on = 'DOI')
reference_node_and_SME2 = reference_node_and_SME2[pd.isnull(reference_node_and_SME2['id'])].reset_index().drop(columns = ['level_0','index','DOI_x','DOI_y','id']).rename(columns = {'id_ref':'id'})
reference_node_and_SME2 = reference_node_and_SME2[['title','doi','id','year','venue']].rename(columns = {'year':'date','venue':'source','doi':'DOI'})

all_paper_node = paper_node_msg.append(reference_node_and_SME2)
all_paper_node = all_paper_node.reset_index().drop(columns=['index'])
# 按照doi进行聚合
relation1 = ref_node_and_relation[pd.notnull(ref_node_and_relation['doi'])]
relation1_df = pd.merge(relation1,all_paper_node[['DOI','title','id']],how = 'left',left_on = 'doi',right_on='DOI')
relation1_df = pd.merge(relation1_df,all_paper_node[['DOI','title','id']],how = 'left',left_on = 'DOI_x',right_on='DOI')
#没有doi的文献，按照title进行聚合
relation2 = ref_node_and_relation[(pd.isnull(ref_node_and_relation['doi']))&(pd.notnull(ref_node_and_relation['title']))]
relation2_df = pd.merge(relation2,all_paper_node[['DOI','title','id']],how = 'left',left_on = 'title',right_on='title')
relation2_df = pd.merge(relation2_df,all_paper_node[['DOI','title','id']],how = 'left',left_on = 'DOI_x',right_on='DOI')
relation = relation1_df[['id_x','id_y','year']].append(relation2_df[['id_x','id_y','year']])
relation = relation.drop_duplicates()


#生成paper_node

properties = pd.DataFrame(
        [['date', 'publicationDate', np.nan],
        ['DOI', 'doi', np.nan],
        ['Citation_Index', 'Citation_Index', np.nan], 
        ['pdf', 'pdf', 'string[]'], 
        ['exp_theory_label', 'exp_theory_label', 'string[]'], 
        ['source', 'publicationName', np.nan],
        ['authors_ss', 'authorNames', 'string[]'],
        ['title', 'title', np.nan],
        ['fund_text', 'fundingText', np.nan],
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



nodes = Nodes(
        parent_label='Publication', 
        data=all_paper_node, 
        id_column='id', 
        reference='paper_node', 
        properties=properties
    )

query = "CREATE CONSTRAINT publications IF NOT EXISTS ON (p:Publication) ASSERT p.id IS UNIQUE"
graph.cypher_query_to_dataframe(query, verbose=False)
batch_size = 2_000
nodes.export_to_neo4j(graph, batch_size=batch_size)

#生成paper_relation
rela_properties = pd.DataFrame(
    [
        ['year', 'publicationDate', np.nan]
    ],
    columns=['old', 'new', 'type']
)
if graph is not None:
    properties['type'] = np.nan

output = Relationships(
    type='CITED_BY',
    id_column_start='id_x',
    id_column_end='id_y',
    data=relation,
    start_node=nodes,
    end_node=nodes,
    allow_unknown_nodes=False, # This may not matter since we write all Pub nodes now from refs
    properties=rela_properties
)

batch_size = 10_000
logger.info(f"Saving {len(output)} citation relationships to Neo4j...")
output.export_to_neo4j(graph, batch_size=batch_size) 


