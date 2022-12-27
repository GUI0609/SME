import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import stellargraph as sg
from stellargraph.connector.neo4j import Neo4jStellarGraph
from stellargraph.layer import GCN
from stellargraph.mapper import ClusterNodeGenerator
import tensorflow as tf
import py2neo
import os
from sklearn import preprocessing, feature_extraction, model_selection
import numpy as np
import pandas as pd
import scipy.sparse as sps
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter

from py2neo import Graph, Node, Relationship, NodeMatcher
import networkx as nx
import random

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from stellargraph import globalvar
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator
import matplotlib.pyplot as plt
# graph = Graph('http://49.234.22.192:7474',auth = ('neo4j','Ggl0609,'))
# neo4j_sg = Neo4jStellarGraph(graph)
default_host = os.environ.get("STELLARGRAPH_NEO4J_HOST")
# Create the Neo4j Graph database object;
# the arguments can be edited to specify location and authentication
neo4j_graph  = py2neo.Graph(host='49.234.22.192', port=7687, user='neo4j', password="Ggl0609,")

def get_embedding(year):
    print(year)
    cypher_query = f"""
      MATCH (n1)-[r:CITED_BY]->(n2)
            WHERE n2.id STARTS WITH 'SME' AND n2.publicationDate.year = {year}
            RETURN n1.id AS id,n1.title as title,n1.abstract as abstract,n1.embedding AS embedding
            UNION
            MATCH (n:Publication)
            WHERE n.id STARTS WITH 'SME' AND n.publicationDate.year = {year}
            RETURN n.id AS id,n.title as title,n.abstract as abstract,n.embedding AS embedding
    """
    homogeneous_nodes2 = neo4j_graph.run(
        cypher_query
    ).to_data_frame()
    homogeneous_nodes2 = homogeneous_nodes2.set_index("id")
    homogeneous_nodes2 = homogeneous_nodes2[['embedding']]
    df2 = pd.DataFrame(homogeneous_nodes2['embedding'].values.tolist())
    df2.index = homogeneous_nodes2.index
    del homogeneous_nodes2#释放内存
    
    query = f"""
            MATCH (n1)-[r:CITED_BY]->(n2)
            WHERE n2.id STARTS WITH 'SME' AND n2.publicationDate.year = {year}
            RETURN n1.id AS source, n2.id AS target
            """
    edges = neo4j_graph.run(
        query
    ).to_data_frame()
    G = StellarGraph(df2, edges)
    # print(G.info())
    nodes = list(G.nodes())
    node_subjects  = df2.index
    number_of_walks = 1
    length = 5
    unsupervised_samples = UnsupervisedSampler(
        G, nodes=nodes, length=length, number_of_walks=number_of_walks
    )
    batch_size = 50
    epochs = 5
    num_samples = [10, 5]
    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [50, 50]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
    )
    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=0,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )
    
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_ids = node_subjects
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
    return node_embeddings
    
    
year=2021
node_embeddings = get_embedding(year)
np.save(f'embedding_{year}',node_embeddings)
