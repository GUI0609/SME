# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher
from vespid.data.neo4j_tools import (
    Neo4jConnectionHandler, 
    test_graph_connectivity
)

graph = Neo4jConnectionHandler(
    db_ip='49.234.22.192',
    database='neo4j',
    db_username='neo4j',
    db_password='Ggl0609,',
    secure_connection=False)



