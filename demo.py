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

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import optuna
import umap
from tqdm import tqdm
from joblib import dump as dump_obj, load as load_obj
from vespid import setup_logger, get_secure_key
from vespid.data.neo4j_tools import (
    Neo4jConnectionHandler
)
from vespid.models.clustering import HdbscanEstimator
from vespid.models.mlflow_tools import setup_mlflow
from vespid.data.aws import test_s3_access
from vespid.models.optuna_tool import Hyperparameter, Criterion, Objectives
from vespid.models.static_communities import hydrate_best_optuna_solution

logger = setup_logger(module_name=__name__)
from vespid.visualization.__init__ import *

from vespid.models.batch_cluster_tuning import *
import nltk
nltk.download('stopwords')
year = 2007
embeddings = np.load('/public/home/ggl/data/npydata/embedding_2007.npy')
full_experiment_name = f'citations-clu-{year}'
# from vespid.models.batch_cluster_tuning import get_embeddings
from vespid.models.static_communities import *
import warnings
warnings.filterwarnings('ignore')
cluster_methods = ['eom', 'leaf']
# def make_language_clu_save_model_vis(year,colab=True):
colab=False
experiment, client = setup_mlflow(
            full_experiment_name,
            tags={
                'project':'SME',
                'dataset':f'SME-{year}'
            },
            return_client=True
        )

experiment_parameters = Objectives(
  [
      Hyperparameter(
          'experiments_per_trial', 
          constant=5 #5
          ),
      Hyperparameter(
          'umap_n_neighbors', 
          min=10, 
          max=30
          ),
      Hyperparameter(
          'umap_n_components',
          min=5, 
          max=50
          ), 
      Hyperparameter(
          'umap_min_dist', 
          min=0.05, 
          max=0.5
          ),
      Hyperparameter(
          'umap_metric', 
          categories=['euclidean', 'cosine']
          ),
      Hyperparameter(
          'hdbscan_min_cluster_size', 
          min=20, 
          max=100
          ),
      Hyperparameter(
          'hdbscan_min_samples', 
          min=5, 
          max=20
      ),
      Hyperparameter(
          'hdbscan_cluster_selection_method',
          categories=cluster_methods
      )
  ],
  [
      Criterion(
          'DBCV Mean', 
          'maximize'
          ),
      Criterion(
          'DBCV StDev', 
          'minimize'
          ),
      Criterion(
          'NumClusters Mean', 
          'maximize',
          range=(0, np.inf)
          ),
      Criterion(
          'NumClusters StDev', 
          'minimize',
          range=(0, np.inf)
          ),
      Criterion(
          'Mean of Cluster Persistence Means', 
          'maximize'
      )
  ],
  objective,
  embeddings,
  mlflow=(experiment, client)) 

study = bayesian_tune(
          experiment_parameters,
          n_trials=5,#100
          n_jobs=25,
          garbage_collect=False
      )
best_model = hydrate_best_optuna_solution(
            experiment_parameters,
            hydration_tolerance=0.05,
            log_to_mlflow=True
        )
if colab:
  model_path = f'/content/best_model_{year}.pkl'
  dump_obj(best_model, model_path, compress=False)
  # from google.colab import files
  # files.download(model_path)

model_path = f'best_model_{year}.pkl'
dump_obj(best_model, model_path, compress=False)
vis_html_path = f'{year}-html.html'
visualize_language_clusters(X=embeddings,
  cluster_pipeline=best_model, 
  umap_model=None, 
  cluster_model=None,
  html_path = vis_html_path)