from vespid.models.batch_cluster_tuning import get_embeddings


embeddings = get_embeddings(
            db_ip='49.234.22.192',
            year=2008,
            db_password_secret=None,to_list=True
        )
print(embeddings.columns)