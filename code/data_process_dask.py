from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import os
import fastparquet
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import pandas as pd
from numba import jit, prange
import json
from pathlib import Path

print("All imported")

client = LocalCluster()

print("Dask setted up")

# CONSTANTES A SEREM MODIFICADAS
data_path = Path("data")
output_path = Path("dask_preprocessed")

parquets_path = data_path / "Parquets_Smiles"
train_date = "2019-05-16"
id_name = "external_identifier"
app_name = "app_package"

another_base_external_identifiers = np.sort(np.unique(pd.read_csv(data_path / "df_user_censo.csv")['Unnamed: 0'].to_numpy())).tolist()

for i in range(len(another_base_external_identifiers)):
    another_base_external_identifiers[i] = str(another_base_external_identifiers[i])

# CRIANDO LISTA DE APLICATIVOS A SEREM ANALISADOS

apps_list = [x for x in json.loads(open("aplications_to_keep.json", "r").read())['apps'] if "://" not in x]
apps_count = len(apps_list)

train_dataframe = []
test_dataframe = []

# ABRE TODOS OS PARQUETS

print("opening parquets")

for file in parquets_path.glob('*.parquet'):
    
    swap_df = dd.read_parquet(file, engine='fastparquet', columns=['external_identifier', 'app_package'])
    
    ## Escolhe apenas usuários da outra base de dados
    swap_df = swap_df[swap_df[id_name].isin(another_base_external_identifiers)]
    
    ## Escolhe apenas aplicativos de interesse
    swap_df = swap_df[swap_df[app_name].isin(apps_list)]
    
    if os.path.basename(file).split("_")[0] == train_date:
        test_dataframe.append(swap_df)
    else:
        train_dataframe.append(swap_df)

swap_df = None     

print("Concatening parquets")

# CONCATENA TODOS OS DATAFRAMES DE TREINO E TESTE

test_dataframe = dd.concat(test_dataframe).drop_duplicates().dropna()
train_dataframe = dd.concat(train_dataframe).drop_duplicates().dropna()

# FAZ INTERSEÇAO ENTRE TREINO E TESTE, UTILIZANDO O EXTERNAL ID E TRANSFORMA APPS EM CATEGORIAS

train_dataframe = train_dataframe[train_dataframe[id_name].isin(test_dataframe[id_name].compute())].categorize(columns=[app_name]).reset_index(drop=True)
test_dataframe = test_dataframe[test_dataframe[id_name].isin(train_dataframe[id_name].compute())].categorize(columns=[app_name]).reset_index(drop=True)

# SALVANDO ARQUIVOS DE TREINO E TESTE COMO PARQUET POIS SAO ARQUIVOS GRANDES

try:
    os.mkdir(output_path)
except:
    print("Pasta já existe!")

print("Saving train e test parquets")

train_dataframe.compute().to_parquet(output_path / "train.parquet")
test_dataframe.compute().to_parquet(output_path / "test.parquet")

print("Saving pivot tables")

train_pivot_table = train_dataframe.compute().pivot_table(index=[id_name], columns=[app_name], aggfunc=[len], fill_value=0)
test_pivot_table = test_dataframe.compute().pivot_table(index=[id_name], columns=[app_name], aggfunc=[len], fill_value=0)

# SALVANDO TABELAS PIVO

try:
    os.mkdir(output_path)
except:
    print("Pasta já existe!")
    
train_pivot_table.to_csv(output_path / "train_pivot_table.csv")
test_pivot_table.to_csv(output_path / "test_pivot_table.csv")


