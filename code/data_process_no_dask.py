from glob import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path

# CONSTANTES A SEREM MODIFICADAS
data_path = Path("data")
output_path = Path("preprocessed")

parquets_path = data_path / "Parquets_Smiles"
train_date = "2019-06-04"
id_name = "external_identifier"
app_name = "app_package"

# PERCORRE TODOS OS ARQUIVOS UMA VEZ PRA DETECTAR USUÁRIOS CANDIDATOS AO FILTRO,
# OU SEJA, USUÁRIOS COM REGISTRO NO DIA DE TREINO E NOS DIAS DE TESTE

def np_remove_non_numeric(a):    
    for i in range(len(a)):
        if not str(a[i]).isnumeric():
            a[i] = 0
        elif float(a[i]) % 1 != 0:
            a[i] = 0

    a = np.unique(a.astype(np.int64))
    
    return a

test_external_identifiers = []
train_external_identifiers = []

for file in parquets_path.glob('*.parquet'):
    if os.path.basename(file).split("_")[0] == train_date:
        train_external_identifiers.append(np_remove_non_numeric(pd.read_parquet(file, columns = [id_name])[id_name].to_numpy()))
    else:
        test_external_identifiers.append(np_remove_non_numeric(pd.read_parquet(file, columns = [id_name])[id_name].to_numpy()))
        
test_external_identifiers = np.sort(np.unique(np.concatenate(test_external_identifiers))).tolist()
train_external_identifiers = np.sort(np.unique(np.concatenate(train_external_identifiers))).tolist()

another_base_external_identifiers = np.sort(np.unique(pd.read_csv(data_path / "df_user_censo.csv")['Unnamed: 0'].to_numpy())).tolist()

# FAZENDO INTERSEÇÃO ENTRE USUARIOS DO DIA DE TREINO E DOS DIAS DE TESTE

def intersection(lst1, lst2, lst3): 
    lst4 = [value for value in lst1 if (value in lst2 and value in lst3)] 
    return lst4

final_external_identifiers = intersection(test_external_identifiers, train_external_identifiers, another_base_external_identifiers)
users_count = len(final_external_identifiers)

# CRIANDO LISTA DE APLICATIVOS A SEREM ANALISADOS

import json
apps_list = [x for x in json.loads(open("aplications_to_keep.json", "r").read())['apps'] if "://" not in x]
apps_count = len(apps_list)

# CRIANDO TABELAS PIVO COM ZEROS

train_pivot_table = np.zeros((users_count, apps_count))
test_pivot_table = np.zeros((users_count, apps_count))

# PERCORRE UM ARQUIVO E ADICIONA NA TABELA PIVÔ 1 NA POSIÇÃO [USUÁRIO][APLICATIVO]

def numpy_add_pivot_table(a, pivot_table):
    for i in range(len(a)):
        if a[i][1] in apps_list and str(a[i][0]).isnumeric() and int(a[i][0]) in final_external_identifiers:
            pivot_table[final_external_identifiers.index(int(a[i][0]))][apps_list.index(a[i][1])] = 1
            
    return pivot_table

# PERCORRENDO ARQUIVOS ADICIONANDO 1 NOS APLICATIVOS DOS USUÁRIOS

for file in parquets_path.glob('*.parquet'):
    if os.path.basename(file).split("_")[0] == train_date:
        numpy_add_pivot_table(pd.read_parquet(file, columns = [id_name, app_name]).to_numpy(), train_pivot_table)
    else:
        numpy_add_pivot_table(pd.read_parquet(file, columns = [id_name, app_name]).to_numpy(), test_pivot_table)

# SALVANDO TABELAS PIVO
os.mkdir(output_path)

pd.DataFrame(data=train_pivot_table, columns=apps_list, index=final_external_identifiers).astype(int).to_csv(output_path / "train.csv")
pd.DataFrame(data=test_pivot_table, columns=apps_list, index=final_external_identifiers).astype(int).to_csv(output_path / "test.csv")

# test_pivot_table = pd.read_csv(output_path / "test.csv").to_numpy()
# train_pivot_table = pd.read_csv(output_path / "train.csv").to_numpy()

# PRE PROCESSAMENTO DO FILTRO COLABORATIVO

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def numpy_cosine_similarity(data_items):

    data_sparse = sparse.csr_matrix(data_items)
    sim = cosine_similarity(data_sparse.transpose())
    return sim

# PRE PROCESSANDO FILTRO COLABORATIVO

cosine_similarity_table = numpy_cosine_similarity(train_pivot_table).astype(float)

pd.DataFrame(data=cosine_similarity_table, columns=apps_list, index=apps_list).to_csv(output_path / "cosine_similarity.csv")

from numba import jit, prange

# APLICA FILTRO COLABORATIVO

@jit(nopython=True, parallel=True)
def numpy_install_chances(data, similarity, output):

    for i in prange(len(data)):
        output[i] = np.divide(similarity.dot(data[i]), similarity.sum(axis=1) + 0.01)

# APLICANDO FILTRO COLABORATIVO

install_chances_table = np.empty((users_count, apps_count))

numpy_install_chances(train_pivot_table, cosine_similarity_table, install_chances_table)

# SALVANDO TABELA CHANCES DE INSTALAR

pd.DataFrame(data=install_chances_table, columns=apps_list, index=final_external_identifiers).to_csv(output_path / "install_chances.csv")

# APLICA THRESHOLD NO FILTRO COLABORATIVO, TRANSFORMANDO CHANCES DE INSTALAR EM 1 OU 0 (POSSIVELMENTE INSTALADO OU NAO)

@jit(nopython=True, parallel=True)
def numpy_predict_threshold(data, threshold, output):

    for i in prange(len(data)):
        for j in prange(len(data[0])):
            if data[i][j]>= threshold:
                output[i][j] = 1
            else:
                output[i][j] = 0

# APLICANDO THRESHOLD

predict_threshold_table = np.empty((users_count, apps_count))
numpy_predict_threshold(install_chances_table, install_chances_table.mean()*3, predict_threshold_table)

# SALVANDO TABELA PREDICT

pd.DataFrame(data=predict_threshold_table, columns=apps_list, index=final_external_identifiers).astype(int).to_csv(output_path / "predict.csv")
