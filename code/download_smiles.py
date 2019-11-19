import boto3
import pandas as pd
import pyarrow as pa
from s3fs import S3FileSystem
import pyarrow.parquet as pq
import os
from datetime import timedelta  
from datetime import datetime
import os

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

import shutil

#access_key = os.environ.get("CINNECTA_S3_A")
#secret_key = os.environ.get("CINNECTA_S3_S")

access_key='AA'
secret_key='aa'


# Listar arquivos dentro do bucket por Data e Hora
# date: str YYYY/MM/DD
# hour: str 00
def list_bucket(bucket, datehour, resource):
    s3_resource = boto3.resource('s3',aws_access_key_id = access_key, aws_secret_access_key = secret_key)
    my_bucket = s3_resource.Bucket(bucket)
    l = []
    for object_summary in my_bucket.objects.filter(Prefix="consolidated_events/customer=Smiles/host_app=Smiles/{}/".format(datehour)):
        l.append(str(object_summary.key))
    return l

#Carregar parquet do S3 em pandas Dataframe
def load_parquet_s3 (file_system, bucket, file):
    s3_path = 's3://{}/{}'.format(bucket, file)
    dataset = pq.ParquetDataset(s3_path, filesystem=s3)
    df = dataset.read_pandas().to_pandas()
    return df

def download (file):
    df_tmp = load_parquet_s3(s3, bucket, file)
    #df_tmp = df_tmp[df_tmp['app_package'].notnull()]
    #df_tmp = df_tmp[~df_tmp['app_package'].str.contains('://')]
    #df_tmp = df_tmp[df_tmp['host_app_package'].str.contains('prime')]

    df_tmp['start_date']=df_tmp['start_date'].astype('datetime64[s]')

    df_tmp.to_parquet('data/Parquets_Smiles/{}_{}.parquet'.format(file[52:62], file[63:73]), engine='pyarrow')


#col = ['host_app_package','host_app_version','installation_number','client_version','config_version','device_id','handset','operating_system','imsi','operator','advertising_id','os_version','api_version','nfc_adapter','nfc_enabled','bluetooth_enabled','data_mobile_enabled','roaming_enabled','memory_usage_percent','cpu_usage_percent','battery_charge_percent','start_date','duration','event_type','network_operator','network_technology','lac','cell_id','latitude','longitude','location_accuracy','signal_strength','network_type','app_name','app_package','app_traffic_rx_total','app_throughput_rx_min','app_throughput_rx_q25','app_throughput_rx_q50','app_throughput_rx_q75','app_throughput_rx_max','app_traffic_tx_total','app_throughput_tx_min','app_throughput_tx_q25','app_throughput_tx_q50','app_throughput_tx_q75','app_throughput_tx_max','external_identifier','local_time_zone','location_provider','other_party_msisdn','send_time','sms_message','sms_phone_number','sms_time']

#Set Parameters
bucket = 'vir-datalake'
s3_resource = boto3.resource('s3',aws_access_key_id = access_key, aws_secret_access_key = secret_key)
s3 = S3FileSystem(key = access_key, secret = secret_key)

initial_date = '2019-06-04'
dias = 20

l = []

for i in range(dias):
    date = (datetime.strptime(initial_date, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
    #print (date)
    l = l+(list_bucket(bucket, date, s3_resource))
    #print(l)

l2 = [k for k in l if '.parquet' in k]
    
total = len(l2)
print(total)

num_cores = multiprocessing.cpu_count()
try: 
    #shutil.rmtree('Parquets_2/')
    os.makedirs('data/Parquets_Smiles/')
except Exception as e:
    print(e)

Parallel(n_jobs=num_cores)(delayed(download)(i) for i in tqdm(l2))
