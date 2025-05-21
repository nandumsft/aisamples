from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
import openai
import pandas as pd
import json
from openai import AzureOpenAI

azure_oai_client = AzureOpenAI(
  api_key = aoai_key,  
  api_version = aoai_api_version,
  azure_endpoint = aoai_endpoint
)

pdf_reader = PdfReader('../data/docs/employee_handbook.pdf')
pages = [page.extract_text() for page in pdf_reader.pages]
text = " ".join(pages)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)
df = pd.DataFrame(chunks, columns=["chunk_content"])

df['chunk_content_vector'] = df['chunk_content'].apply(lambda x : azure_oai_client.embeddings.create(input = [x], model=aoai_embedding_deployed_model).data[0].embedding) 
df['id'] = df.index
df = df[['id', 'chunk_content', 'chunk_content_vector']]

df.to_json('../data/docs/employee_handbook_embeddings.json', orient="records")

import requests
import json
import pandas as pd
from azure.search.documents import SearchClient  

text_df = pd.read_json('../../data/text/product_docs_embeddings.json') 

batch_size = 10
total_records = text_df.shape[0]
fields = text_df.columns.to_numpy()
text_df['id'] = text_df['id'].astype(str)

records = []

for index, row in text_df.iterrows():
    record = {}
    for field in fields:
            record[field] = row[field]

    records.append(
        record
    )

    if index % batch_size == 0 or (index+1 == total_records):
        search_client = SearchClient(ais_endpoint, text_index_name, AzureKeyCredential(ais_key))
        result = search_client.upload_documents(documents=records)
        records = []