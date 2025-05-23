from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
import openai
import pandas as pd
import json
from openai import AzureOpenAI
import os 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
import requests
import json
import pandas as pd
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient
import pandas as pd
from azure.search.documents.indexes.models import (
    SemanticSearch,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    SearchIndex,
)


#read .env file
from dotenv import load_dotenv  
load_dotenv()


azure_oai_client = AzureOpenAI(
  api_key = os.environ["AZURE_OPENAI_KEY"],  
  api_version = os.environ["AZURE_OPENAI_API_VERSION"],
  azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
)
title = "temp_title"
#write a function to loop through the list of files in the directory and create embeddings for each file

text_index_name = "first_pdf_index"


def create_index_definition(index_name: str, model: str) -> SearchIndex:
    dimensions = 1536  # text-embedding-ada-002
    if model == "text-embedding-3-large":
        dimensions = 3072

    # The fields we want to index. The "embedding" field is a vector field that will
    # be used for vector search.
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="chunk_content", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchField(
            name="chunk_content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            # Size of the vector created by the text-embedding-ada-002 model.
            vector_search_dimensions=dimensions,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    # The "content" field should be prioritized for semantic ranking.
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            keywords_fields=[],
            content_fields=[SemanticField(field_name="chunk_content")],
        ),
    )

    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=1000,
                    ef_search=1000,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            ),
        ],
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index definition
    return SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=semantic_search,
        vector_search=vector_search,
    )

def create_index():
    #check if index already exists
    search_client = SearchIndexClient(
        endpoint=os.environ['SEARCH_ENDPOINT'],
        credential=AzureKeyCredential(os.environ["SEARCH_KEY"])
    )
    index_client = SearchIndexClient(endpoint=os.environ['SEARCH_ENDPOINT'], credential=AzureKeyCredential(key=os.environ["SEARCH_KEY"]))
    total_records = 0
    try:
        search_client.get_index(text_index_name)
        print(f"Index {text_index_name} already exists.")
    #get the number of documents in the index
        index = search_client.get_index(text_index_name)
        # print(f"Index {text_index_name} has {search_client.get_document_count()} documents.")

        # total_records = search_client.get_document_count()
        total_records = 1000

    except Exception as e:
        print(f"Index {text_index_name} does not exist. Creating a new index.")
        # Create the index
        # Create the index definition

        index_definition = create_index_definition(text_index_name, model="")
        index_client.create_index(index_definition)

    return total_records


def upload_text(text,title):
    total_records = create_index()
    text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
    chunks = text_splitter.split_text(text)
    df = pd.DataFrame(chunks, columns=["chunk_content"])

    df['chunk_content_vector'] = df['chunk_content'].apply(lambda x : azure_oai_client.embeddings.create(input = [x], model="text-embedding-ada-002").data[0].embedding) 
    df['id'] = df.index + total_records + 1
    df['title']  = title
    df = df[['id', 'chunk_content', 'chunk_content_vector']]

    df.to_json(title+".json", orient="records")
    text_df = pd.read_json(title+".json") 

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
        search_client = SearchClient(os.environ['SEARCH_ENDPOINT'], text_index_name, AzureKeyCredential(os.environ["SEARCH_KEY"]))
        result = search_client.upload_documents(documents=records)
        records = []

            


for filename in os.listdir("./data"):
        if filename.endswith(".pdf"):
            title = filename.split('/')[-1].split('.')[0]
            print(title)
            print(filename)
            pdfFile = "./data/"+filename
            pdf_reader = PdfReader(pdfFile)

            for page in pdf_reader.pages:

                
                text = (page.extract_text())


            # Close the PDF file object
                # print(f"Extracted text from {filename}", text )


                upload_text(text,title)
                print(f"Uploaded {filename} to the index.")
        else:
            with open("./data/"+filename, 'r', encoding='utf-8') as file:
                
                title = filename.split('/')[-1].split('.')[0]
                print(title)
                text = file.read()
                upload_text(text,title)
                print(f"Uploaded {filename} to the index.")
        