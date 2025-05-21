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


from azure.search.documents.indexes import SearchIndexClient

index_client = SearchIndexClient(
    endpoint="", credential=AzureKeyCredential(key=os.environ["SEARCH_KEY"])
)

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


def create_index_definition(index_name: str, model: str) -> SearchIndex:
    dimensions = 1536  # text-embedding-ada-002
    if model == "text-embedding-3-large":
        dimensions = 3072

    # The fields we want to index. The "embedding" field is a vector field that will
    # be used for vector search.
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="chunk_content", type=SearchFieldDataType.String),
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
            content_fields=[SemanticField(field_name="content")],
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

index_definition = create_index_definition("first_pdf_index", model="")
index_client.create_index(index_definition)


if index % batch_size == 0 or (index+1 == total_records):
    search_client = SearchClient(ais_endpoint, text_index_name, AzureKeyCredential(ais_key))
    result = search_client.upload_documents(documents=records)
    records = []