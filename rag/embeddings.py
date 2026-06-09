from langchain_community.embeddings import FastEmbedEmbeddings

def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")