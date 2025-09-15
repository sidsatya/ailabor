import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv  
from sklearn.neighbors import NearestNeighbors
import networkx as nx   # library for creating graphs/networks
import faiss

load_dotenv()  # Load environment variables from .env file
# Ensure the OpenAI API key is set in the environment

if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embeddings(texts, model='text-embedding-ada-002'): 
    # texts: list of strings
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    # Return a list of embeddings
    return [item.embedding for item in response.data]

def create_embeddings(data, batch_size=512, save_path=None):
    embeddings = []
    for i in range (0, len(data), batch_size): 
        batch = data['task_clean'].iloc[i:i+batch_size].tolist()
        print(f"Processing batch from index {i} to {min(i + batch_size, len(data)) - 1}")
        batch_embeddings = get_openai_embeddings(batch)
        embeddings.extend(batch_embeddings)
        print(f"Processed batch {i // batch_size + 1}/{(len(data) + batch_size - 1) // batch_size}")

    # format of return is [[embedding1], [embedding2], ...]
    E = np.vstack(embeddings)

    if save_path:
        np.save(save_path, E)
        print(f"Embeddings saved to {save_path}")
    return E

# read the data from the csv file
directory = '/Users/sidsatya/dev/ailabor/onet_transformations/intermediate_data/'
data_path = os.path.join(directory, 'unique_task_statements.csv')

data = pd.read_csv(data_path, encoding='latin1')

# clean the data by lowering the case and removing extra spaces. The regex `r'\s+'` matches one or more whitespace characters.
data['task_clean'] = data['Task'].str.lower().replace(r'\s+', ' ', regex=True).str.strip()  # remove extra spaces

'''
Step 1:
Get the embedding for each task statement using OpenAI's text-embedding-ada-002 model.
'''
# check if the embeddings already exist
if os.path.exists(os.path.join(directory, 'task_embeddings.npy')):
    print("Embeddings already exist. Loading from file...")
    E = np.load(os.path.join(directory, 'task_embeddings.npy'))
else:
    print("Creating embeddings for task statements...")
    E = create_embeddings(data, batch_size=512, save_path=os.path.join(directory, 'task_embeddings.npy'))

print("Embeddings created successfully. Shape of embeddings:", E.shape)

'''
Step 2: 
Once we have embeddings, we can first normalize each embedding to have an L2 norm of 1 (i.e., make them unit vectors), 
and then we can use FAISS to find the top-k (k=50) nearest neighbors for each embedding based on cosine similarity. 
We want to have the L2 norm so that we can use the inner product between two embedding vectors as a measure of cosine similarity.
This further allows us to perform the index search using FAISS, which is optimized for such operations.
'''
if E.dtype != np.float32 or not E.flags['C_CONTIGUOUS']:
    E = np.ascontiguousarray(E, dtype=np.float32)

d = E.shape[1]  # dimension of the embeddings
faiss.normalize_L2(E)
index = faiss.IndexFlatIP(d)
index.add(E)
D, I = index.search(E, k=50)

'''
Step 3: 
We create a graph where each node is a task statement, and we add an edge between two nodes if their cosine similarity is 
above a certain threshold (0.97 in this case). This allows us to identify clusters of similar task statements.
'''
G = nx.Graph()
G.add_nodes_from(range(len(data)))
threshold = 0.97
for i, neighbors in enumerate(I):
    for j, score in zip(neighbors, D[i]):
        if i < j and score >= threshold:
            G.add_edge(i, j)
clusters = list(nx.connected_components(G))
canon = {}
for cid, comp in enumerate(clusters, 1):
    for idx in comp:
        canon[idx] = f"C{cid:05d}"
        
data['canon_id'] = data.index.map(canon)
data.to_csv(os.path.join(directory, 'task_statements_with_canon_id.csv'), index=False)

# Save a sample of 100 task statements with canon IDs for manual inspection, making sure that the sample is reproducible and 
# only selects rows that aggregated over multiple task statements
grouped = data.groupby('canon_id')['Task'].apply(lambda x: '; '.join(x.unique())).reset_index()
grouped['count'] = grouped['Task'].apply(lambda x: len(x.split('; ')))
sampled = grouped[grouped['count'] > 1].sample(n=100, random_state=42)
sampled.to_csv(os.path.join(directory, 'sample_task_statements_with_canon_id.csv'), index=False)
print("Task statements with canon IDs saved successfully.")

