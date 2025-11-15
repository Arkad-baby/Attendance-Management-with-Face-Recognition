from SupabaseClient import supabase
import numpy as np

embeddings_all = np.load("Embedding_file2.npy", allow_pickle=True).item()
print(embeddings_all)

for name, embedding in embeddings_all.items():

    embeddingList = []
    for emb in embedding:
        embeddingList.append(emb.tolist())
    data = {"Username": name, "Embeddings": embeddingList}

    response = supabase.table("Users").insert(data).execute()
    print(response)
