from sentence_transformers import SentenceTransformer
from scipy.spatial import distance


def get_similar_command(text, data):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    test_vec = model.encode([text])[0]
    similarity_arr = []
    for sent in data:
        similarity_score = 1 - distance.cosine(
            test_vec, model.encode([sent["name"]])[0]
        )
        if similarity_score > 0.1:
            similarity_arr.append(
                {"score": similarity_score, "text": sent["name"], "id": sent["id"]}
            )
    similarity_arr = sorted(similarity_arr, key=lambda x: x["score"], reverse=True)
    return similarity_arr
