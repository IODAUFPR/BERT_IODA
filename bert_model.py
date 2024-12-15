from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Carregando o modelo BERT e o tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embedding(text):
    """Obtém o embedding BERT para o texto fornecido."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings


def bert_similarity(s1, s2):
    """Calcula a similaridade entre dois textos usando embeddings do BERT."""
    emb1 = get_bert_embedding(s1)
    emb2 = get_bert_embedding(s2)
    similarity = 1 - cosine(emb1, emb2)
    return similarity * 100
