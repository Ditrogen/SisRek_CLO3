import torch
from recbole.model.sequential_recommender.gru4rec import GRU4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
import os

# Pastikan jalur model yang disimpan sudah benar relatif terhadap lokasi skrip ini akan dijalankan.
# Anda mungkin perlu menyesuaikan jalur ini tergantung pada struktur proyek Anda.
# Misalnya, jika 'saved' ada di direktori induk, gunakan '../saved/...'
MODEL_PATH = 'saved/GRU4Rec-Jun-04-2025_12-44-33.pth' # [cite: 1]

# Inisialisasi model dan dataset secara global atau teruskan
# Menginisialisasi sekali lebih efisien untuk GUI yang membuat beberapa prediksi
_model = None
_dataset = None
_config = None

def initialize_recommender():
    """Menginisialisasi model dan dataset RecBole."""
    global _model, _dataset, _config
    if _model is None or _dataset is None:
        _config = Config(model='GRU4Rec', dataset='ml-100k', config_file_list=['test.yaml']) # [cite: 1]
        _dataset = create_dataset(_config) # [cite: 1]
        # Kita tidak memerlukan data train/valid/test untuk prediksi, hanya objek dataset
        # data_preparation(_config, _dataset) # Ini tidak terlalu diperlukan untuk prediksi
        _model = GRU4Rec(_config, _dataset).to(_config['device']) # [cite: 1]
        checkpoint = torch.load(MODEL_PATH, weights_only=False) # [cite: 1]
        _model.load_state_dict(checkpoint['state_dict']) # [cite: 1]
        _model.eval() # [cite: 1]
        print("Model rekomendasi diinisialisasi.")
    return _model, _dataset, _config

def get_recommendations(user_item_sequences, top_k=10):
    """
    Menghasilkan rekomendasi top-k untuk urutan item pengguna yang diberikan.

    Args:
        user_item_sequences (list of list of int): Daftar di mana setiap daftar bagian dalam
                                                   mewakili item yang telah dilihat pengguna.
                                                   misalnya, [[1, 2, 3], [4, 5]]
        top_k (int): Jumlah rekomendasi teratas yang akan dikembalikan.

    Returns:
        list of list of str: Daftar di mana setiap daftar bagian dalam berisi
                             ID item yang direkomendasikan (sebagai string) untuk
                             pengguna yang sesuai.
    """
    model, dataset, config = initialize_recommender()

    # Tentukan panjang urutan maksimum untuk padding
    max_len = max(len(seq) for seq in user_item_sequences)
    # Pad urutan dengan 0s
    padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in user_item_sequences]
    item_lengths = [len(seq) for seq in user_item_sequences]

    input_inter = Interaction({
        'user_id': torch.arange(len(user_item_sequences)), # Dummy user_ids untuk batching
        'item_id_list': torch.tensor(padded_sequences, dtype=torch.long), # [cite: 1]
        'item_length': torch.tensor(item_lengths, dtype=torch.long), # [cite: 1]
    })
    with torch.no_grad(): # [cite: 1]
        scores = model.full_sort_predict(input_inter) # [cite: 1]

    top_k_indices = torch.topk(scores, k=top_k, dim=1).indices # [cite: 1]
    top_k_indices = top_k_indices.cpu().numpy() # [cite: 1]

    item_field = 'item_id' # [cite: 1]
    all_recommendations = []
    for rec_indices in top_k_indices:
        rec_item_ids = [dataset.id2token(item_field, int(i)) for i in rec_indices] # [cite: 1]
        all_recommendations.append(rec_item_ids)

    return all_recommendations

if __name__ == '__main__':
    # Contoh penggunaan untuk menguji modul ini secara independen
    initialize_recommender()
    user_sequences = [[1, 2, 3], [4, 5]]
    recommendations = get_recommendations(user_sequences, top_k=5)
    for i, recs in enumerate(recommendations):
        print(f"Rekomendasi untuk pengguna {i+1}: {recs}")