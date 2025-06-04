import numpy as np
import torch
from recbole.model.sequential_recommender.gru4rec import GRU4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction



config = Config(model='GRU4Rec', dataset='ml-100k', config_file_list=['test.yaml'])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = GRU4Rec(config, dataset).to(config['device'])  
checkpoint = torch.load('saved/GRU4Rec-Jun-04-2025_12-44-33.pth', weights_only=False)  # *.pth file is used for load saved model.
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Here is a example to predict user 1's and user 2's next item.
# user 1 have seen item 1, item 2, item 3
# user 2 have seen item 4, item 5
# 0 is used for padding.
# scores is a [2, n_items] matrix which represent the score of every items.
input_inter = Interaction({
    'user_id': torch.tensor([1, 2]),
    'item_id_list': torch.tensor([[1, 2, 3, 0, 0],
                                    [4, 5, 0, 0, 0]]),
    'item_length': torch.tensor([3, 2]),
})
with torch.no_grad():
    scores = model.full_sort_predict(input_inter)
print("Scores shape\n",scores.shape)
print("Scores: ",scores)

top_k = torch.topk(scores, k=10, dim=1).indices  # shape [2, 10]
top_k = top_k.cpu().numpy()

item_field = 'item_id'  # hardcoded for ML-100k

for user_idx, rec_indices in enumerate(top_k):
    rec_item_ids = [dataset.id2token(item_field, int(i)) for i in rec_indices]
    print(f"Top-10 recommended items for user {user_idx + 1}: {rec_item_ids}")

