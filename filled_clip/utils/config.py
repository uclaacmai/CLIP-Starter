import torch

debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42

# training configs
epochs = 10
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5

# dataset configs
image_path = "../data/Images"
captions_path = "../data/"
im_size = 224
max_length = 200
batch_size = 32
num_workers = 2

# projection head configs
projection_dropout = 0.1
projection_dim = 256

# clip configs
temperature = 1.0
image_embedding = 2048
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"