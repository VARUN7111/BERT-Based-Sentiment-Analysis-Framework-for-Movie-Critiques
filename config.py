import transformers

DEVICE = "cuda"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "bert_base_uncased"
MODEL_PATH = "bert_model.bin"
TRAINING_FILE = "IMDB_Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
