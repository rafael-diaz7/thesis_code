from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

MODEL_NAME = 'bert-base-uncased'
EPOCHS = 1
BATCH_SIZE = 32
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.8
MODEL_OUT_FILE_NAME = 'bert_baseline_model.h5'

def create_model():
    bert = TFAutoModel.from_pretrained(MODEL_NAME)

