from transformers import BartForConditionalGeneration, BertTokenizer, BartModel
from transformers import TextDataset

from utils import read_txt_data


model_name = r"fnlp/bart-base-chinese"

customize_vocab = r"./customized_vocab.txt"

# 自定义的词表, 加载到 BertTokenizer
tokenizer = BertTokenizer.from_pretrained(customize_vocab)
# tokenizer.save_pretrained(r"./tokenizer_customized_0/")     # 保存自定义的tokenizer

# 加载Bart模型
_model = BartModel.from_pretrained(model_name)

_model.encoder.resize_token_embeddings(len(tokenizer))
_model.decoder.resize_token_embeddings(len(tokenizer))

# _model.save_pretrained(r"./model_customized_0/")        # 保存未经过再次预训练的模型

sample_train_data = read_txt_data(r"./desp_raw_train.txt")[0]

sample_encode = tokenizer.encode(sample_train_data)
