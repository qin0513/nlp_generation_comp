import pandas as pd

from utils import Vocab


def convert_to_list(input_str_list):
    """
    将数字字符串转换成list
    :param input_str_list:
    :return:
    """
    converted_list = []     # 转化成数字后的列表
    for input_str in input_str_list:
        converted_list_i = input_str.split()
        converted_list.append(converted_list_i)
    return converted_list


def wrt_vocab(vocab_instance, wrt_name):
    with open(wrt_name, "w", encoding="utf-8") as f_write:
        for _token, _token_id in vocab_instance.token_to_idx.items():
            # _token_id = _token + " " + str(_token_id) + "\n"
            f_write.write(_token + "\n")


def wrt_train_data_encoder(input_data, wrt_name):
    with open(wrt_name, "w", encoding="utf-8") as f_write:
        max_len = 120       # 超过长度120的需要裁剪
        for _input_sent_i in input_data:
            len_i = len(_input_sent_i.split())
            if len_i > max_len:     # 这里做了一个裁剪
                _input_sent_i = " ".join(_input_sent_i.split()[:max_len])
            f_write.write(_input_sent_i + "\n")


if __name__ == '__main__':
    # 读取原数据, 其中 desp_raw 为输入影像描述, diag_raw 为输出诊断报告
    train_csv = pd.read_csv(r"./data_train/train.csv", header=None)

    desp_raw = train_csv.iloc[:, 1].tolist()
    diag_raw = train_csv.iloc[:, 2].tolist()
    desp_num = convert_to_list(desp_raw)
    diag_num = convert_to_list(diag_raw)

    # 用于计算总共的token
    token_total = desp_raw + diag_raw
    vocab_train = Vocab(token_total)

    # 写入vocab数据
    # wrt_vocab(vocab_train, r"./customized_vocab.txt")

    # 将训练数据 encoder 写入text
    # wrt_train_data_encoder(desp_raw, r"./desp_raw_train.txt")
