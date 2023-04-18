from collections import Counter


class Vocab:
    """
    生成对应的词典
    """
    def __init__(self, tokens_input=None, min_freq=0, reserved_tokens=None):
        if tokens_input is None:
            tokens_input = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 词频
        tokens = [_token for _line_str in tokens_input for _token in _line_str.split()]

        tokens_counter = Counter(tokens)
        self.sorted_counter = sorted(tokens_counter.items(), key=lambda token_counter_i: token_counter_i[1], reverse=True)

        # 建立词表索引
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.sorted_counter:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(list(self.idx_to_token.keys()))

    @property
    def unk(self):
        return 0


def read_txt_data(input_dir):
    with open(input_dir, "r", encoding="utf-8") as f_read:
        return f_read.readlines()



