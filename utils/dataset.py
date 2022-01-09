from utils.vocab import tokenizer


class Dataset:
    def __init__(self, corpus: str, window_size: int, task: str):
        self.corpus = corpus
        self.window_size = window_size
        self.task = task

    def __iter__(self):
        assert self.task in ["CBOW","skip-gram"]
        with open(self.corpus, encoding="utf8") as f:
            for line in f:
                tokens = tokenizer(line)
                if len(tokens) <= 1:
                    continue
                for i, target in enumerate(tokens):
                    left_context = tokens[max(0, i - self.window_size): i]
                    right_context = tokens[i + 1: i + 1 + self.window_size]
                    context = left_context + right_context
                    if self.task == "CBOW" :
                        yield context, target
                    elif self.task == "skip-gram":
                        yield target, context