import collections
import random
import os
import requests
import zipfile
import tarfile
import hashlib

import torch


DATA_HUB = {
    "wikitext-2": (
        "https://s3.amazonaws.com/research.metamind.io/wikitext/" "wikitext-2-v1.zip",
        "3c914d17d80b1459be871a5039ac23e752a53cbe",
    )
}

CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
MASK_TOKEN = "<MASK>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, "wiki.train.tokens")
    with open(file_name, "r") as f:
        lines = f.readlines()
    paragraphs = [
        line.strip().lower().split(" . ")
        for line in lines
        if len(line.split(" . ")) >= 2
    ]
    random.shuffle(paragraphs)
    return paragraphs


def download(name, cache_dir=os.path.join(os.path.expanduser("~"), "data")):
    if name not in DATA_HUB:
        raise KeyError(f"{name} does not exist in {DATA_HUB}")

    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                file_data = f.read(1048576)
                if not file_data:
                    break
                sha1.update(file_data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f"Downloading {fname} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "Only zip/tar files can be extracted."
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def tokenize(lines, token="word"):
    """Split text lines into word or character tokens."""
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("ERROR: unknown token type: " + token)


def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = [CLS_TOKEN] + tokens_a + [SEP_TOKEN]
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + [SEP_TOKEN]
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class Vocab:
    """
    Represents a vocabulary for a text corpus
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = self._count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, [UNK_TOKEN] + reserved_tokens
        uniq_tokens += [
            token
            for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]
        self.idx_to_token, self.token_to_idx = [], {}
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def _count_corpus(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)


class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        """
        paragraphs[i] is a list of sentence strings
        """
        paragraphs = [tokenize(paragraph, token="word") for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(
            sentences,
            min_freq=5,
            reserved_tokens=[PAD_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN],
        )

        examples = []
        # examples[i] is a tuple (List[str], List[int], Boolean),
        # representing a) two tokenized sentences separated by SEP,
        # b) a list of labels for next sentence (e.g., a bunch of zeros then a bunch of ones),
        # c) whether the next sentence is actually the next sentence in the text
        for paragraph in paragraphs:
            examples.extend(
                self._get_nsp_data_from_paragraph(
                    paragraph, paragraphs, max_len
                )
            )

        # examples[i] is now a tuple (List[int], List[int], List[int], List[int], Boolean)
        # representing a) the token ids for the original sentence, with some entries masked
        # b) the position of the masked tokens (i.e., from 0 to len(s) - 1)
        # c) the label (i.e., token id) of the masked tokens
        # d and e as before: labels for the next sentence and whether it's actually the next
        examples = [
            (self._get_mlm_data_from_tokens(tokens) + (segments, is_next))
            for tokens, segments, is_next in examples
        ]

        # Pad inputs
        (
            self.all_token_ids,
            self.all_segments,
            self.valid_lens,
            self.all_pred_positions,
            self.all_mlm_weights,
            self.all_mlm_labels,
            self.nsp_labels,
        ) = self._pad_bert_inputs(examples, max_len)

    def __getitem__(self, idx):
        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx],
            self.all_pred_positions[idx],
            self.all_mlm_weights[idx],
            self.all_mlm_labels[idx],
            self.nsp_labels[idx],
        )

    def __len__(self):
        return len(self.all_token_ids)

    # next sentence prediction private methods
    def _get_next_sentence(self, sentence, next_sentence, paragraphs):
        if random.random() < 0.5:
            is_next = True
        else:
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    def _get_nsp_data_from_paragraph(self, paragraph, paragraphs, max_len):
        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self._get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs
            )
            # two sentences, plus 1 <CLS> and 2 <SEP> tokens
            if len(tokens_a) + len(tokens_b) + 3 > max_len:
                continue
            tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))

        return nsp_data_from_paragraph

    # masked language training methods
    def _replace_mlm_tokens(
        self, tokens, candidate_pred_positions, num_mlm_preds
    ):
        """
        Make a new copy of tokens to be used as input of a masked language model, where input
        contains either <mask> or random tokens

        Returns:
        mlm_input_tokens: the input list of tokens with certain words replaced
        by MASK

        pred_positions_and_labels: a list of tuples (position, label) with the
        replaced tokens. E.g.: [(1, 'it'), (2, 'has'),...]
        """
        mlm_input_tokens = [token for token in tokens]
        pred_positions_and_labels = []

        random.shuffle(candidate_pred_positions)
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions_and_labels) >= num_mlm_preds:
                break
            masked_token = None
            if random.random() < 0.8:
                masked_token = MASK_TOKEN
            else:
                if random.random() < 0.5:  # 10% of the time (0.5 * 0.2)
                    masked_token = tokens[mlm_pred_position]
                else:
                    masked_token = random.randint(0, len(self.vocab) - 1)
            mlm_input_tokens[mlm_pred_position] = masked_token
            pred_positions_and_labels.append(
                (mlm_pred_position, tokens[mlm_pred_position])
            )

        return mlm_input_tokens, pred_positions_and_labels

    def _get_mlm_data_from_tokens(self, tokens):
        """
        Given a set of tokens, strip the special tokens out,
        randomly replace 15% of them with masks and
        return the position and label of the masked tokens.

        Parameters:
        tokens: a list of strings

        Returns:
        vocab[mlm_input_tokens]: A vocab id of the input tokens,
        including the masked ones.

        pred_positions: the position (0-indexed) of the masked tokens
        to be predicted in the input / list of vocab ids returned.

        vocab[mlm_pred_labels]: the vocab id of the masked tokens
        to be predicted.
        """
        candidate_pred_positions = []

        for i, token in enumerate(tokens):
            if token in [CLS_TOKEN, SEP_TOKEN]:
                continue
            candidate_pred_positions.append(i)

        num_mlm_preds = max(1, round(len(tokens) * 0.15))
        mlm_input_tokens, pred_positions_and_labels = self._replace_mlm_tokens(
            tokens, candidate_pred_positions, num_mlm_preds
        )
        pred_positions_and_labels = sorted(
            pred_positions_and_labels, key=lambda x: x[0]
        )
        pred_positions = [v[0] for v in pred_positions_and_labels]
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

        return self.vocab[mlm_input_tokens], pred_positions, self.vocab[mlm_pred_labels]

    def _pad_bert_inputs(self, examples, max_len):
        """
        Outputs lists composed of padded max_len-length tensors
        for pytorch consumption.

        Parameters:
        examples: the 5 dimensional tuple generated in the __init__
        max_len: the length of all the tensors. Sentences larger than max_len
        will be truncated

        Returns:
        Every single return element is a list of torch tensors. The content
        of which tensor is as follows:

        all_token_ids: the masked sentence, by token ids, padded with PAD tokens

        all_segments: segment label vector (i.e., 0s then 1s representing the
        next sentence), padded with zeros. So it looks like a bunch of zeros,
        a bunch of ones, then a bunch of zeros again

        valid_lens: a single number with the actual sentence length (i.e., how many elements before padding)

        all_pred_positions: list of positions for the masked elements to be predicted,
        padded up until max_num_mlm_preds (15% of max_len)

        all_mlm_weights: 1s and 0s representing the "weights" of the previous tensor,
        which is just where it is padded and where it is not

        all_mlm_labels: the actual token_ids of the masked elements to be predicted

        nsp_labels: whether the next sentence is the actual next sentence in the text
        """
        max_num_mlm_preds = round(max_len * 0.15)
        all_token_ids = []
        all_segments = []
        valid_lens = []
        all_pred_positions = []
        all_mlm_weights = []
        all_mlm_labels = []
        nsp_labels = []

        for (
            token_ids,
            pred_positions,
            mlm_pred_label_ids,
            segments,
            is_next,
        ) in examples:
            all_token_ids.append(
                torch.tensor(
                    token_ids + [self.vocab[PAD_TOKEN]] * (max_len - len(token_ids)),
                    dtype=torch.long,
                )
            )
            all_segments.append(
                torch.tensor(
                    segments + [0] * (max_len - len(segments)), dtype=torch.long
                )
            )
            valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
            all_pred_positions.append(
                torch.tensor(
                    pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)),
                    dtype=torch.long,
                )
            )
            all_mlm_weights.append(
                torch.tensor(
                    [1.0] * len(mlm_pred_label_ids)
                    + [0.0] * (max_num_mlm_preds - len(pred_positions)),
                    dtype=torch.float32,
                )
            )
            all_mlm_labels.append(
                torch.tensor(
                    mlm_pred_label_ids
                    + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                    dtype=torch.long,
                )
            )
            nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
        return (
            all_token_ids,
            all_segments,
            valid_lens,
            all_pred_positions,
            all_mlm_weights,
            all_mlm_labels,
            nsp_labels,
        )


def load_data_wiki(batch_size, max_len):
    num_workers = 4
    data_dir = download_extract("wikitext-2", "wikitext-2")
    paragraphs = _read_wiki(data_dir)
    train_set = WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=True, num_workers=num_workers
    )
    return train_iter, train_set.vocab
