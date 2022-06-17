import os
import pickle
import random
import time
from typing import Dict, List, Optional
import time
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

# This python file includes some of the modified versions of  SSP
# The original SSP repository: https://github.com/kaansonmezoz/bert-same-sentence-prediction
# Line 27:  Hard Negative Pairs modification
# Line 225: HNP + M1 modification
# Line 422: HNP + M2 Modification

# Picking negative pairs from the same document.
# This process generated negative pairs which are hard to predict as negative.
class TextDatasetForSSPHardNegativePairs(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            ssp_probability=0.5,
            load_small=False,
            max_length=512
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.ssp_probability = ssp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_ssp_hnp_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer
        self.max_length_without_special_tokens = max_length - self.tokenizer.num_special_tokens_to_add(pair=True)

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    if load_small:
                        lines = lines[:250]
                    for line in tqdm(lines):
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int, block_size: int):
        """Creates examples for a single document."""

        print("Started creating examples for document => {}/{}".format(doc_index + 1, len(self.documents)))
        doc_start_time = time.time()

        i = 0
        sentences_in_document = len(document)

        ## Every sentence in document will be splitted into two segments.
        ## Every sentence will be tokenized seperately from the other ones.
        while i < sentences_in_document:
            sentence = document[i]

            seq_a, seq_b, split_index = self.split_sentence(sentence)
            if len(seq_a) < 5:
                is_from_same_sentence = True
                self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))

                i += 1
                continue

            if random.random() < self.ssp_probability:
                ## Pick a random segment from a sentence in a different document.
                is_from_same_sentence = False
                random_segment = self.pick_random_segment(doc_index, split_index, i) # sentence index
                seq_a = random_segment
            else:
                ## Two segments should be from the same sentence
                ## So there is no need to replace one of them.
                is_from_same_sentence = True

            assert len(seq_a) >= 1
            assert len(seq_b) >= 1

            self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))
            i += 1

        doc_finish_time = time.time()
        print("Finished creating examples for document. Time: {} seconds".format(doc_finish_time - doc_start_time))

    def split_sentence(self, sentence: List[int]):
        split_index = len(sentence) // 2
        return sentence[: split_index], sentence[split_index:], split_index

    def pick_random_segment(self, doc_index, token_count, sentence_index):
        for _ in range(10):
            random_sentence = self.pick_random_sentence(doc_index, sentence_index)

            if len(random_sentence) >= token_count:
                return random_sentence[: token_count]

        return random_sentence

    def pick_random_sentence(self, doc_index, sentence_index):
        random_document = self.documents[doc_index] # using the same document
        random_sentence_index = self.pick_random_index(len(random_document), sentence_index)
        return random_document[random_sentence_index]

    def pick_random_document(self, doc_index) -> List[int]:
        random_index = self.pick_random_index(len(self.documents), doc_index)
        return self.documents[random_index]

    def pick_random_index(self, len_array, current_index=None):
        for _ in range(10):
            random_index = random.randint(0, len_array - 1)
            if random_index != current_index:
                return random_index

        return random_index
    
    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
            trunc_tokens = tokens_a if len(
                tokens_a) > len(tokens_b) else tokens_b
            if not (len(trunc_tokens) >= 1):
                raise ValueError(
                    "Sequence length to be truncated must be no less than one")
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def create_example(self, seq_a, seq_b, is_from_same_sentence):
        self.truncate_seq_pair(seq_a, seq_b, self.max_length_without_special_tokens)
        # add special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(seq_a, seq_b)

        # add token type ids, 0 for segment a, 1 for segment b
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(seq_a, seq_b)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "same_sentence_label": torch.tensor(1 if is_from_same_sentence else 0, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


### HNP with M1
# Picking negative pairs from the same document.
# This process generated negative pairs which are hard to predict as negative.
# Also using a short sequence to create shorter sequences.
# for hnp changed pick_random_sentence method
# for creating shorter sequences added three lines of code to the create_examples_from_document method.
class TextDatasetForSSPWithHNPAndShortSeq(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            ssp_probability=0.5,
            short_seq_prob=0.1,
            max_len=512,
            load_small=False
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.ssp_probability = ssp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_ssp_hnp_v1_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer
        self.max_length_without_special_tokens = max_len - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_count = 0

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    if load_small:
                        lines = lines[:250]
                    for line in tqdm(lines):
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(tqdm(self.documents)):
                    self.create_examples_from_document(document, doc_index, short_seq_prob)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int, short_seq_prob=0.1):
        """Creates examples for a single document."""
        i = 0
        sentences_in_document = len(document)

        ## Every sentence in document will be splitted into two segments.
        ## Every sentence will be tokenized seperately from the other ones.
        while i < sentences_in_document:

            sentence = document[i]
            if random.random() < short_seq_prob:
                target_seq_length = random.randint(2, self.max_length_without_special_tokens)
                sentence = sentence[:target_seq_length]
                self.short_seq_count += 1

            seq_a, seq_b, split_index = self.split_sentence(sentence)

            if len(seq_a) < 5:
                is_from_same_sentence = True
                self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))

                i += 1
                continue

            if random.random() < self.ssp_probability:
                ## Pick a random segment from a sentence in a different document.
                is_from_same_sentence = False
                random_segment = self.pick_random_segment(doc_index, split_index, i) # sentence index
                seq_a = random_segment
            else:
                ## Two segments should be from the same sentence
                ## So there is no need to replace one of them.
                is_from_same_sentence = True

            assert len(seq_a) >= 1
            assert len(seq_b) >= 1

            self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))
            i += 1

    def split_sentence(self, sentence: List[int]):
        split_index = len(sentence) // 2
        return sentence[: split_index], sentence[split_index:], split_index

    def pick_random_segment(self, doc_index, token_count, sentence_index):
        for _ in range(10):
            random_sentence = self.pick_random_sentence(doc_index, sentence_index)

            if len(random_sentence) >= token_count:
                return random_sentence[: token_count]

        return random_sentence

    def pick_random_sentence(self, doc_index, sentence_index):
        random_document = self.documents[doc_index] # using the same document
        random_sentence_index = self.pick_random_index(len(random_document), sentence_index)
        return random_document[random_sentence_index]

    def pick_random_document(self, doc_index) -> List[int]:
        random_index = self.pick_random_index(len(self.documents), doc_index)
        return self.documents[random_index]

    def pick_random_index(self, len_array, current_index=None):
        for _ in range(10):
            random_index = random.randint(0, len_array - 1)
            if random_index != current_index:
                return random_index

        return random_index
    
    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
            trunc_tokens = tokens_a if len(
                tokens_a) > len(tokens_b) else tokens_b
            if not (len(trunc_tokens) >= 1):
                raise ValueError(
                    "Sequence length to be truncated must be no less than one")
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def create_example(self, seq_a, seq_b, is_from_same_sentence):
        self.truncate_seq_pair(seq_a, seq_b, self.max_length_without_special_tokens)
        # add special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(seq_a, seq_b)

        # add token type ids, 0 for segment a, 1 for segment b
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(seq_a, seq_b)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "same_sentence_label": torch.tensor(1 if is_from_same_sentence else 0, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


# Picking negative pairs from the same document.
# This process generated negative pairs which are hard to predict as negative.
# to add HNP feature changed pick_random_sentence method
# to add M2 feature changed split_sentence method
class TextDatasetForSSPWithHNPAndRandomSplitPoint(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            ssp_probability=0.5,
            load_small=False,
            max_len=512
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.ssp_probability = ssp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_ssp_hnp_v2_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer
        self.out_of_bound = 0
        self.max_length_without_special_tokens = max_len - tokenizer.num_special_tokens_to_add(pair=True)

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    if load_small:
                        lines = lines[:250]
                    for line in tqdm(lines):
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(tqdm(self.documents)):
                    self.create_examples_from_document(document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int, block_size: int):
        """Creates examples for a single document."""

        i = 0
        sentences_in_document = len(document)

        ## Every sentence in document will be splitted into two segments.
        ## Every sentence will be tokenized seperately from the other ones.
        while i < sentences_in_document:

            sentence = document[i]

            seq_a, seq_b, split_index = self.split_sentence(sentence)
            if len(seq_a) < 5:
                is_from_same_sentence = True
                self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))
                i += 1
                continue

            if random.random() < self.ssp_probability:
                ## Pick a random segment from a sentence in a different document.
                is_from_same_sentence = False
                random_segment = self.pick_random_segment(doc_index, split_index, i) # sentence index
                seq_a = random_segment
            else:
                ## Two segments should be from the same sentence
                ## So there is no need to replace one of them.
                is_from_same_sentence = True

            assert len(seq_a) >= 1
            assert len(seq_b) >= 1

            self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))
            i += 1


    def split_sentence(self, sentence: List[int], split_strategy='gaussian'):
        min = 1
        max = len(sentence) - 2
        if split_strategy == 'gaussian':
            mean = len(sentence) / 2
            std = mean / 3.2
            split_index = random.gauss(mean, std)
            if split_index <= min or split_index >= max :
                self.out_of_bound += 1
                split_index = mean # if its beyond limits we set it to mean
            split_index = round(split_index)
        else:
            split_index = random.randint(min, max)
        return sentence[: split_index], sentence[split_index:], split_index

    def pick_random_segment(self, doc_index, token_count, sentence_index):
        for _ in range(10):
            random_sentence = self.pick_random_sentence(doc_index, sentence_index)

            if len(random_sentence) >= token_count:
                return random_sentence[: token_count]

        return random_sentence

    def pick_random_sentence(self, doc_index, sentence_index):
        random_document = self.documents[doc_index] # using the same document
        random_sentence_index = self.pick_random_index(len(random_document), sentence_index)
        return random_document[random_sentence_index]

    def pick_random_document(self, doc_index) -> List[int]:
        random_index = self.pick_random_index(len(self.documents), doc_index)
        return self.documents[random_index]

    def pick_random_index(self, len_array, current_index=None):
        for _ in range(10):
            random_index = random.randint(0, len_array - 1)
            if random_index != current_index:
                return random_index

        return random_index
    
    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
            trunc_tokens = tokens_a if len(
                tokens_a) > len(tokens_b) else tokens_b
            if not (len(trunc_tokens) >= 1):
                raise ValueError(
                    "Sequence length to be truncated must be no less than one")
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def create_example(self, seq_a, seq_b, is_from_same_sentence):
        self.truncate_seq_pair(seq_a, seq_b, self.max_length_without_special_tokens)
        # add special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(seq_a, seq_b)

        # add token type ids, 0 for segment a, 1 for segment b
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(seq_a, seq_b)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "same_sentence_label": torch.tensor(1 if is_from_same_sentence else 0, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]