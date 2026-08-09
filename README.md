# Transformer-model-using-Pytorch
To make a transfomer language model using pytorch

Dataset for training: https://huggingface.co/datasets/Digilidaahz/chat_dataset_2B_cha/

# What is a transformer?
The Transformer is a model that uses self-attention to process all words in a sentence at the same time, 
so it understands context better and doesn’t need recurrent or convolutional layers.

## Structure of a transformer
**Input:**
* Tokenizer
* Embedding
* Position Embedding

**Transfomer block**
* LayerNorm
* Multihead attention
* +Residual
* LayerNorm
* Feed Forward Network
* +Residual

(repeated N times)
* Final LayerNorm

**Output**
* Linear layer
* Vocabulary size logits
* Softmax
* possibility every token

**Training...**

## 1. Tokenizer
Tokenizer is a tool that converts string characters into token ID's, for example:
a = 0

b = 1

c = 2

d = 3

e = ...

A proper tokenizer doesn't make every character a token. Instead, it uses subword splitting—common words get 1 token, while rare or long words are broken into multiple pieces. On average, 
1 token equals about 4 characters.

So how can we make a tokenizer using Python? There is a type of tokenizer called Byte-Level BPE tokenizer, and this is how we write in Python:

**Python Library**
`
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers import AddedToken
`
**File and Special Tokens (for training use after)**
`
TOKENIZER_FILE = "tokenizer.json"
`
`
SPECIAL = [ 
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>
]
`
**Class BPE_tokenizer**

`class BPETokenizer:`


    def __init__(self):

        self.tokenizer = None



    def train(self, text):

        print(
            "Training HuggingFace BPE tokenizer..."
        )


        self.tokenizer = Tokenizer(
            BPE(
                unk_token="<|unk|>"
            )
        )


        # Byte-level BPE preprocessing
        self.tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=True
        )


        # IMPORTANT:
        # This fixes Ġ and Ċ decoding
        self.tokenizer.decoder = ByteLevelDecoder()



        special_tokens = [
            AddedToken(
                x,
                special=True
            )
            for x in SPECIAL
        ]



        trainer = BpeTrainer(
            vocab_size=CONFIG["vocab_size"],
            special_tokens=special_tokens
        )



        def text_iterator():

            chunk_size = 1000000

            for i in range(
                0,
                len(text),
                chunk_size
            ):

                yield text[
                    i:i+chunk_size
                ]



        self.tokenizer.train_from_iterator(
            text_iterator(),
            trainer=trainer
        )


        print(
            "Vocabulary:",
            self.tokenizer.get_vocab_size()
        )

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def encode(self, text):
        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        ).ids

    def decode(self, ids):
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=False
        )

    def get_vocab_size(self):

        return self.tokenizer.get_vocab_size()





    def save(self):

        self.tokenizer.save(
            TOKENIZER_FILE
        )

        print(
            "Tokenizer saved:",
            TOKENIZER_FILE
        )

    def load(self):

        print("Loading tokenizer...")

        self.tokenizer = Tokenizer.from_file(
            TOKENIZER_FILE
        )

        self.tokenizer.decoder = ByteLevelDecoder()

        for token in SPECIAL:

            if self.tokenizer.token_to_id(token) is None:
                print(
                    "WARNING missing:",
                    token
                )

        print(
            "Vocabulary:",
            self.tokenizer.get_vocab_size()
        )
`
`


