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
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers import AddedToken
```
**Config of the entire model architecture**
```python
CONFIG = {

    # model size
    "dim": 510,
    "layers": 10,
    "heads": 10,

    # context window
    "context": 512,

    "dropout": 0.1,

    # training
    "lr": 3e-4,
    "batch": 2,
    "gradient_accumulation": 16,
    "steps": 88000,

    # tokenizer
    "vocab_size": 8000 
}
```
**File and Special Tokens (for training use after)**
```python
TOKENIZER_FILE = "tokenizer.json"

SPECIAL = [ 
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>,
]
```
**Class BPE_tokenizer**
```python
class BPETokenizer:


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
```
**Create and Load tokenizer**
```python
def load_or_create_tokenizer(text):


    if os.path.exists(
        TOKENIZER_FILE
    ):


        tokenizer.load()


    else:


        print(
            "Creating tokenizer..."
        )


        tokenizer.train(
            text
        )


        tokenizer.save()


    return tokenizer
```

**Lets start with the function train():**
```python
 def train(self, text):

        print(
            "Training HuggingFace BPE tokenizer..."
        )


        self.tokenizer = Tokenizer(
            BPE(
                unk_token="<|unk|>"
            )
        )


        # Byte-level BPE preprocessing23
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
```
```python
def train(self, text):
    print("Training HuggingFace BPE tokenizer...")
```
This function creates a vocabulary from your dataset.

Example:

Your dataset:
```
The cat sat on the mat.
The cat likes milk.
```
The tokenizer learns:
```
Vocabulary:

0   The
1   cat
2   sat
3   on
4   the
5   mat
...
```
Next, the BPE tokenizer will be created:
```python
self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
```
This creates:
```
tokenizer
    |
    |
    v
BPE model
```
**What is unk tokem?**
it stands for unknown token

Example:

Your vocabulary:
```
dog
cat
cow
```
input:
```
qwertyasdf
```
tokenizer cannot find, so it outputs `<|unk|>` instead of crashing
**What is BPE?**

BPE = Byte Pair Encoding.

It learns common pieces of words.

Instead of:
```
unbelievable
```
being one unknown word

BPE may split:
```
un
believe
able
```
or:
```
un
believ
able
```
depending on training.

Why?

Because vocabulary is limited.

the config:
```python
"vocab_size":8000
```
You cannot store every possible word. English has hundreds of thousands of words. So BPE creates reusable pieces.

Next is byte level precessing:
```python
self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
```
