import torch
import torch.nn as nn
import random
import json
import os
import sys
import time

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers import AddedToken



# ==========================================================
# CONFIG
# ==========================================================

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




device = "cpu"

print("Device:", device)



# ==========================================================
# FILES
# ==========================================================

CHECKPOINT = "checkpoint.pt"

CHECKPOINT_DIR = "checkpoints"

TOKENIZER_FILE = "tokenizer.json"

TRAIN_HISTORY = "training_history.json"

REPLAY_FILE = "replay.txt"

CHAT_FILE = "chat_memory.txt"


os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)



SPECIAL = [

    "<|user|>",

    "<|assistant|>",

    "<|system|>",

    "<|end|>"

]



# ==========================================================
# HUGGINGFACE BPE TOKENIZER
# ==========================================================

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
tokenizer = BPETokenizer()



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




# ==========================================================
# TRANSFORMER BLOCK
# ==========================================================

class Block(nn.Module):


    def __init__(self):

        super().__init__()


        d = CONFIG["dim"]


        self.norm1 = nn.LayerNorm(
            d
        )


        self.attn = nn.MultiheadAttention(

            d,

            CONFIG["heads"],

            dropout=CONFIG["dropout"],

            batch_first=True

        )


        self.norm2 = nn.LayerNorm(
            d
        )


        self.ff = nn.Sequential(

            nn.Linear(
                d,
                d * 4
            ),

            nn.GELU(),

            nn.Dropout(
                CONFIG["dropout"]
            ),

            nn.Linear(
                d * 4,
                d
            ),

            nn.Dropout(
                CONFIG["dropout"]
            )

        )



    def forward(
        self,
        x,
        mask
    ):


        h = self.norm1(
            x
        )


        a, _ = self.attn(

            h,

            h,

            h,

            attn_mask=mask

        )


        x = x + a


        x = x + self.ff(
            self.norm2(x)
        )


        return x




# ==========================================================
# GPT MODEL
# ==========================================================

class GPT(nn.Module):


    def __init__(
        self,
        vocab
    ):

        super().__init__()


        d = CONFIG["dim"]


        self.context = CONFIG["context"]


        self.token = nn.Embedding(

            vocab,

            d

        )


        self.pos = nn.Embedding(

            self.context,

            d

        )


        self.blocks = nn.ModuleList(

            [

                Block()

                for _ in range(
                    CONFIG["layers"]
                )

            ]

        )


        self.norm = nn.LayerNorm(
            d
        )


        self.head = nn.Linear(

            d,

            vocab,

            bias=False

        )


        self.head.weight = self.token.weight



    def forward(
        self,
        x
    ):


        B,T = x.shape


        positions = torch.arange(

            T,

            device=x.device

        )


        x = (

            self.token(x)

            +

            self.pos(
                positions
            )

        )


        mask = torch.triu(

            torch.ones(

                T,

                T,

                device=x.device

            ),

            diagonal=1

        ).bool()



        for block in self.blocks:

            x = block(

                x,

                mask

            )


        return self.head(

            self.norm(x)

        )
# ==========================================================
# DATASET LOADING
# ==========================================================

import pandas as pd



def load_dataset(dataset_file):

    print("Loading dataset:", dataset_file)


    if dataset_file.endswith(".parquet"):

        print("Reading parquet...")

        df = pd.read_parquet(
            dataset_file
        )

        text = ""

        for _, row in df.iterrows():
            text += row["conversation"]


    elif dataset_file.endswith(".txt"):

        print("Reading text file...")

        chunks = []

        with open(
                dataset_file,
                "r",
                encoding="utf-8"
        ) as f:

            for i, line in enumerate(f):

                chunks.append(line)

                if i % 100000 == 0:
                    print(
                        "Loaded lines:",
                        i
                    )

        text = "".join(chunks)


    else:

        raise ValueError(
            "Unsupported file type"
        )


    print(
        "Characters:",
        len(text)
    )


    return text




# ==========================================================
# TOKEN DATA PREPARATION
# ==========================================================

import numpy as np


import numpy as np


def create_training_tokens(text):

    output = "tokens.bin"

    with open(output, "wb") as f:

        total = 0

        for i in range(0, len(text), 1000000):

            chunk = text[i:i+1000000]

            ids = tokenizer.encode(chunk)

            arr = np.array(
                ids,
                dtype=np.uint32
            )

            arr.tofile(f)

            total += len(arr)


            print(
                "Encoded",
                i,
                "/",
                len(text),
                "tokens:",
                total
            )


    print(
        "Saved tokens:",
        total
    )


    return np.memmap(
        output,
        dtype=np.uint32,
        mode="r"
    )



# ==========================================================
# CHECKPOINT
# ==========================================================

def save_checkpoint(

    model,

    step,

    tokens_seen

):


    path = os.path.join(

        CHECKPOINT_DIR,

        "checkpoint_latest.pt"

    )

    torch.save({
        "model": model.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen
    }, path)


    print(
        "Checkpoint saved:",
        path
    )




def load_checkpoint(model, optimizer):

    path = "checkpoints/checkpoint_latest_15005_(1).pt"

    if not os.path.exists(path):
        return 0,0

    print("Loading checkpoint...")

    ck = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        ck["model"]
    )

    print("Model loaded")

    step = ck.get(
        "step",
        0
    )

    tokens_seen = ck.get(
        "tokens_seen",
        0
    )

    return step, tokens_seen




# ==========================================================
# TRAINING
# ==========================================================
def train(
    dataset_file
):

    text = load_dataset(
        dataset_file
    )


    load_or_create_tokenizer(
        text
    )


    vocab = tokenizer.get_vocab_size()


    tokens = create_training_tokens(
        text
    )


    model = GPT(
        vocab
    ).to(
        device
    )


    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"]
    )


    # DirectML does not use CUDA GradScaler
    scaler = None


    step, tokens_seen = load_checkpoint(
        model,
        optimizer,
    )


    model.train()


    for step in range(
        step,
        CONFIG["steps"]
    ):


        optimizer.zero_grad()


        total_loss = 0


        for _ in range(
            CONFIG["gradient_accumulation"]
        ):


            batch = []
            targets = []


            for b in range(
                CONFIG["batch"]
            ):


                s = random.randint(
                    0,
                    len(tokens)-CONFIG["context"]-1
                )

                x = torch.tensor(
                    tokens[s:s + CONFIG["context"]],
                    dtype=torch.long
                )

                y = torch.tensor(
                    tokens[s + 1:s + CONFIG["context"] + 1],
                    dtype=torch.long
                )


                batch.append(x)
                targets.append(y)



            x = torch.stack(batch).to(device)


            y = torch.stack(
                targets
            ).to(
                device
            )


            # No CUDA autocast
            logits = model(
                x
            )


            loss = nn.functional.cross_entropy(
                logits.view(
                    -1,
                    vocab
                ),
                y.view(-1)
            )


            # gradient accumulation
            loss = loss / CONFIG["gradient_accumulation"]



            loss.backward()



            total_loss += (
                loss.item()
                *
                CONFIG["gradient_accumulation"]
            )



        optimizer.step()

        tokens_seen += (
                CONFIG["batch"]
                *
                CONFIG["context"]
                *
                CONFIG["gradient_accumulation"]
        )



        if step % 1 == 0:


            print(
                "step:",
                step,
                "loss:",
                total_loss / CONFIG["gradient_accumulation"]
            )


            save_checkpoint(
                model,
                step,
                tokens_seen
            )


    print(
        "Training finished"
    )
# ==========================================================
# TEXT GENERATION
# ==========================================================


def generate(
    model,
    prompt,
    max_tokens=400,
    temperature=0.6,
    min_tokens=20,
):

    end_id = tokenizer.tokenizer.token_to_id(
        "<|end|>"
    )


    if end_id is None:

        raise ValueError(
            "<|end|> missing from tokenizer"
        )


    model.eval()


    ids = tokenizer.encode(
        prompt
    )


    x = torch.tensor(
        ids,
        dtype=torch.long
    ).unsqueeze(0).to(
        device
    )


    # Protect context length
    if x.shape[1] > CONFIG["context"]:

        x = x[:, -CONFIG["context"]:]


    generated_ids = []


    for _ in range(
        max_tokens
    ):


        # Keep latest context window

        if x.shape[1] >= CONFIG["context"]:

            x = x[:, -CONFIG["context"]:]


        with torch.no_grad():

            logits = model(
                x
            )


        # Only predict next token

        logits = logits[:, -1, :]


        # Temperature sampling

        logits = logits / temperature


        probs = torch.softmax(
            logits,
            dim=-1
        )


        # Prevent ending too early

        if len(generated_ids) < min_tokens:

            probs[0, end_id] = 0

            probs = probs / probs.sum()



        next_token = torch.multinomial(
            probs,
            1
        )


        token = next_token.item()



        # Stop generation

        if token == end_id:

            break



        generated_ids.append(
            token
        )



        x = torch.cat(
            [
                x,
                next_token
            ],
            dim=1
        )



    output = tokenizer.decode(
        generated_ids
    )


    return output




# ==========================================================
# LOAD MODEL FOR CHAT
# ==========================================================

def load_chat_model():


    if not os.path.exists(

        TOKENIZER_FILE

    ):

        print(
            "Tokenizer not found."
        )

        return None



    tokenizer.load()


    vocab = tokenizer.get_vocab_size()



    model = GPT(

        vocab

    ).to(

        device

    )



    ck_path = os.path.join(

        CHECKPOINT_DIR,

        "checkpoint_latest.pt"

    )



    if not os.path.exists(

        ck_path

    ):

        print(
            "Checkpoint not found."
        )

        return None



    ck = torch.load(

        ck_path,

        map_location=device

    )



    model.load_state_dict(

        ck["model"]

    )



    model.eval()



    print(
        "Model loaded."
    )


    return model





# ==========================================================
# CHAT MODE
# ==========================================================

def chat():


    model = load_chat_model()


    if model is None:

        return



    print()

    print(
        "TinyGPT Chat"
    )

    print(
        "Type exit to quit"
    )

    print()



    while True:


        user = input(

            "User: "

        )


        if user.lower() == "exit":

            break

        prompt = (
                "<|system|>\n"
                "You are a helpful AI assistant.\n"
                "<|user|>\n"
                +
                user
                +
                "\n<|assistant|>\n"
        )



        response = generate(

            model,

            prompt,

            max_tokens=400

        )



        print()

        print(

            "AI:",

            response

        )

        print()




# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":


    if len(sys.argv) < 2:


        print(
            """
Usage:

Training:
python ai_t2_fixed.py --train dataset.parquet


Chat:
python ai_t2_fixed.py --chat

"""
        )

        sys.exit()



    command = sys.argv[1]



    if command == "--train":


        if len(sys.argv) < 3:

            print(
                "Missing dataset file"
            )

            sys.exit()



        train(

            sys.argv[2]

        )



    elif command == "--chat":


        chat()



    else:


        print(
            "Unknown command"
        )
