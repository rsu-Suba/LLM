from collections import Counter
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="data/tokenizer/tokenizer.model")

counter = Counter()

target_word = ""

with open("data/corpus/train.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ids = sp.encode(line, out_type=int)
        counter.update(ids)

target_id = sp.piece_to_id(target_word)

count = counter[target_id]

print(f"'{target_word}' was appeared: {count}")
