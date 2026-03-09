import sentencepiece as spm
import os

input_file = "data/corpus/train.txt"
model_prefix = "data/tokenizer/tokenizer"
vocab_size = 28000
model_type = "bpe"

command = (
    f'--input={input_file} '
    f'--model_prefix={model_prefix} '
    f'--vocab_size={vocab_size} '
    f'--model_type={model_type} '
    f'--character_coverage=1.0 '
    f'--byte_fallback=true '
    f'--pad_id=0 '
    f'--unk_id=1 '
    f'--bos_id=2 '
    f'--eos_id=3 '
    f'--input_sentence_size=1000000 '
    f'--shuffle_input_sentence=true '
    f'--num_threads=8 '
    f'--max_sentence_length=20000 '
    f'--split_by_whitespace=false '
)

print(f"Training Tokenizer (Vocab: {vocab_size}, Type: {model_type})")
spm.SentencePieceTrainer.train(command)
print("Tokenizer train finish!")

print("\n--- New Tokenizer test ---")
sp = spm.SentencePieceProcessor()
sp.load(f'{model_prefix}.model')

test_texts = [
    "",
    "",
    ""
]

for text in test_texts:
    encoded = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)
    print(f"\nInput: {text}")
    print(f"Pieces: {encoded}")
    print(f"IDs: {ids}")
    print(f"Decoded: {sp.decode_ids(ids)}")
