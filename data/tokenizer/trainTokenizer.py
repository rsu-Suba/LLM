import sentencepiece as spm
import os

input_file = "data/tokenizer/corpus.shuffled.txt"
model_prefix = "data/tokenizer/tokenizer"
vocab_size = 50000
model_type = "bpe"

command = (
    f'--input={input_file} '
    f'--model_prefix={model_prefix} '
    f'--vocab_size={vocab_size} '
    f'--model_type={model_type} '
    f'--character_coverage=0.9995 '
    f'--pad_id=0 '
    f'--unk_id=1 '
    f'--bos_id=2 '
    f'--eos_id=3 '
    f'--input_sentence_size=1000000 '
    f'--shuffle_input_sentence=true '
    f'--num_threads=6 '
)

# print("Tokenizer train start")
# spm.SentencePieceTrainer.train(command)
# print("Tokenizer train finish!")

print("\n--- Tokenizer test ---")
sp = spm.SentencePieceProcessor()
sp.load(f'{model_prefix}.model')

text1 = "大規模言語モデル"
encoded = sp.encode_as_pieces(text1)
ids = sp.encode_as_ids(text1)

print(f"Input: {text1}")
print(f"Tokenizer: {encoded}")
print(f"ID convert: {ids}")

decoded = sp.decode_ids(ids)
print(f"Decoded: {decoded}")