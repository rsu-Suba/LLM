import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/tokenizer.model")

print("Vocab size:", sp.get_piece_size())

for i in range(20):
    print(i, sp.id_to_piece(i))
