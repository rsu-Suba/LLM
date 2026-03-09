import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/tokenizer.model")

print(f"Vocab size: {sp.get_piece_size()}")
print("\n--- Special Tokens ---")
for i in range(5):
    print(f"ID {i}: '{sp.id_to_piece(i)}'")

print("\n--- First 50 Pieces ---")
pieces = []
for i in range(50):
    pieces.append(sp.id_to_piece(i))
print(pieces)

print("\n--- Encoding Test ---")
test_sentences = [
    "",
    "",
    ""
]

for text in test_sentences:
    encoded = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)
    print(f"\nInput: {text}")
    print(f"Pieces: {encoded}")
    print(f"IDs: {ids}")
    print(f"Decoded: {sp.decode_ids(ids)}")
