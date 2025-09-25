import argparse
import re
import random
from collections import Counter
from janome.tokenizer import Tokenizer
from tqdm import tqdm
import unicodedata

MIN_LENGTH = 50
MAX_LENGTH = 30000
CHAR_REPETITION_LIMIT = 10
SYMBOL_RATIO_THRESHOLD = 0.3
NGRAM_REPETITION_THRESHOLD = 0.2
NOUN_RATIO_THRESHOLD = 0.8
ADULT_KEYWORD_REMOVAL_PROB = 0.9

ADULT_KEYWORDS = ["アダルト", "成人向け", "出会い", "無料動画"]
AD_KEYWORDS = ["激安", "セール", "キャンペーン", "限定価格"]

CONTROL_CHAR_REGEX = re.compile(r'[\x00-\x1f\x7f]')
URL_REGEX = re.compile(r'https?://[^\s/$.?#].[^\s]*')
HTML_TAG_REGEX = re.compile(r'<[^>]+>')
COPYRIGHT_REGEX = re.compile(r'©|Copyright|All rights reserved', re.IGNORECASE)
NAV_MENU_REGEX = re.compile(r'ホーム|プライバシーポリシー|お問い合わせ|サイトマップ')
SYMBOL_REGEX = re.compile(r'[\s\W_]|[^\w\s]')

JANOME_TOKENIZER = Tokenizer()

def remove_control_chars(text):
    return CONTROL_CHAR_REGEX.sub('', text)

def is_outside_length_range(doc):
    return not (MIN_LENGTH <= len(doc) <= MAX_LENGTH)

def has_excessive_char_repetition(doc):
    if not doc:
        return False

    pattern = re.compile(r'(.)\1{' + str(CHAR_REPETITION_LIMIT - 1) + r',}')
    match = pattern.search(doc)
    if match:
        repeated_char = match.group(1)
        return True
    return False

def contains_disallowed_patterns(doc):
    if URL_REGEX.search(doc): return True
    if HTML_TAG_REGEX.search(doc): return True
    if COPYRIGHT_REGEX.search(doc): return True
    if NAV_MENU_REGEX.search(doc): return True
    return False

def has_high_symbol_ratio(doc):
    if not doc:
        return False

    symbol_count = 0
    for char in doc:
        if unicodedata.category(char)[0] in 'PSZC':
            symbol_count += 1

    return (symbol_count / len(doc)) > SYMBOL_RATIO_THRESHOLD

def is_likely_unwanted_content(doc):
    doc_lower = doc.lower()
    for keyword in ADULT_KEYWORDS + AD_KEYWORDS:
        if keyword in doc_lower:
            return random.random() < ADULT_KEYWORD_REMOVAL_PROB
    return False

def has_high_ngram_repetition(doc, n=4):
    if len(doc) < n:
        return False

    ngrams = [doc[i:i+n] for i in range(len(doc) - n + 1)]
    if not ngrams:
        return False

    unique_ngrams = set(ngrams)
    repetition_rate = len(unique_ngrams) / len(ngrams)

    return repetition_rate < NGRAM_REPETITION_THRESHOLD

def has_high_noun_ratio(doc):
    if not doc:
        return False

    tokens = JANOME_TOKENIZER.tokenize(doc)

    noun_count = 0
    token_count = 0
    for token in tokens:
        token_count += 1
        if token.part_of_speech.startswith('Noun'):
            noun_count += 1

    if token_count == 0:
        return False

    return (noun_count / token_count) > NOUN_RATIO_THRESHOLD

def clean_corpus(input_path, output_path):
    print("Cleaning began, this may take time...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
    except FileNotFoundError:
        print(f"Error: '{input_path}' is not found.")
        return

    kept_count = 0
    discarded_counts = Counter()

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for doc in tqdm(f_in, total=total_lines, desc="Cleaning"):
            original_doc = doc.strip()

            doc = remove_control_chars(original_doc)

            if is_outside_length_range(doc):
                discarded_counts['length'] += 1
                continue
            if contains_disallowed_patterns(doc):
                discarded_counts['pattern'] += 1
                continue
            if has_excessive_char_repetition(doc):
                discarded_counts['char_rep'] += 1
                continue
            if has_high_symbol_ratio(doc):
                discarded_counts['symbol_ratio'] += 1
                continue
            if is_likely_unwanted_content(doc):
                discarded_counts['unwanted'] += 1
                continue
            if has_high_ngram_repetition(doc):
                discarded_counts['ngram_rep'] += 1
                continue
            if has_high_noun_ratio(doc):
                discarded_counts['noun_ratio'] += 1
                continue

            f_out.write(doc + '\n')
            kept_count += 1

    print("\n--- Cleaning finish ---")
    print(f"Texts: {total_lines}")
    print(f"Saved texts: {kept_count} ({kept_count/total_lines:.2%})")
    print("Removed texts\n:")
    for reason, count in discarded_counts.items():
        print(f"- {reason}: {count}件")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script of cleaning corpus from rules.")
    parser.add_argument("--input", required=True, help="Input path (corpus.txt)")
    parser.add_argument("--output", required=True, help="Output path (corpus_cleaned.txt)")
    args = parser.parse_args()
    
    clean_corpus(args.input, args.output)
