import argparse
import re
from tqdm import tqdm

def clean_file(input_path, output_path):
    MIN_LENGTH = 10
    PUNC_RATIO = 0.5

    punctuations = re.compile(r'[、。「」『』（）!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]')

    print(f"'{input_path}' > Cleaning start")

    line_count = 0
    written_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in):
            line_count += 1
            line = line.strip()

            if len(line) < MIN_LENGTH:
                continue

            punc_count = len(punctuations.findall(line))
            if len(line) > 0 and (punc_count / len(line)) > PUNC_RATIO:
                continue

            f_out.write(line + '\n')
            written_count += 1

    print(f"Cleaning finish > {line_count} -> {written_count} saved -> '{output_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Corpus cleaning script")
    parser.add_argument("--input", required=True, help="Input path")
    parser.add_argument("--output", required=True, help="Output path")
    args = parser.parse_args()

    clean_file(args.input, args.output)