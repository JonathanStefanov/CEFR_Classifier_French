import csv
from nltk import sent_tokenize
import nltk

def text_to_csv(input_file, level, output_file):
    nltk.download('punkt')

    # Reading text from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Splitting the text into sentences
    sentences = sent_tokenize(text, language='french')

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header
        writer.writerow(['id', 'sentence', 'difficulty'])

        # Writing each sentence with an ID and difficulty level
        for i, sentence in enumerate(sentences, 1):
            # Split the sentence into words and check if the length is 5 or more
            words = sentence.split()
            if len(words) >= 5:
                writer.writerow([i, sentence, level])

# Example usage
input_file = 'utils/input_text.txt'  # Replace with the path to your text file
output_file = 'datasets/B2_3.csv'
text_to_csv(input_file, "B2", output_file)
