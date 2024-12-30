import os
import re
from collections import Counter
from docx import Document
import fitz
from PyPDF2 import PdfReader
import pandas as pd

# Fungsi untuk membaca file .txt
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Fungsi untuk membaca file .docx
def read_docx(file_path):
    doc = Document(file_path)
    return '\n'.join(para.text for para in doc.paragraphs)

# Fungsi untuk membaca file .pdf
def read_pdf(file_path):
    doc = fitz.open(file_path)
    return '\n'.join(page.get_text() for page in doc)

# Fungsi untuk tokenisasi
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Fungsi untuk case folding
def case_folding(text):
    return text.lower()

# Fungsi untuk memuat stopwords dari file CSV
def load_stopwords_from_csv(file_path):
    stopwords_df = pd.read_csv(file_path, header=None)
    return set(stopwords_df[0].str.strip().tolist())

# Fungsi untuk memfilter stopwords
def remove_stopwords(text, stop_words):
    tokens = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in tokens if word not in stop_words]
    return filtered_words

# Fungsi untuk menghitung kata penting
def count_important_words(text, stop_words):
    filtered_words = remove_stopwords(text, stop_words)
    word_counts = Counter(filtered_words)
    return word_counts

# Fungsi untuk memuat kamus kata dasar
def load_kamus(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file)

# Fungsi untuk menghapus imbuhan dari kata
def remove_affixes(word, kamus):
    if word in kamus:
        return word

    # Aturan khusus
    if word == "mempersiapkan":
        return "siap"
    elif word == "meningkatkan":
        return "tingkat"

    prefixes = ["me", "mem", "men", "meng", "meny", "be", "ber", "per", "pe",
                "di", "ke", "se", "ter", "pem", "pen", "peng", "peny"]
    suffixes = ["kan", "an", "i", "lah", "kah", "tah", "nya", "ku", "mu"]
    infixes = ["el", "em", "er", "in"]

    for prefix in prefixes:
        if word.startswith(prefix):
            stripped_word = word[len(prefix):]
            if stripped_word in kamus:
                return stripped_word

    for suffix in suffixes:
        if word.endswith(suffix):
            stripped_word = word[:-len(suffix)]
            if stripped_word in kamus:
                return stripped_word

    for infix in infixes:
        if infix in word:
            stripped_word = word.replace(infix, "")
            if stripped_word in kamus:
                return stripped_word

    for prefix in prefixes:
        for suffix in suffixes:
            if word.startswith(prefix) and word.endswith(suffix):
                stripped_word = word[len(prefix):-len(suffix)]
                if stripped_word in kamus:
                    return stripped_word

    return word

class Stemmer:
    def __init__(self, kamus):
        self.kamus = kamus

    def stem(self, word):
        return remove_affixes(word, self.kamus)

def stem_words(words, kamus):
    stemmer = Stemmer(kamus)
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

# Fungsi untuk memproses file sesuai format yang dipilih untuk stopword removal
def process_file_stopwords(file_path, stopwords):
    # print(f"\nMembaca file: {file_path}")
    
    if file_path.endswith('.txt'):
        text = read_txt(file_path)
    elif file_path.endswith('.docx'):
        text = read_docx(file_path)
    elif file_path.endswith('.pdf'):
        text = read_pdf(file_path)
    else:
        print(f"Format file {file_path} tidak didukung.")
        return
    
    word_counts = count_important_words(text, stopwords)
    return word_counts

# Fungsi untuk memproses file sesuai format yang dipilih untuk stemming
def process_file_stemming(file_path, kamus):
    # print(f"\nMembaca file: {file_path}")
    
    if file_path.endswith('.txt'):
        text = read_txt(file_path)
    elif file_path.endswith('.docx'):
        text = read_docx(file_path)
    elif file_path.endswith('.pdf'):
        text = read_pdf(file_path)
    else:
        print(f"Format file {file_path} tidak didukung.")
        return
    
    words = tokenize(text)
    stemmed_words = stem_words(words, kamus)
    
    word_counts = Counter(stemmed_words)
    return word_counts

# Program Utama
def main():
    stopword_file = 'data/stopwordbahasa.csv'
    kamus_file = 'data/kamus.txt'

    # Load stopwords dan kamus kata dasar
    stopwords = load_stopwords_from_csv(stopword_file)
    kamus = load_kamus(kamus_file)

    print("Masukkan direktori folder yang berisi file:")
    folder_path = input().strip()

    # Ambil daftar file dalam direktori
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.txt', '.docx', '.pdf'))]

    if not file_paths:
        print("Tidak ada file yang ditemukan di folder tersebut.")
        return

    print(f"\nList file pada direktori {os.path.basename(folder_path)}:")
    for idx, file_path in enumerate(file_paths, 1):
        print(f"{idx}. {os.path.basename(file_path)}")

    print("\n=== Proses 2: PreProcessing ===")
    
    for file_path in file_paths:
        word_counts_stopwords = process_file_stopwords(file_path, stopwords)
        word_counts_stemming = process_file_stemming(file_path, kamus)

        # Tampilkan hasil dalam format tabel
        print(f"\nMembaca file: {os.path.basename(file_path)}")
        print("+-------------------------+-----------------------+--------+")
        print("| hasil stopremoval       | hasil stemming         | jumlah |")
        print("+-------------------------+-----------------------+--------+")
        
        all_words = set(word_counts_stopwords.keys()).union(set(word_counts_stemming.keys()))
        
        for word in sorted(all_words):  # Sort kata-katanya agar tampil teratur
            stopword_count = word_counts_stopwords.get(word, 0)
            stemmed_word = remove_affixes(word, kamus)
            stemmed_count = word_counts_stemming.get(stemmed_word, 0)
            print(f"| {word:<23} | {stemmed_word:<21} | {stopword_count + stemmed_count:<6} |")
        
        print("+-------------------------+-----------------------+--------+")

if __name__ == "__main__":
    main()
