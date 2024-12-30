import os
import re
from collections import Counter
from docx import Document
import fitz
import pandas as pd
import math

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

# Fungsi untuk menghitung kemiripan dokumen menggunakan cosine similarity
def cosine_similarity(vec1, vec2):
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[word] * vec2[word] for word in intersection])
    sum1 = sum([vec1[word] ** 2 for word in vec1]) ** 0.5
    sum2 = sum([vec2[word] ** 2 for word in vec2]) ** 0.5
    denominator = sum1 * sum2
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


def display_similarity(file_paths, stopwords, kamus, query):
    # Preprocessing query
    query_words = tokenize(query)
    query_words_stemmed = stem_words(query_words, kamus)

    # Menghitung bobot term
    document_word_counts = {}
    for file_path in file_paths:
        word_counts_stopwords = process_file_stopwords(file_path, stopwords)
        word_counts_stemming = process_file_stemming(file_path, kamus)
        
        # Gabungkan stopword removal dan stemming
        document_word_counts[file_path] = {**word_counts_stopwords, **word_counts_stemming}

    # Matriks bobot
    terms = set(query_words_stemmed)
    for file_path in file_paths:
        for word in document_word_counts[file_path]:
            terms.add(word)
    
    term_weights = {term: idx for idx, term in enumerate(sorted(terms))}

    # Membuat vector term untuk setiap dokumen
    doc_vectors = {}
    for file_path in file_paths:
        vector = [document_word_counts[file_path].get(term, 0) for term in sorted(terms)]
        doc_vectors[file_path] = vector
    
    # Query vector dengan menghitung frekuensi term dalam query
    query_vector = [query_words_stemmed.count(term) for term in sorted(terms)]

    # Menampilkan informasi proses
    print("=== Proses 3: Cari Query ===")
    print(f"Masukkan query: {query}")
    print("\n(VSM)")

    print("diketahui:")
    for idx, file_path in enumerate(file_paths, 1):
        print(f"D{idx} = {os.path.basename(file_path)}")

    # Menampilkan bobot term
    print("\nBobot term:")
    print("+--+--------------+" + "--------------" * len(query_words) + "+")
    header = "|     | " + " | ".join(query_words) + " |"
    print(header)
    print("+--+--------------+" + "--------------" * len(query_words) + "+")
    
    for file_path in file_paths:
        vector = doc_vectors[file_path]
        row = f"|D{file_paths.index(file_path) + 1}|"
        for word in query_words:
            if word in term_weights:
                weight = vector[term_weights[word]]
            else:
                weight = 0
            row += f" {weight:<12} |"
        print(row)

    print("+--+--------------+" + "--------------" * len(query_words) + "+")
    
    # Query vector
    query_vector_display = [str(query_vector[term_weights[word]]) if word in term_weights else '0' for word in query_words]
    print(f"|Q | {' | '.join(query_vector_display)} |")

    # Perhitungan cosine similarity
    def vector_length(vector):
        return math.sqrt(sum(x**2 for x in vector))

    def cosine_similarity_manual(vec1, vec2):
        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        return dot_product / (vector_length(vec1) * vector_length(vec2))

    similarities = {}
    print("\nPerhitungan:")
    for idx, file_path in enumerate(file_paths, 1):
        vector_d = doc_vectors[file_path]
        print(f"\nD{idx} = (", end="") 
        for i, word in enumerate(query_words_stemmed):
            print(f"({vector_d[i]} x {query_vector[i]})", end=" + " if i < len(query_words_stemmed) - 1 else ")\n")
        
        dot_product = sum(x * y for x, y in zip(vector_d, query_vector))
        length_d = vector_length(vector_d)
        length_q = vector_length(query_vector)
        sim = cosine_similarity_manual(vector_d, query_vector)
        similarities[file_path] = sim

        # Detail perhitungan
        print(f"Sim(D{idx}, Q) = (({', '.join([f'{vector_d[i]}x{query_vector[i]}' for i in range(len(query_words_stemmed))])})) / "
              f"(√({' + '.join([f'{x**2}' for x in vector_d])})) (√({' + '.join([f'{x**2}' for x in query_vector])}))")
        print(f"        = ({dot_product}) / (√({sum(x**2 for x in vector_d)})) (√({sum(x**2 for x in query_vector)}))")
        print(f"        = {sim:.5f}")

    # Hasil cosine similarity
    print("\nHasil Kemiripan:")
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for rank, (file_path, sim) in enumerate(sorted_similarities, 1):
        print(f"{rank}. D{file_paths.index(file_path)+1} = {sim:.5f} -> {os.path.basename(file_path)}")



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

    # Proses pencarian query
    print("\n=== Proses 3: Cari Query ===")
    query = input("Masukkan query: ").strip()
    display_similarity(file_paths, stopwords, kamus, query)

if __name__ == "__main__":
    main()
