import math
import os

# Fungsi untuk tokenisasi
def tokenize(text):
    return text.lower().split()

# Fungsi untuk stemming (misal menggunakan kamus)
def stem_words(words, kamus):
    return [kamus.get(word, word) for word in words]

# Fungsi untuk menghitung frekuensi kata dalam dokumen
def count_word_frequencies(file_path, stopwords, kamus):
    word_counts = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = tokenize(text)
        words = stem_words(words, kamus)
        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

# Fungsi untuk menghitung panjang vektor
def vector_length(vector):
    return math.sqrt(sum(x**2 for x in vector))

# Fungsi untuk menghitung cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    return dot_product / (vector_length(vec1) * vector_length(vec2))

# Fungsi utama untuk menampilkan kemiripan
def display_similarity(file_paths, stopwords, kamus, query):
    # Tokenisasi dan stemming query
    query_words = tokenize(query)
    query_words_stemmed = stem_words(query_words, kamus)

    # Menghitung frekuensi kata pada query
    query_word_counts = {}
    for word in query_words_stemmed:
        query_word_counts[word] = query_word_counts.get(word, 0) + 1

    # Menyusun daftar semua kata yang ada dalam dokumen dan query
    all_words = set(query_word_counts.keys())
    document_word_counts = {}
    for file_path in file_paths:
        doc_word_counts = count_word_frequencies(file_path, stopwords, kamus)
        document_word_counts[file_path] = doc_word_counts
        all_words.update(doc_word_counts.keys())

    all_words = sorted(all_words)

    # Membuat vektor term untuk setiap dokumen
    doc_vectors = {}
    for file_path in file_paths:
        vector = [document_word_counts[file_path].get(word, 0) for word in all_words]
        doc_vectors[file_path] = vector

    # Membuat query vector
    query_vector = [query_word_counts.get(word, 0) for word in all_words]

    # Menampilkan informasi
    print("=== Proses 3: Cari Query ===")
    print(f"Masukkan query: {query}")
    print("\n(VSM)")

    # Menampilkan bobot term
    print("\nBobot term:")
    print("+--+--------------+" + "--------------" * len(all_words) + "+")
    header = "|     | " + " | ".join(all_words) + " |"
    print(header)
    print("+--+--------------+" + "--------------" * len(all_words) + "+")
    
    for idx, file_path in enumerate(file_paths, 1):
        vector = doc_vectors[file_path]
        row = f"|D{idx} |"
        for word in all_words:
            weight = vector[all_words.index(word)]
            row += f" {weight:<12} |"
        print(row)

    print("+--+--------------+" + "--------------" * len(all_words) + "+")
    
    # Menampilkan query vector
    query_vector_display = [str(query_vector[all_words.index(word)]) for word in all_words]
    print(f"|Q | {' | '.join(query_vector_display)} |")

    # Perhitungan cosine similarity dan hasil
    similarities = {}
    print("\nPerhitungan:")
    for idx, file_path in enumerate(file_paths, 1):
        vector_d = doc_vectors[file_path]
        sim = cosine_similarity(vector_d, query_vector)
        similarities[file_path] = sim

        # Detail perhitungan
        print(f"\nD{idx} = (", end="") 
        for i, word in enumerate(all_words):
            print(f"({vector_d[i]} x {query_vector[i]})", end=" + " if i < len(all_words) - 1 else ")\n")
        
        print(f"Sim(D{idx}, Q) = {sim:.5f}")

    # Hasil cosine similarity
    print("\nHasil Kemiripan:")
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for rank, (file_path, sim) in enumerate(sorted_similarities, 1):
        print(f"{rank}. D{file_paths.index(file_path)+1} = {sim:.5f} -> {os.path.basename(file_path)}")

# Contoh pemanggilan fungsi
stopwords = {"dan", "di", "ke", "dari", "yang"}  # Stopword contoh
kamus = {"belajar": "belajar", "python": "python"}  # Kamus contoh

file_paths = ["document/dokumen 1.txt", "document/dokumen 2.txt"]
query = "belajar python"

# Memanggil fungsi untuk menghitung dan menampilkan kemiripan
display_similarity(file_paths, stopwords, kamus, query)
