import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu

# Pastikan model tokenizer NLTK yang diperlukan sudah terunduh
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub("\n", " ", text)  # Menghapus karakter newline
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Menghapus tautan
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Menghapus karakter non-ASCII
    text = re.sub(r"\d+", "", text)  # Menghapus digit
    text = re.sub(r"[^\w\s]", "", text)  # Menghapus karakter khusus
    text = re.sub(r"[{}]".format(string.punctuation), "", text)  # Menghapus tanda baca
    return text

# Normalisasi kata
normalized_word = pd.read_excel('Slangword-indonesian.xlsm')
normalized_word_dict = dict(zip(normalized_word['slang'], normalized_word['formal']))

def normalized_term(document):
    return ' '.join([normalized_word_dict.get(term, term) for term in document.split()])

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# Muat stopwords dari NLTK
stopword = stopwords.words('indonesian')

# Fungsi untuk membaca stopwords dari file teks
def stopword_rm(filename):
    with open(filename, 'r') as file:
        stopwords_list = file.readlines()
    return [word.strip() for word in stopwords_list]

# Baca stopwords dari file teks
txt_stopword_path = "id.stopwords.02.01.2016.txt"
additional_stopwords = stopword_rm(txt_stopword_path)

# Tambahkan stopwords tambahan ke dalam daftar
stopword.extend(["d", "an", "semapt", "deh", "so", "di", "nya", "sd", "sih", "ke", "lah", "se", "dong", "dst", "ya"])
stopword.extend(additional_stopwords)
stopword = set(stopword)

def stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk stemming tokens
def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    text = clean_text(text)
    text = text.lower()  # Case folding
    text = normalized_term(text)
    text = word_tokenize(text)  # Tokenizing
    text = stopwords(text)
    text = stem_tokens(text)
    return ' '.join(text)

df_fitur = pd.read_csv("tfidf_with_sentiment.csv")

# Fungsi untuk melatih dan mengevaluasi model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear')
    
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    return svm_model, accuracy

# Aplikasi Streamlit
st.set_page_config(
    page_title="TA-Aderisa Dyta Okvianti",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Sidebar dengan menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Dataset", "Implementasi", "Grafik"],
        styles={
            "container": {"background-color": "#00008B"},
            "nav-link": {
                "font-size": "16px",
                "color": "white",
                "padding": "10px 20px",
                "transition": "0.3s",
            },
            "nav-link-hover": {
                "background-color": "#4169E1",
                "color": "white",
            },
            "nav-link-selected": {
                "background-color": "#1E90FF",
                "color": "white",
                "font-weight": "bold",
            }
        }
    )
# Home
if selected == "Home":
    st.write(""" <center><h3 style = "text-align: center;">ANALISIS SENTIMEN DESTINASI PARIWISATA DI MADURA MENGGUNAKAN SUPPORT VECTOR MACHINE DENGAN SELEKSI FITUR RECURSIVE FEATURE ELIMINATION</h3></center>
        """,unsafe_allow_html=True)
    st.subheader("Penjelasan Singkat")
    st.write("""
        <div style="text-align:justify;">
        Dalam aplikasi ini, akan melakukan analisis sentimen Destinasi Pariwisata Madura dari ulasan Google Maps dengan menggunakan
        metode Support Vector Machine. Dengan analisis ini diharapkan bisa memberikan pandangan masyarakat tentang destinasi pariwisata tertentu. 
        </div>
    """, unsafe_allow_html=True)

# Dataset
elif selected == "Dataset":
    st.subheader("Dataset Pariwisata Madura")
    st.write("""
        <div style="text-align:justify;">
        Data penelitian ini diperoleh melalui Google Maps. Jumlah data pariwisata 1.722 ulasan, mencakup informasi dari 6 tempat populer di Madura
        yaitu Air Terjun Toroan, Api Tak Kunjung Padam, Pantai Sembilan, Bukit Jaddih, Pantai Slopeng, Pantai Lombang
        </div>
    """, unsafe_allow_html=True)

    
    # Load datasets
    df_raw = pd.read_excel('databaru 3 (2).xlsx')
    df_preprocessed = pd.read_excel('hasil_preprocessing.xlsx')
    
    st.write("### Dataset Sebelum Preprocessing")
    st.write(df_raw)
    
    st.write("### Dataset Setelah Preprocessing")
    st.write(df_preprocessed)

# Prediksi Teks Tunggal
elif selected == "Implementasi":
    st.write(""" <center><h3 style = "text-align: center;">Implementasi</h3></center>""",unsafe_allow_html=True)
    user_input = st.text_area("", key="input_text", placeholder="Masukkan teks untuk prediksi sentimen:", height=150)

    if st.button("Prediksi Sentimen", key="prediksi_button"):
        if not user_input.strip():
            st.error("Silakan masukkan kalimat.")
        else:
            preprocessed_text = preprocess_text(user_input)
            
            selected_features = df_fitur.columns.drop('sentimen')

            X_selected = df_fitur[selected_features]
            svm_model, accuracy = train_and_evaluate_model(X_selected, df_fitur['sentimen'])

            vectorized_text = pd.DataFrame([dict.fromkeys(selected_features, 0)])
            for word in preprocessed_text.split():
                if word in vectorized_text.columns:
                    vectorized_text[word] += 1

            prediction = svm_model.predict(vectorized_text)
            sentiment = "Positif" if prediction[0] == 1 else "Negatif"
            
            # Display sentiment with background color
            if sentiment == "Positif":
                st.markdown(f"<div style='padding: 10px; background-color: #d4edda; color: #155724;'>{sentiment}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding: 10px; background-color: #f8d7da; color: #721c24;'>{sentiment}</div>", unsafe_allow_html=True)

# Prediksi Teks Tunggal
elif selected == "Grafik":
    st.markdown(
    """
    <h2 style="text-align: center;">Grafik SVM Dengan Seleksi Fitur RFE</h2>
    """,
    unsafe_allow_html=True)
    st.write("""<h4> 1. SVM-RFE Pembagian data 70% : 30%</h4>""",unsafe_allow_html=True)
    st.image("hasil grafik 70%.png")
    st.write("""
        <div style="text-align:justify;">
        grafik ini menunjukkan hasil klasifikasi SVM dengan Seleksi Fitur RFE pada pembagian data 70% untuk data training dan 30% untuk data testing, dengan N-Top 99%, 98%, 97% sampai dengan 1%. 
        Dari proses yang telah dilakukan terdapat hasil terbaik pada penggunaan fitur 51% dengan akurasi 74.08%. 
        </div>
    """, unsafe_allow_html=True)

    st.write("""<h4> 2. SVM-RFE Pembagian data 80% : 20%</h4>""",unsafe_allow_html=True)
    st.image("hasil grafik 80%.png")
    st.write("""
        <div style="text-align:justify;">
        grafik ini menunjukkan hasil klasifikasi SVM dengan Seleksi Fitur RFE pada pembagian data 80% untuk data training dan 20% untuk data testing, dengan N-Top 99%, 98%, 97% sampai dengan 1%. 
        Dari proses yang telah dilakukan terdapat hasil terbaik pada penggunaan fitur 6% dengan akurasi 74.20%. 
        </div>
    """, unsafe_allow_html=True)

