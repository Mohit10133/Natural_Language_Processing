import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

st.set_page_config(page_title="NLP Sparse Vector & Embedding Lab", layout="wide")

st.title("🔍 NLP Lab 5: Sparse Vector (Embedding) & Similarity Measures")

# Tabs for each question
tab1, tab2, tab3 = st.tabs(["📊 TF-IDF & Euclidean Normalization", "📈 Cosine Similarity", "📌 Pointwise Mutual Information (PMI)"])

# -------------------------------
# Q1: TF-IDF & Euclidean Normalization
# -------------------------------
with tab1:
    st.header("📊 Question 1: TF-IDF & Euclidean Normalization")

    st.markdown("""
    **Instructions:**
    1. Enter term frequencies for each document.
    2. Provide IDF values.
    3. Enter your query to compute document scores.
    """)

    # Default term frequency table
    terms = ["car", "auto", "insurance", "best"]
    tf_data = pd.DataFrame({
        "Term": terms,
        "Doc1": [27, 3, 0, 14],
        "Doc2": [4, 33, 33, 0],
        "Doc3": [24, 0, 29, 17]
    })

    idf_values = pd.DataFrame({
        "Term": terms,
        "IDF": [1.65, 2.08, 1.62, 1.5]
    })

    st.subheader("📥 Edit Term Frequencies & IDF Values")
    tf_data = st.data_editor(tf_data, num_rows="fixed")
    idf_values = st.data_editor(idf_values, num_rows="fixed")

    query = st.text_input("Enter your query (e.g., 'car insurance')", "car insurance")

    # Merge TF and IDF
    merged_df = pd.merge(tf_data, idf_values, on="Term")
    tfidf_df = merged_df.copy()

    # Calculate TF-IDF
    for doc in ["Doc1", "Doc2", "Doc3"]:
        tfidf_df[doc] = tfidf_df[doc] * tfidf_df["IDF"]

    st.subheader("📄 TF-IDF Table")
    st.dataframe(tfidf_df[["Term", "Doc1", "Doc2", "Doc3"]])

    # Query Score
    query_terms = query.lower().split()
    scores = {}
    for doc in ["Doc1", "Doc2", "Doc3"]:
        scores[doc] = sum(tfidf_df.loc[tfidf_df["Term"].isin(query_terms), doc])

    st.subheader("🔎 Document Scores for Query")
    st.write(scores)

    # Euclidean Normalization
    st.subheader("📏 Euclidean Normalization")
    norm_df = merged_df.copy()
    for doc in ["Doc1", "Doc2", "Doc3"]:
        norm = np.linalg.norm(norm_df[doc])
        norm_df[doc] = norm_df[doc] / norm if norm != 0 else norm_df[doc]
    st.dataframe(norm_df[["Term", "Doc1", "Doc2", "Doc3"]])

# -------------------------------
# Q2: Cosine Similarity
# -------------------------------
with tab2:
    st.header("📈 Question 2: Cosine Similarity")
    st.markdown("""
    **Steps:**
    1. Upload or type N documents.
    2. Choose vectorization type (Count Vector or TF-IDF).
    3. Enter a word to find nearest neighbours.
    """)

    docs_input = st.text_area("Enter documents (one per line):", 
                              "car insurance best car\nbest auto insurance\ncar and auto are vehicles")

    vector_type = st.radio("Choose Vectorization Type:", ["Count Vector", "TF-IDF"])

    docs = docs_input.strip().split("\n")
    if vector_type == "Count Vector":
        vectorizer = CountVectorizer()
    else:
        vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    df_matrix = pd.DataFrame(X.toarray(), columns=terms)

    st.subheader("📄 Vector Representation")
    st.dataframe(df_matrix)

    word = st.text_input("Enter a word to find nearest neighbours:", "car")
    if word in terms:
        idx = terms.tolist().index(word)
        cosine_sim = cosine_similarity(X.T)
        similarities = list(enumerate(cosine_sim[idx]))
        sorted_similarities = sorted(similarities, key=lambda x: -x[1])
        nearest_words = [(terms[i], score) for i, score in sorted_similarities if i != idx]
        st.subheader(f"🔍 Nearest Words to '{word}'")
        st.write(nearest_words)
    else:
        st.warning("Word not found in vocabulary.")

# -------------------------------
# Q3: PMI
# -------------------------------
with tab3:
    st.header("📌 Question 3: Pointwise Mutual Information (PMI)")
    st.markdown("""
    **PMI Formula:**
    \n
    $PMI(w_i, w_j) = \\log \\frac{p(w_i, w_j)}{p(w_i)p(w_j)}$
    """)

    text_input = st.text_area("Enter text for PMI calculation:",
                              "car insurance best car\nbest auto insurance\ncar and auto are vehicles")

    tokens = text_input.lower().split()
    total_count = len(tokens)
    token_counts = Counter(tokens)
    bigram_counts = Counter(zip(tokens, tokens[1:]))

    def p(word):
        return token_counts[word] / total_count

    def p_bigram(w1, w2):
        return bigram_counts[(w1, w2)] / (total_count - 1)

    word1 = st.text_input("Word 1:", "car")
    word2 = st.text_input("Word 2:", "insurance")

    if word1 in token_counts and word2 in token_counts:
        pmi = math.log2(p_bigram(word1, word2) / (p(word1) * p(word2))) if p_bigram(word1, word2) > 0 else float('-inf')
        st.write(f"📊 PMI({word1}, {word2}) = {pmi:.4f}")
    else:
        st.warning("One or both words not found in text.")

    st.subheader("📄 Bigram PMI Table")
    pmi_table = []
    for (w1, w2), count in bigram_counts.items():
        prob_w1 = p(w1)
        prob_w2 = p(w2)
        prob_w1w2 = p_bigram(w1, w2)
        if prob_w1w2 > 0:
            pmi_val = math.log2(prob_w1w2 / (prob_w1 * prob_w2))
            pmi_table.append((w1, w2, pmi_val))
    pmi_df = pd.DataFrame(pmi_table, columns=["Word1", "Word2", "PMI"])
    st.dataframe(pmi_df)
