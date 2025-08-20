import streamlit as st
import nltk
import spacy
import string
import math
from collections import Counter, defaultdict
import pandas as pd
import os


# --- Page Configuration and Initial Setup ---
st.set_page_config(
    page_title="Lab - 3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Asset Downloading and Model Loading ---
@st.cache_resource(show_spinner="Loading NLP models...")
def load_models_and_data():
    """Downloads NLTK data and loads the spaCy model."""
    
    # Set NLTK data path to avoid permission issues
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Try downloading with error handling
    required_datasets = ['brown', 'punkt', 'universal_tagset', 'treebank']
    
    for dataset in required_datasets:
        try:
            nltk.download(dataset, quiet=True, download_dir=nltk_data_dir)
        except Exception as e:
            st.warning(f"Could not download {dataset}: {e}")
            try:
                nltk.download(dataset, quiet=True)
            except:
                st.error(f"Failed to download {dataset}. Please check your internet connection.")
                st.stop()
    
    # Try loading a better spaCy model
    try:
        nlp = spacy.load("en_core_web_lg")  # Large model (preferred)
        st.success("Using spaCy model: en_core_web_lg ✅")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")  # Fallback
            st.warning("Using spaCy model: en_core_web_sm (less accurate). Install en_core_web_lg for better results.")
        except OSError:
            st.error("No spaCy model found. Please run:\n\n"
                     "`python -m spacy download en_core_web_sm`\n"
                     "or\n"
                     "`python -m spacy download en_core_web_lg`")
            st.stop()

    return nlp


nlp = load_models_and_data()


st.title("N-gram, POS, NER")
st.markdown("An interactive tool for exploring N-gram models, POS Tagging, and Named Entity Recognition.")


tab1, tab2, tab3 = st.tabs([
    "1️⃣ N-gram Model & Perplexity",
    "2️⃣ HMM POS Tagger",
    "3️⃣ Named Entity Recognition (NER)",
])



# ==============================================================================
# TAB 1: N-GRAM LANGUAGE MODEL
# ==============================================================================
with tab1:
    st.header("N-gram Language Model")
    st.markdown("This section trains a bigram model on the **Brown Corpus** to calculate probabilities and perplexity.")


    @st.cache_data(show_spinner="Training N-gram model on Brown Corpus...")
    def train_ngram_model():
        """Pads sentences, counts unigrams/bigrams, and returns model components."""
        try:
            padded_sentences = [
                ['<s>'] + [word.lower() for word in sent] + ['</s>']
                for sent in nltk.corpus.brown.sents()
            ]
        except LookupError:
            st.error("Brown corpus not available. Please check your NLTK installation.")
            st.stop()
        
        all_words = [word for sent in padded_sentences for word in sent]
        unigram_counts = Counter(all_words)
        bigram_counts = Counter(nltk.bigrams(all_words))
        
        vocab_size = len(unigram_counts)
        total_tokens = len(all_words)
        
        return unigram_counts, bigram_counts, vocab_size, total_tokens


    # Train the model (uses cache)
    unigram_counts, bigram_counts, vocab_size, total_tokens = train_ngram_model()
    st.success(f"Model trained! Vocabulary Size: **{vocab_size}**, Total Tokens: **{total_tokens}**")


    # --- Unigram Probability ---
    with st.expander("Show Unigram Probability Table (Top 20)"):
        top_unigrams = unigram_counts.most_common(20)
        unigram_data = {
            "Word": [f"`{word}`" for word, count in top_unigrams],
            "Count": [count for word, count in top_unigrams],
            "Probability": [f"{(count / total_tokens):.6f}" for word, count in top_unigrams]
        }
        st.dataframe(pd.DataFrame(unigram_data), use_container_width=True)


    # --- Bigram Probability ---
    st.subheader("Calculate Bigram Probability")
    st.markdown("Calculate $P(w_2 | w_1)$ ")


    def get_bigram_probability(w1, w2, unigrams, bigrams, V):
        """Calculates P(w2|w1) with Add-1 smoothing."""
        w1, w2 = w1.lower(), w2.lower()
        numerator = bigrams.get((w1, w2), 0) + 1
        denominator = unigrams.get(w1, 0) + V
        return numerator / denominator


    col1, col2 = st.columns(2)
    with col1:
        w1_input = st.text_input("Enter first word ($w_1$):", value="like")
    with col2:
        w2_input = st.text_input("Enter second word ($w_2$):", value="green")


    if w1_input and w2_input:
        prob = get_bigram_probability(w1_input, w2_input, unigram_counts, bigram_counts, vocab_size)
        st.info(f"**$P({w2_input.lower()} | {w1_input.lower()})$** = **{prob:.8f}**")


    # --- Perplexity Calculation ---
    st.subheader("Calculate Sentence Perplexity")
    st.markdown("Perplexity measures how well a model predicts a sentence. Lower is better.")


    def calculate_perplexity(sentence, unigrams, bigrams, V):
        """Calculates the perplexity of a sentence."""
        words = ['<s>'] + [word.lower() for word in nltk.word_tokenize(sentence)] + ['</s>']
        log_prob_sum = 0.0
        
        for i in range(1, len(words)):
            w1, w2 = words[i-1], words[i]
            prob = get_bigram_probability(w1, w2, unigrams, bigrams, V)
            log_prob_sum += math.log2(prob)
            
        perplexity = math.pow(2, -log_prob_sum / len(words))
        return perplexity


    test_sentence = st.text_input("Enter a test sentence:", value="The fulton county grand jury said friday.")
    if test_sentence:
        perplexity_score = calculate_perplexity(test_sentence, unigram_counts, bigram_counts, vocab_size)
        st.info(f"The perplexity of the sentence is: **{perplexity_score:.4f}**")



# ==============================================================================
# TAB 2: HMM POS TAGGER
# ==============================================================================
with tab2:
    st.header("Part-of-Speech Tagging with spaCy")
    st.markdown("This tab uses **spaCy** for high-accuracy POS tagging and shows only the token and its POS tag.")

    # Text input
    pos_input_sentence = st.text_area(
        "Enter a sentence to tag:",
        value="The quick brown fox jumps over the lazy dog.",
        height=100,
    )

    if pos_input_sentence.strip():
        doc = nlp(pos_input_sentence)

        tag_data = {
            "Token": [token.text for token in doc],
            "POS Tag": [token.pos_ for token in doc]
        }

        st.subheader("Tagged Output (spaCy - Simplified)")
        st.dataframe(pd.DataFrame(tag_data), use_container_width=True)



# ==============================================================================
# TAB 3: NAMED ENTITY RECOGNITION (NER)
# ==============================================================================
with tab3:
    st.header("Named Entity Recognition (NER) with spaCy")
    st.markdown("This tool uses spaCy's pre-trained model to identify named entities like persons, organizations, and locations in text. It displays the results using the **IOB (Inside, Outside, Beginning)** format.")
    
    ner_text_input = st.text_area(
        "Enter text for NER:",
        value="John lives in New York and works for the United Nations.",
        height=150
    )

    if ner_text_input:
        processed_text = " " + ner_text_input.strip() if not ner_text_input.startswith(" ") else ner_text_input
        doc = nlp(processed_text)
        
        # --- Display IOB tags ---
        st.subheader("Tokens and IOB Tags")
        iob_data = []
        for token in doc:
            if token.text.strip() == "":
                continue
            if token.ent_iob_ == "O":
                iob_tag = "O"
            else:
                iob_tag = f"{token.ent_iob_}-{token.ent_type_}"
            iob_data.append({"Token": token.text, "POS Tag": token.pos_, "IOB Tag": iob_tag})
            
        st.dataframe(pd.DataFrame(iob_data), use_container_width=True)
        
        # --- Display detected entities ---
        st.subheader("Detected Entities")
        entities = doc.ents
        if entities:
            for ent in entities:
                entity_text = ent.text.strip()
                if entity_text:
                    st.markdown(f"- **{entity_text}** (`{ent.label_}`)")
        else:
            st.warning("No named entities were found in the provided text.")
            st.info("💡 **Tip**: Try using sentences with proper nouns like names of people, places, or organizations.")


# --- Sidebar Info ---
st.sidebar.title("About")
st.sidebar.info(
    "This application demonstrates three fundamental NLP tasks. "
    "It uses the **NLTK** and **spaCy** libraries, with models trained on the **Brown** and **Treebank** corpora."
)
st.sidebar.title("How to Use")
st.sidebar.markdown(
    """
    1.  **N-gram Model**: Explore word probabilities and see how 'surprising' a sentence is to the model via its perplexity score.
    2.  **HMM POS Tagger**: See how a probabilistic model assigns parts of speech (like Noun, Verb, Adjective) to words in a sentence.
    3.  **NER**: Find real-world objects like people, places, and organizations mentioned in your text.
    """
)
