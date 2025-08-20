# app.py
import re, math, itertools, collections, json, textwrap
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tabulate import tabulate

# --------------------------- NLTK helpers
porter, wnl = PorterStemmer(), WordNetLemmatizer()
TOKEN = re.compile(r"\b\w+\b")

def norm_tok(text):
    return [w.lower() for w in word_tokenize(text) if w.isalnum()]

# --------------------------- Levenshtein helpers
def levenshtein_matrix(a: str, b: str):
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp

def backtrack(a, b, dp):
    i, j, ops = len(a), len(b), []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i, j] == dp[i - 1, j - 1]:
            ops.append(("Match", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("Delete", a[i - 1], ""))
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ops.append(("Insert", "", b[j - 1]))
            j -= 1
        else:
            ops.append(("Substitute", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
    return ops[::-1]

# --------------------------- Streamlit layout
st.set_page_config(page_title="Lab - 4", layout="wide")
st.title("📚 Revision and Word Sense Disambiguation (WSD)")
tabs = st.tabs(["🔍 Q1 Positional Index",
                "🗃️ Q2 Word Matrix",
                "⚙️ Q3 Tokenize & Edit",
                "🧮 Q4 Edit-Distance Trace",
                "🧬 Q5 HMM POS",
                "🧠 Q6 Word Senses & Lesk"])

# ----------------------------------------------------------- Q1
with tabs[0]:
    st.header("🔍 Question 1 – Positional Index")
    docs_preset = {
        "Doc 1": "I am a student, and I currently take MDS472C. I was a student in MDS331 last trimester.",
        "Doc 2": "I was a student. I have taken MDS472C."
    }
    docs = {}
    for k in docs_preset:
        docs[k] = st.text_area(k, value=docs_preset[k], height=80, key=f"q1_{k}")
    # Build index
    index = collections.defaultdict(lambda: collections.defaultdict(list))
    for doc_id, text in docs.items():
        for m in TOKEN.finditer(text.lower()):
            index[m.group()][doc_id].append(m.start())
    st.subheader("Pre-defined words")
    for w in ["student", "mds472c"]:
        st.json({w: dict(index[w])} if w in index else {w: "not found"})
    words = st.text_input("Query any word(s) (space separated):", key="q1_query")
    if words:
        queried = set(words.lower().split())
        st.json({w: dict(index[w]) if w in index else "not found" for w in queried})

# ----------------------------------------------------------- Q2
with tabs[1]:
    st.header("🗃️ Question 2 – Word / Document Matrix")
    vocab = sorted({w for t in docs.values() for w in norm_tok(t)})
    matrix = [[int(w in norm_tok(t)) for w in vocab] for t in docs.values()]
    df = pd.DataFrame(matrix, columns=vocab, index=list(docs.keys()))
    st.dataframe(df.style.highlight_max(axis=1, color="#c8e6c9"))

# ----------------------------------------------------------- Q3
with tabs[2]:
    st.header("⚙️ Question 3 – Tokenize, Pre-process, Frequency & Edit Distance")
    tokens = {k: norm_tok(v) for k, v in docs.items()}
    st.write("Tokens"); st.json(tokens)
    # stemming & lemmatization
    stem = {k: [porter.stem(w) for w in lst] for k, lst in tokens.items()}
    lem = {k: [wnl.lemmatize(w) for w in lst] for k, lst in tokens.items()}
    st.write("Stemmed"); st.json(stem)
    st.write("Lemmatized"); st.json(lem)
    freq = collections.Counter(itertools.chain.from_iterable(lem.values()))
    st.write("Frequency (lemmatized, sorted)")
    st.json(freq.most_common())
    corpus = list(freq)
    c1, c2 = st.columns(2)
    w1 = c1.selectbox("Word 1", corpus, key="q3_w1")
    w2 = c2.selectbox("Word 2", corpus, index=1, key="q3_w2")
    if w1 and w2:
        dp = levenshtein_matrix(w1, w2)
        st.write(f"Levenshtein distance = {dp[-1, -1]}")
        st.dataframe(pd.DataFrame(dp, index=[f"{i}_{c}" for i, c in enumerate(" " + w1)],
                                  columns=[f"{j}_{c}" for j, c in enumerate(" " + w2)]))

# ----------------------------------------------------------- Q4
with tabs[3]:
    st.header("🧮 Question 4 – Full Edit-Distance Trace")
    a, b = "characterization", "categorization"
    st.write("**Words**", dict(A=a, B=b))
    dp = levenshtein_matrix(a, b)
    ops = backtrack(a, b, dp)
    st.write("DP matrix")
    # Make safe unique labels
    row_lab = [f"{i}_{ch}" for i, ch in enumerate(" " + a)]
    col_lab = [f"{j}_{ch}" for j, ch in enumerate(" " + b)]
    st.dataframe(pd.DataFrame(dp, index=row_lab, columns=col_lab))
    st.write("Operations"); st.dataframe(pd.DataFrame(ops, columns=["Operation", "From", "To"]))
    # Alignment
    al_a, al_b, al_sym = [], [], []
    for op, x, y in ops:
        if op == "Match":
            al_a.append(x); al_b.append(y); al_sym.append("*")
        elif op == "Delete":
            al_a.append(x); al_b.append("-"); al_sym.append("D")
        elif op == "Insert":
            al_a.append("-"); al_b.append(y); al_sym.append("I")
        else:
            al_a.append(x); al_b.append(y); al_sym.append("S")
    st.text("Aligned Words")
    st.text("Word A: " + "".join(al_a))
    st.text("Word B: " + "".join(al_b))
    st.text("Ops   : " + "".join(al_sym))
    st.json({
        "Total min distance": int(dp[-1, -1]),
        "Insertions": sum(1 for op, _, _ in ops if op == "Insert"),
        "Deletions": sum(1 for op, _, _ in ops if op == "Delete"),
        "Substitutions": sum(1 for op, _, _ in ops if op == "Substitute"),
        "Matches": sum(1 for op, _, _ in ops if op == "Match")
    })

# ----------------------------------------------------------- Q5
with tabs[4]:
    st.header("🧬 Question 5 – Simple HMM POS Tagger")
    training_raw = ["The cat chased the rat",
                    "A rat can run",
                    "The dog can chase the cat"]
    st.write("Training corpus"); st.json(training_raw)
    test_sentence = st.text_input("Test sentence:", "The rat can chase the cat")
    # Tagging
    train_tok = [norm_tok(s) for s in training_raw]
    train_pos = [nltk.pos_tag(sent) for sent in train_tok]
    vocab = set(itertools.chain.from_iterable(train_tok))
    tags = sorted({t for sent in train_pos for _, t in sent})
    # counts
    tag_ct = collections.Counter(t for sent in train_pos for _, t in sent)
    em = collections.defaultdict(lambda: collections.defaultdict(int))
    tr = collections.defaultdict(lambda: collections.defaultdict(int))
    for sent in train_pos:
        for w, t in sent:
            em[t][w] += 1
    for sent in train_pos:
        prev = "<s>"
        for _, t in sent:
            tr[prev][t] += 1
            prev = t
        tr[prev]["</s>"] += 1
    # probabilities
    def norm(dd):
        return {k: {kk: vv / sum(v.values()) for kk, vv in v.items()} for k, v in dd.items()}
    transition = norm(tr)
    emission = norm(em)
    st.write("Transition probabilities")
    st.dataframe(pd.DataFrame(transition).fillna(0).T)
    st.write("Emission probabilities (sample)")
    st.dataframe(pd.DataFrame(emission).fillna(0).T)
    # Viterbi
    obs = norm_tok(test_sentence)
    T = len(obs)
    K = len(tags)
    V = np.zeros((K, T))
    ptr = np.zeros((K, T), dtype=int)
    for s, tag in enumerate(tags):
        V[s, 0] = np.log(transition.get("<s>", {}).get(tag, 1e-8)) + \
                  np.log(emission.get(tag, {}).get(obs[0], 1e-8))
    for t in range(1, T):
        for s, tag in enumerate(tags):
            probs = [V[sp, t - 1] +
                     np.log(transition.get(tags[sp], {}).get(tag, 1e-8)) +
                     np.log(emission.get(tag, {}).get(obs[t], 1e-8))
                     for sp in range(K)]
            V[s, t] = max(probs)
            ptr[s, t] = np.argmax(probs)
    last = np.argmax([V[s, T - 1] + np.log(transition.get(tags[s], {}).get("</s>", 1e-8)) for s in range(K)])
    path = [last]
    for t in range(T - 1, 0, -1):
        last = ptr[last, t]
        path.append(last)
    path = [tags[i] for i in reversed(path)]
    st.write("Viterbi tagging result")
    st.json(list(zip(obs, path)))
# ----------------------------------------------------------- Q6
with tabs[5]:
    st.header("🧠 Question 6 – Word Senses (WordNet) & Lesk WSD")
    st.write("Collect a small corpus, count senses for open-class words, and disambiguate using the Lesk algorithm.")

    from nltk.corpus import wordnet as wn, stopwords
    from nltk.tokenize import word_tokenize

    # Ensure stopwords fallback
    try:
        STOP = set(stopwords.words('english'))
    except:
        STOP = set("a an the and or but if while with to of in on at by for from is are was were be been being it this that as into over under across".split())

    default_corpus = "\n".join([
        "The central bank raised interest rates on Tuesday, citing persistent inflation.",
        "Wildfires continue to burn across the region as temperatures soar.",
        "The tech giant unveiled its latest smartphone at a press event in San Francisco.",
        "Scientists discovered a new exoplanet orbiting a nearby star.",
        "The prime minister announced new measures to boost the economy.",
    ])
    user_corpus = st.text_area("Paste one sentence per line:", value=default_corpus, height=160)
    sentences = [ln.strip() for ln in user_corpus.splitlines() if ln.strip()]

    def is_open_class(tag):
        return tag.startswith(("N","V","J","R"))

    def tag_to_wn(tag):
        if tag.startswith('N'): return wn.NOUN
        if tag.startswith('V'): return wn.VERB
        if tag.startswith('J'): return wn.ADJ
        if tag.startswith('R'): return wn.ADV
        return None

    def clean_tokens(tokens):
        return [t.lower() for t in tokens if t.isalpha() and t.lower() not in STOP]

    # --- THIS IS THE CORRECTED FUNCTION ---
    def synset_signature(ss):
        """
        Creates an enriched signature for a synset by including definitions and
        lemma names from the synset itself and its related senses (hypernyms/hyponyms).
        """
        # Start with the synset's own definition and examples
        sig_tokens = word_tokenize(ss.definition())
        for ex in ss.examples():
            sig_tokens += word_tokenize(ex)

        # Get related synsets (hypernyms and hyponyms are good general ones)
        related_senses = ss.hypernyms() + ss.hyponyms()

        # For each related sense, add its lemma names and its definition to the signature
        for sense in related_senses:
            # .lemma_names() gives a list like ['financial_institution', 'depository']
            # We need to split these into individual words
            lemmas = [lemma.replace('_', ' ') for lemma in sense.lemma_names()]
            sig_tokens.extend(' '.join(lemmas).split()) # Add the words from the lemma names
            sig_tokens.extend(word_tokenize(sense.definition())) # Add words from the definition

        # Clean the final list of tokens
        return set(clean_tokens(sig_tokens))


    def simplified_lesk(context_tokens, target, wn_pos=None):
        candidates = wn.synsets(target, pos=wn_pos) if wn_pos else wn.synsets(target)
        if not candidates: return None, 0, 0
        context = set(clean_tokens(context_tokens)) - {target.lower()}
        best, best_overlap = None, -1
        for ss in candidates:
            # Use the new, more powerful signature function
            signature = synset_signature(ss)
            overlap = len(context & signature)
            if overlap > best_overlap:
                best, best_overlap = ss, overlap
        return best, best_overlap, len(candidates)

    # Process sentences
    for i, sent in enumerate(sentences, 1):
        st.subheader(f"Sentence {i}")
        st.write(sent)
        tokens = word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        rows = []
        for w, t in pos_tags:
            if not is_open_class(t): continue
            wn_pos = tag_to_wn(t)
            # This line is redundant but harmless, keeping it as is.
            senses = wn.synsets(w, pos=wn_pos) if wn_pos else wn.synsets(w)
            best, overlap, n_senses = simplified_lesk(tokens, w, wn_pos)
            rows.append({
                "word": w,
                "POS": t,
                "# senses": n_senses,
                "chosen_sense": best.name() if best else None,
                "definition": best.definition() if best else None,
                "overlap": overlap,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No open-class words in this sentence.")
