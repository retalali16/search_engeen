import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
import time
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from collections import defaultdict, Counter
import math
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {e}")
    raise

app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "*"}})

# Load CSV files
try:
    raw_news = pd.read_csv('bbc_scraped_news.csv')
    preprocessed_news = pd.read_csv('preprocessed_news.csv')
    logging.info("CSV files loaded successfully")
    logging.debug(f"raw_news columns: {raw_news.columns}")
    logging.debug(f"preprocessed_news columns: {preprocessed_news.columns}")
except FileNotFoundError as e:
    logging.error(f"CSV file not found: {e}")
    raise
except pd.errors.EmptyDataError as e:
    logging.error(f"CSV file is empty: {e}")
    raise
except Exception as e:
    logging.error(f"Error loading CSV files: {e}")
    raise

# Validate CSV columns
required_raw_columns = ['Text', 'URL']
required_preprocessed_columns = ['URL', 'Processed_Text']
if not all(col in raw_news.columns for col in required_raw_columns):
    logging.error(f"raw_news.csv missing required columns: {required_raw_columns}")
    raise ValueError("raw_news.csv missing required columns")
if not all(col in preprocessed_news.columns for col in required_preprocessed_columns):
    logging.error(f"preprocessed_news.csv missing required columns: {required_preprocessed_columns}")
    raise ValueError("preprocessed_news.csv missing required columns")

# Extract titles from Text
raw_news['title'] = raw_news['Text'].apply(
    lambda x: sent_tokenize(str(x))[0] if sent_tokenize(str(x)) else 'No Title'
)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Initialize SentenceTransformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("SentenceTransformer model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer model: {e}")
    raise

def preprocess(text):
    """Preprocess text for search."""
    try:
        tokens = word_tokenize(str(text).lower())
        return [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    except Exception as e:
        logging.error(f"Error in preprocessing text: {e}")
        return []

def preprocess_for_expansion(text):
    """Preprocess text for query expansion."""
    return str(text).lower().split()

def expand_query_with_embeddings(query, corpus, top_k_terms=3, top_k_similar=2, relevance_feedback=True):
    """Expand query using embeddings."""
    try:
        query_tokens = preprocess_for_expansion(query)
        query_vector = model.encode([query])[0]
        corpus_tokens = [preprocess_for_expansion(doc) for doc in corpus]
        corpus_vectors = model.encode(corpus)
        similarities = cosine_similarity([query_vector], corpus_vectors)[0]

        top_docs_indices = similarities.argsort()[-5:][::-1]
        top_docs = [(i, similarities[i]) for i in top_docs_indices]
        candidate_terms = []
        if relevance_feedback:
            for doc_id, _ in top_docs:
                candidate_terms.extend(corpus_tokens[doc_id])
        term_counts = Counter(candidate_terms)
        frequent_terms = [term for term, _ in term_counts.most_common(top_k_terms)]

        all_terms = list(set(candidate_terms))
        term_vectors = model.encode(all_terms)
        similarities = cosine_similarity([query_vector], term_vectors)[0]
        similar_term_indices = similarities.argsort()[-top_k_similar:][::-1]
        similar_terms = [all_terms[i] for i in similar_term_indices]

        expanded_query = list(set(query_tokens + frequent_terms + similar_terms))
        return ' '.join(expanded_query), top_docs
    except Exception as e:
        logging.error(f"Error in query expansion: {e}")
        return query, []

class TFIDF:
    def __init__(self, docs):
        self.docs = [preprocess(doc) for doc in docs]
        self.vocab = set(word for doc in self.docs for word in doc)
        self.doc_count = len(docs)
        self.idf = self.compute_idf()
        self.doc_vectors = [self.compute_tf(doc) for doc in self.docs]
        logging.debug(f"TFIDF initialized with {self.doc_count} documents")

    def compute_tf(self, doc):
        tf = defaultdict(float)
        for word in doc:
            tf[word] += 1.0
        doc_len = len(doc) or 1
        for word in tf:
            tf[word] /= doc_len
        return tf

    def compute_idf(self):
        idf = defaultdict(float)
        for word in self.vocab:
            doc_freq = sum(1 for doc in self.docs if word in doc)
            idf[word] = math.log((self.doc_count + 1) / (1 + doc_freq)) + 1
        return idf

    def search(self, query):
        try:
            query_tokens = preprocess(query)
            query_tf = self.compute_tf(query_tokens)
            scores = []
            for i, doc_vector in enumerate(self.doc_vectors):
                score = sum(query_tf[word] * doc_vector.get(word, 0) * self.idf.get(word, 0) for word in query_tokens)
                scores.append((i, score))
            return sorted(scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logging.error(f"Error in TFIDF search: {e}")
            return []

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [preprocess(doc) for doc in docs]
        self.doc_count = len(self.docs)
        self.avgdl = sum(len(doc) for doc in self.docs) / (self.doc_count or 1)
        self.k1 = k1
        self.b = b
        self.idf = self.compute_idf()
        self.doc_freqs = [defaultdict(int) for _ in range(self.doc_count)]
        for i, doc in enumerate(self.docs):
            for word in doc:
                self.doc_freqs[i][word] += 1
        logging.debug(f"BM25 initialized with {self.doc_count} documents")

    def compute_idf(self):
        idf = defaultdict(float)
        vocab = set(word for doc in self.docs for word in doc)
        for word in vocab:
            doc_freq = sum(1 for doc in self.docs if word in doc)
            idf[word] = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return idf

    def search(self, query):
        try:
            query_tokens = preprocess(query)
            scores = []
            for i, doc in enumerate(self.docs):
                score = 0
                doc_len = len(doc) or 1
                for word in query_tokens:
                    if word in self.doc_freqs[i]:
                        tf = self.doc_freqs[i][word]
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                        score += self.idf.get(word, 0) * (numerator / denominator)
                scores.append((i, score))
            return sorted(scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logging.error(f"Error in BM25 search: {e}")
            return []

# Initialize search models
try:
    tfidf = TFIDF(preprocessed_news['Processed_Text'].tolist())
    bm25 = BM25(preprocessed_news['Processed_Text'].tolist())
except Exception as e:
    logging.error(f"Error initializing search models: {e}")
    raise

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering template: {e}")
        return jsonify({'error': 'Failed to load page'}), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        if not data:
            logging.warning("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400

        query = data.get('query', '').strip()
        algorithm = data.get('algorithm', 'tfidf').lower()
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)
        expand_query = data.get('expand_query', False)

        # Validate inputs
        if not query:
            logging.warning("Empty query received")
            return jsonify({'error': 'Query is required'}), 400
        if algorithm not in ['tfidf', 'bm25']:
            logging.warning(f"Invalid algorithm: {algorithm}")
            return jsonify({'error': 'Invalid algorithm'}), 400
        if not isinstance(page, int) or page < 1:
            page = 1
        if not isinstance(per_page, int) or per_page < 1 or per_page > 100:
            per_page = 10

        logging.debug(f"Search request: query={query}, algorithm={algorithm}, page={page}, per_page={per_page}, expand_query={expand_query}")

        start_time = time.time()

        # Expand query if requested
        final_query = query
        top_docs = []
        if expand_query:
            final_query, top_docs = expand_query_with_embeddings(
                query, preprocessed_news['Processed_Text'].tolist(), top_k_terms=3, top_k_similar=2
            )
            logging.debug(f"Expanded query: {final_query}")

        # Perform search
        results = tfidf.search(final_query) if algorithm == 'tfidf' else bm25.search(final_query)

        search_duration = round((time.time() - start_time) * 1000, 2)

        # Paginate results
        start = (page - 1) * per_page
        end = start + per_page
        paginated_results = results[start:end]

        response = []
        for doc_id, score in paginated_results:
            try:
                article_url = preprocessed_news.iloc[doc_id]['URL']
                article = raw_news[raw_news['URL'] == article_url]
                if article.empty:
                    logging.warning(f"No matching article found for URL: {article_url}")
                    continue
                article = article.iloc[0]
                content = str(article['Text'])[:400] + '...' if len(str(article['Text'])) > 400 else str(article['Text'])
                response.append({
                    'doc_id': doc_id,
                    'score': round(score, 4),
                    'title': article['title'],
                    'content': content,
                    'url': article['URL']
                })
            except Exception as e:
                logging.error(f"Error processing document {doc_id}: {e}")
                continue

        logging.debug(f"Search completed: {len(response)} results returned, total={len(results)}")

        return jsonify({
            'results': response,
            'total': len(results),
            'page': page,
            'per_page': per_page,
            'time_ms': search_duration,
            'expanded_query': final_query if expand_query else None
        })
    except Exception as e:
        logging.error(f"Search endpoint error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)