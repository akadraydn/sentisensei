from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Flask uygulamasını oluştur
app = Flask(__name__)
CORS(app)

# NLTK verilerini indir
nltk.download('stopwords')

# Önişleme fonksiyonları
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_stopwords_arabic(text):
    stop_words = set(stopwords.words('arabic'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

stemmer = ISRIStemmer()
def stem_arabic_text(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def preprocess_text(text):
    # Metni küçük harfe çevir
    text = str(text).lower()
    
    # Emojileri kaldır
    text = remove_emoji(text)
    
    # Noktalama işaretlerini kaldır
    text = remove_punctuation(text)
    
    # Sayıları kaldır
    text = re.sub(r'\d+', '', text)
    
    # URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Özel karakterleri kaldır
    text = re.sub(r'[^\w\s]', '', text)
    
    # Arapça stopwords'leri kaldır
    text = remove_stopwords_arabic(text)
    
    # Fazla boşlukları temizle
    text = remove_extra_spaces(text)
    
    # Stemming uygula
    text = stem_arabic_text(text)
    
    # En az 2 karakterli kelimeleri tut
    words = text.split()
    text = ' '.join([word for word in words if len(word) > 1])
    
    # Tekrarlanan kelimeleri azalt
    text = re.sub(r'(\b\w+\b)(\s+\1\b)+', r'\1 \1', text)
    
    return text

class CombinedAnalyzer:
    def __init__(self,
                 sentiment_model_path='models/deep_model.h5',
                 sentiment_vectorizer_path='models/vectorizer.pkl',
                 category_model_path='models/stacking_model.pkl',
                 category_vectorizer_path='models/tfidf_vectorizer.pkl'):
        
        # Duygu analizi modelini yükle
        self.sentiment_model = tf.keras.models.load_model(sentiment_model_path)
        self.sentiment_vectorizer = joblib.load(sentiment_vectorizer_path)
        
        # Kategori sınıflandırma modelini yükle
        self.category_model = joblib.load(category_model_path)
        self.category_vectorizer = joblib.load(category_vectorizer_path)
        
        # Kategori etiketleri
        self.categories = ['Kültür', 'Finans', 'Tıp', 'Siyaset', 'Din', 'Spor', 'Teknoloji']
    
    def analyze(self, text):
        # Metni önişle
        processed_text = preprocess_text(text)
        
        # Duygu analizi
        sentiment_vector = self.sentiment_vectorizer.transform([processed_text])
        sentiment_pred = self.sentiment_model.predict(sentiment_vector.toarray())[0][0]
        sentiment = "olumlu" if sentiment_pred > 0.5 else "olumsuz"
        sentiment_conf = sentiment_pred if sentiment_pred > 0.5 else 1 - sentiment_pred
        
        # Kategori sınıflandırma
        category_vector = self.category_vectorizer.transform([processed_text])
        category_pred = self.category_model.predict_proba(category_vector)[0]
        category_idx = category_pred.argmax()
        category = self.categories[category_idx]
        category_conf = category_pred[category_idx]
        
        # Sonucu formatla
        result = f"bu metin {category.lower()} konusunda {sentiment} bir içeriğe sahip"
        
        return {
            'result': result,
            'sentiment': sentiment,
            'sentiment_confidence': float(sentiment_conf),
            'category': category,
            'category_confidence': float(category_conf),
            'processed_text': processed_text
        }

# Global analyzer nesnesi
analyzer = CombinedAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Metin boş olamaz'}), 400
            
        result = analyzer.analyze(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 