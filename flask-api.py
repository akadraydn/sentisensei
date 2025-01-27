from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import joblib
import logging
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from nltk.stem.isri import ISRIStemmer
import re
import string
import nltk
from nltk.corpus import stopwords
import os
import gdown

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# CORS ayarları
CORS(app, resources={
    r"/*": {
        "origins": ["https://www.sentisensei.com", "http://www.sentisensei.com", "https://sentisensei.com", "http://sentisensei.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in ["https://www.sentisensei.com", "http://www.sentisensei.com", "https://sentisensei.com", "http://sentisensei.com"]:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '3600'
    return response

@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    response = make_response()
    origin = request.headers.get('Origin')
    if origin in ["https://www.sentisensei.com", "http://www.sentisensei.com", "https://sentisensei.com", "http://sentisensei.com"]:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '3600'
    return response

# NLTK verilerini indir
nltk.download('stopwords')
nltk.download('punkt')

# Model dosyalarını Google Drive'dan indir
def download_models():
    app.logger.info("Model dosyaları indiriliyor...")
    
    # Model dosyalarının Drive linkleri - ID'leri kullanarak
    model_urls = {
        'best_deep_model.keras': '1v9I4cuD-ZGmNg3GDH_6HcanljmlKlLol',
        'arabic_classifier.keras': '1dyyFnkONVUjnVlsb51gurjClylvX5HBg',
        'tfidf_vectorizer.joblib': '1YRAfHNFTS8v6eFGgx_q1C62HKqX0lYJa',
        'sgd_classifier.joblib': '1HlgGEJh80NHqtAmyoPckt7nIFQ33KaCd',
        'logistic_regression.joblib': '1Iur28SO0TLU-QNH7ZhIuNjwgwPinPQsT',
        'ensemble_weights.npy': '192h_9rCYtPEiFGTR1-8IQuJg2epN-tSB',
        'tokenizer.joblib': '1mPDn_4eSuCQJ6n26HtDzKE3Dt4gqP5P-',
        'label_encoder.joblib': '1m6FO98N_aBRX9mW6kK2Bx6RH0LTLANvZ'
    }
    
    # Models klasörünü oluştur
    if not os.path.exists('models'):
        os.makedirs('models')
        app.logger.info("Models klasörü oluşturuldu")
    
    # Modelleri indir
    for model_name, file_id in model_urls.items():
        model_path = f'models/{model_name}'
        if not os.path.exists(model_path):
            app.logger.info(f"{model_name} indiriliyor... (ID: {file_id})")
            try:
                url = f'https://drive.google.com/file/d/{file_id}/view'
                app.logger.debug(f"İndirme URL'i: {url}")
                
                success = gdown.download(url, model_path, quiet=False, fuzzy=True, use_cookies=False)
                
                if success:
                    app.logger.info(f"{model_name} başarıyla indirildi")
                    if os.path.exists(model_path):
                        file_size = os.path.getsize(model_path)
                        app.logger.info(f"{model_name} dosya boyutu: {file_size} bytes")
                    else:
                        app.logger.error(f"{model_name} indirildi ama dosya bulunamıyor!")
                else:
                    raise Exception("gdown indirme başarısız oldu")
                    
            except Exception as e:
                app.logger.error(f"{model_name} indirilirken hata oluştu: {str(e)}")
                app.logger.error(f"Hata detayları: {type(e).__name__}")
                raise Exception(f"Model dosyası indirilemedi: {model_name}, Hata: {str(e)}")
        else:
            app.logger.info(f"{model_name} zaten mevcut. Boyut: {os.path.getsize(model_path)} bytes")

# Modelleri indir
download_models()

# Model yükleme
app.logger.info("Modeller yükleniyor...")

try:
    # Duygu Analizi Modelleri
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    sgd_classifier = joblib.load('models/sgd_classifier.joblib')
    lr_classifier = joblib.load('models/logistic_regression.joblib')
    
    # Derin öğrenme modellerini yükle
    custom_objects = {}
    deep_model = tf.keras.models.load_model('models/best_deep_model.keras', 
                                          custom_objects=custom_objects,
                                          compile=False)
    deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ensemble_weights = np.load('models/ensemble_weights.npy')

    # Sınıflandırma Modeli
    classifier_model = tf.keras.models.load_model('models/arabic_classifier.keras',
                                               custom_objects=custom_objects,
                                               compile=False)
    classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    tokenizer = joblib.load('models/tokenizer.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')

    app.logger.info("Modeller başarıyla yüklendi")
except Exception as e:
    app.logger.error(f"Model yükleme hatası: {str(e)}")
    raise

def is_arabic_text(text):
    """Arapça metin kontrolü fonksiyonu"""
    if not text or not text.strip():
        return False
        
    # Arapça karakter pattern'i
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFE70-\uFEFF\u0660-\u0669]+')
    arabic_chars = len(re.findall(arabic_pattern, text))
    total_chars = len(text.strip())
    
    # Debug için log
    app.logger.debug(f"Metin: {text}")
    app.logger.debug(f"Toplam karakter: {total_chars}")
    app.logger.debug(f"Arapça karakter: {arabic_chars}")
    
    # Arapça karakter oranı %5'den az ise Arapça değil
    arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
    app.logger.debug(f"Arapça oran: {arabic_ratio}")
    
    return arabic_ratio >= 0.05

def preprocess_arabic_text(text):
    """Arapça metin ön işleme fonksiyonu"""
    text = str(text).lower()
    
    # Emojileri kaldır
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Noktalama işaretlerini kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Stop words'leri kaldır
    stop_words = set(stopwords.words('arabic'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    # Stemming uygula
    stemmer = ISRIStemmer()
    words = text.split()
    text = ' '.join([stemmer.stem(word) for word in words])
    
    # URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Sayıları kaldır
    text = re.sub(r'\d+', '', text)
    
    # Tekrarlanan kelimeleri azalt
    text = re.sub(r'(\b\w+\b)(\s+\1\b)+', r'\1', text)
    
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.debug("Tahmin isteği alındı")
        data = request.get_json()
        text = data.get('text', '')
        
        # Boş metin kontrolü
        if not text or not text.strip():
            return jsonify({
                'error': 'Metin boş olamaz.',
                'status': 'error'
            }), 400
        
        # Arapça metin kontrolü
        if not is_arabic_text(text):
            return jsonify({
                'error': 'Lütfen Arapça bir metin girin. Bu metin Arapça değil.',
                'status': 'error'
            }), 400
        
        # Metin ön işleme
        processed_text = preprocess_arabic_text(text)
        
        # Duygu Analizi
        text_vector = tfidf_vectorizer.transform([processed_text])
        
        # Her modelden tahmin al
        sgd_pred = sgd_classifier.predict_proba(text_vector)[0, 1]
        lr_pred = lr_classifier.predict_proba(text_vector)[0, 1]
        deep_pred = deep_model.predict(text_vector.toarray())[0, 0]
        
        # Ensemble tahmin
        ensemble_pred = (
            ensemble_weights[0] * sgd_pred +
            ensemble_weights[1] * lr_pred +
            ensemble_weights[2] * deep_pred
        )
        
        sentiment = 'positive' if ensemble_pred > 0.5 else 'negative'
        sentiment_confidence = ensemble_pred if ensemble_pred > 0.5 else 1 - ensemble_pred
        
        # Kategori Sınıflandırma
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=200, padding='post')
        category_pred = classifier_model.predict(padded)
        
        # En yüksek olasılıklı 3 kategoriyi al
        top_3_indices = category_pred[0].argsort()[-3:][::-1]
        top_3_categories = label_encoder.inverse_transform(top_3_indices)
        top_3_probabilities = category_pred[0][top_3_indices]
        
        response = {
            'sentiment': {
                'label': sentiment,
                'confidence': float(sentiment_confidence),
                'model_predictions': {
                    'sgd': float(sgd_pred),
                    'logistic_regression': float(lr_pred),
                    'deep_learning': float(deep_pred),
                    'ensemble': float(ensemble_pred)
                }
            },
            'category': {
                'top_category': str(top_3_categories[0]),
                'confidence': float(top_3_probabilities[0]),
                'top_3_predictions': [
                    {'category': str(cat), 'probability': float(prob)}
                    for cat, prob in zip(top_3_categories, top_3_probabilities)
                ]
            }
        }
        
        return jsonify(response)
            
    except Exception as e:
        app.logger.error(f"Hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Production ortamında port ve host ayarları
    port = int(os.environ.get('PORT', 10000))  # Render için varsayılan port
    app.logger.info(f"Uygulama {port} portunda başlatılıyor...")
    app.run(host='0.0.0.0', port=port, debug=False)
