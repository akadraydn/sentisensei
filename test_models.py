import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.isri import ISRIStemmer
import re
import string
import nltk
from nltk.corpus import stopwords

# NLTK verilerini indir
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_arabic_text(text):
    """Arapça metin ön işleme fonksiyonu"""
    # Metni küçük harfe çevir
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
    
    # En az 2 karakterli kelimeleri tut
    words = text.split()
    text = ' '.join([word for word in words if len(word) > 1])
    
    return text

class SentimentAnalyzer:
    def __init__(self):
        """Duygu analizi modellerini yükle"""
        print("Duygu analizi modelleri yükleniyor...")
        self.vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        self.sgd_classifier = joblib.load('models/sgd_classifier.joblib')
        self.lr_classifier = joblib.load('models/logistic_regression.joblib')
        self.deep_model = tf.keras.models.load_model('models/best_deep_model.keras')
        self.ensemble_weights = np.load('models/ensemble_weights.npy')
    
    def predict(self, text):
        """Metni analiz et ve duygu tahminini döndür"""
        # Metin ön işleme
        processed_text = preprocess_arabic_text(text)
        
        # Vektörleştirme
        text_vector = self.vectorizer.transform([processed_text])
        
        # Her modelden tahmin al
        sgd_pred = self.sgd_classifier.predict_proba(text_vector)[0, 1]
        lr_pred = self.lr_classifier.predict_proba(text_vector)[0, 1]
        deep_pred = self.deep_model.predict(text_vector.toarray())[0, 0]
        
        # Ensemble tahmin
        ensemble_pred = (
            self.ensemble_weights[0] * sgd_pred +
            self.ensemble_weights[1] * lr_pred +
            self.ensemble_weights[2] * deep_pred
        )
        
        # Sonucu etiketle
        sentiment = 'positive' if ensemble_pred > 0.5 else 'negative'
        confidence = ensemble_pred if ensemble_pred > 0.5 else 1 - ensemble_pred
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'model_predictions': {
                'sgd': float(sgd_pred),
                'logistic_regression': float(lr_pred),
                'deep_learning': float(deep_pred),
                'ensemble': float(ensemble_pred)
            }
        }

class TextClassifier:
    def __init__(self):
        """Metin sınıflandırma modelini yükle"""
        print("Metin sınıflandırma modeli yükleniyor...")
        self.model = tf.keras.models.load_model('models/arabic_classifier.keras')
        self.tokenizer = joblib.load('models/tokenizer.joblib')
        self.label_encoder = joblib.load('models/label_encoder.joblib')
        self.max_length = 200
    
    def predict(self, text):
        """Metni sınıflandır ve kategori tahminini döndür"""
        # Metin ön işleme
        processed_text = preprocess_arabic_text(text)
        
        # Tokenization ve padding
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Tahmin
        predictions = self.model.predict(padded)
        
        # En yüksek olasılıklı 3 kategoriyi al
        top_3_indices = predictions[0].argsort()[-3:][::-1]
        top_3_categories = self.label_encoder.inverse_transform(top_3_indices)
        top_3_probabilities = predictions[0][top_3_indices]
        
        return {
            'top_category': top_3_categories[0],
            'confidence': float(top_3_probabilities[0]),
            'top_3_predictions': [
                {'category': cat, 'probability': float(prob)}
                for cat, prob in zip(top_3_categories, top_3_probabilities)
            ]
        }

def test_models():
    """Kullanıcı girişi ile modelleri test et"""
    # Modelleri yükle
    print("Modeller yükleniyor, lütfen bekleyin...")
    sentiment_analyzer = SentimentAnalyzer()
    text_classifier = TextClassifier()
    
    print("\nMetin analizi başladı!")
    print("Çıkmak için 'q' tuşuna basın")
    print("-" * 50)
    
    while True:
        print("\nAnaliz edilecek Arapça metni girin:")
        text = input("> ")
        
        if text.lower() == 'q':
            print("\nProgram sonlandırılıyor...")
            break
        
        if not text.strip():
            print("Lütfen bir metin girin!")
            continue
            
        print("\nMetin analiz ediliyor...")
        
        try:
            # Duygu analizi
            sentiment_result = sentiment_analyzer.predict(text)
            print("\nDuygu Analizi Sonucu:")
            print(f"Duygu: {sentiment_result['sentiment']}")
            print(f"Güven: {sentiment_result['confidence']:.2f}")
            
            # Kategori sınıflandırma
            category_result = text_classifier.predict(text)
            print("\nKategori Sınıflandırma Sonucu:")
            print(f"Ana Kategori: {category_result['top_category']}")
            print(f"Güven: {category_result['confidence']:.2f}")
            print("\nİlk 3 Tahmin:")
            for pred in category_result['top_3_predictions']:
                print(f"- {pred['category']}: {pred['probability']:.2f}")
            
        except Exception as e:
            print(f"\nHata oluştu: {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_models() 