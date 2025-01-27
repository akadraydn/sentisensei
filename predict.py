import tensorflow as tf
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

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

class SentimentAnalyzer:
    def __init__(self, model_path='models/deep_model.h5', vectorizer_path='models/vectorizer.pkl'):
        # Modeli yükle
        self.model = tf.keras.models.load_model(model_path)
        # Vektörizeri yükle
        self.vectorizer = joblib.load(vectorizer_path)
    
    def predict(self, text):
        # Metni önişle
        processed_text = preprocess_text(text)
        
        # Metni vektöre dönüştür
        text_vector = self.vectorizer.transform([processed_text])
        
        # Tahmin yap
        prediction = self.model.predict(text_vector.toarray())[0][0]
        
        # Sonucu yorumla
        sentiment = "pozitif" if prediction > 0.5 else "negatif"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'processed_text': processed_text
        }

# Kullanım örneği
if __name__ == "__main__":
    # Analiz yapacak sınıfı oluştur
    analyzer = SentimentAnalyzer()
    
    # Örnek metinler
    sample_texts = [
        "الذكاء الاصطناعي يغير مستقبل التكنولوجيا",  # "Yapay zeka teknolojinin geleceğini değiştiriyor"
        "هذا المنتج سيء للغاية",  # "Bu ürün çok kötü"
        "أحب هذا الكتاب كثيرا"  # "Bu kitabı çok seviyorum"
    ]
    
    # Her metin için tahmin yap
    for text in sample_texts:
        result = analyzer.predict(text)
        print(f"\nMetin: {text}")
        print(f"Duygu: {result['sentiment']}")
        print(f"Güven: %{result['confidence']*100:.2f}")
        print(f"İşlenmiş metin: {result['processed_text']}")
