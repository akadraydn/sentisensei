import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from nltk.stem.isri import ISRIStemmer
import string
import joblib
import random

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

def load_dataset(base_path, max_samples_per_category=None):
    """Veri setini yükle ve hazırla"""
    data = []
    categories = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            files = os.listdir(category_path)
            if max_samples_per_category:
                files = files[:max_samples_per_category]
            
            for filename in files:
                with open(os.path.join(category_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append({
                        'text': text,
                        'category': category
                    })
    
    df = pd.DataFrame(data)
    print(f"Toplam örnek sayısı: {len(df)}")
    print("\nKategori dağılımı:")
    print(df['category'].value_counts())
    
    return df

def create_deep_model(vocab_size, max_length, num_classes):
    """Derin öğrenme modelini oluştur"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Eğitim geçmişini görselleştir"""
    plt.figure(figsize=(12, 4))
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Validasyon Doğruluğu')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    
    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Validasyon Kaybı')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Karmaşıklık matrisini görselleştir"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()

def augment_arabic_text(text, label):
    """Arapça metin veri artırma fonksiyonu"""
    augmented_texts = [text]  # Orijinal metin
    augmented_labels = [label]
    
    words = text.split()
    if len(words) < 4:  # Çok kısa metinleri atla
        return augmented_texts, augmented_labels
    
    # 1. Kelime Sırası Değiştirme
    if len(words) > 5:
        shuffled_words = words.copy()
        random.shuffle(shuffled_words)
        augmented_texts.append(' '.join(shuffled_words))
        augmented_labels.append(label)
    
    # 2. Rastgele Kelime Silme
    if len(words) > 8:
        num_words_to_keep = int(len(words) * 0.8)
        kept_words = random.sample(words, num_words_to_keep)
        augmented_texts.append(' '.join(kept_words))
        augmented_labels.append(label)
    
    # 3. Rastgele Kelime Tekrarı
    if len(words) > 3:
        words_to_repeat = random.sample(words, min(3, len(words)))
        repeated_text = text + ' ' + ' '.join(words_to_repeat)
        augmented_texts.append(repeated_text)
        augmented_labels.append(label)
    
    # 4. Kelime Alt Kümesi Seçme
    if len(words) > 10:
        start_idx = random.randint(0, len(words) // 2)
        subset_length = random.randint(len(words) // 2, len(words))
        subset_words = words[start_idx:start_idx + subset_length]
        augmented_texts.append(' '.join(subset_words))
        augmented_labels.append(label)
    
    return augmented_texts, augmented_labels

def train_classifier():
    """Ana eğitim fonksiyonu"""
    # Veri setini yükle
    print("Veri seti yükleniyor...")
    df = load_dataset('/Users/akadraydn/Desktop/SentiSensei/class-dataset', max_samples_per_category=1000)
    
    # Metin ön işleme
    print("Metinler ön işleniyor...")
    df['processed_text'] = df['text'].apply(preprocess_arabic_text)
    
    # Veri artırma uygula
    print("\nVeri artırma uygulanıyor...")
    augmented_texts = []
    augmented_labels = []
    
    # Her kategori için veri artırma sayısını belirle
    category_augmentation = {
        'Religion': 3,  # Religion kategorisi için 3 kat artırma
        'Medical': 2,   # Medical kategorisi için 2 kat artırma
        'Culture': 1,   # Diğer kategoriler için 1 kat artırma
        'Finance': 1,
        'Politics': 1,
        'Sports': 1,
        'Tech': 1
    }
    
    for text, label in zip(df['processed_text'], df['category']):
        num_augmentations = category_augmentation[label]
        for _ in range(num_augmentations):
            aug_texts, aug_labels = augment_arabic_text(text, label)
            augmented_texts.extend(aug_texts)
            augmented_labels.extend(aug_labels)
    
    # Artırılmış veriyi DataFrame'e ekle
    aug_df = pd.DataFrame({
        'processed_text': augmented_texts,
        'category': augmented_labels
    })
    
    # Orijinal veri ile birleştir
    df = pd.concat([df, aug_df], ignore_index=True)
    
    # Veri setini karıştır
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nVeri artırma sonrası kategori dağılımı:")
    print(df['category'].value_counts())
    
    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    y = to_categorical(y)
    
    # Tokenization ve padding
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['processed_text'])
    X = tokenizer.texts_to_sequences(df['processed_text'])
    max_length = 200
    X = pad_sequences(X, maxlen=max_length, padding='post')
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model oluştur ve eğit
    print("\nModel eğitiliyor...")
    model = create_deep_model(10000, max_length, len(label_encoder.classes_))
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Modeli kaydet
    print("\nModel kaydediliyor...")
    model.save('models/arabic_classifier.keras')
    joblib.dump(tokenizer, 'models/tokenizer.joblib')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    
    # Performans değerlendirmesi
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Sonuçları görselleştir
    plot_training_history(history)
    plot_confusion_matrix(y_test_classes, y_pred_classes, label_encoder.classes_)
    
    # Performans raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test_classes, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    return model, tokenizer, label_encoder

if __name__ == "__main__":
    model, tokenizer, label_encoder = train_classifier()