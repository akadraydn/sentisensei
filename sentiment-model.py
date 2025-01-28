import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import re
import string
import random
import joblib

# NLTK verilerini indir
nltk.download('stopwords')

# Veri Yükleme ve Hazırlama
def load_dataset():
    """Veri setlerini yükle ve hazırla"""
    # Pozitif örnekleri yükle
    pos_train = pd.read_csv('neg-pos-dataset/train_pos.tsv', sep='\t', names=['text'])
    pos_test = pd.read_csv('neg-pos-dataset/test_pos.tsv', sep='\t', names=['text'])
    pos_data = pd.concat([pos_train, pos_test])
    pos_data['label'] = 1

    # Negatif örnekleri yükle
    neg_train = pd.read_csv('neg-pos-dataset/train_neg.tsv', sep='\t', names=['text'])
    neg_test = pd.read_csv('neg-pos-dataset/test_neg.tsv', sep='\t', names=['text'])
    neg_data = pd.concat([neg_train, neg_test])
    neg_data['label'] = 0

    # Veri setlerini birleştir
    data = pd.concat([pos_data, neg_data])
    
    # Çok kısa metinleri filtrele
    data = data[data['text'].str.len() > 10]
    
    # Her sınıftan eşit sayıda örnek al
    pos_samples = data[data['label'] == 1].sample(n=15000, random_state=42)
    neg_samples = data[data['label'] == 0].sample(n=15000, random_state=42)
    data = pd.concat([pos_samples, neg_samples])
    
    return data

# Metin Ön İşleme Fonksiyonları
def preprocess_arabic_text(text):
    """Arapça metinler için ön işleme fonksiyonu"""
    # Küçük harfe çevir
    text = str(text).lower()
    
    # Emojileri kaldır
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
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
    
    # Tekrarlanan kelimeleri azalt
    text = re.sub(r'(\b\w+\b)(\s+\1\b)+', r'\1 \1', text)
    
    # En az 2 karakterli kelimeleri tut
    words = text.split()
    text = ' '.join([word for word in words if len(word) > 1])
    
    # URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    return text

def augment_text(text, label):
    """Metin veri artırma fonksiyonu"""
    augmented_texts = [text]
    augmented_labels = [label]
    
    words = text.split()
    if len(words) < 4:
        return augmented_texts, augmented_labels
    
    # Kelime sırası değiştirme
    if len(words) > 5:
        shuffled_words = words.copy()
        random.shuffle(shuffled_words)
        augmented_texts.append(' '.join(shuffled_words))
        augmented_labels.append(label)
    
    # Rastgele kelime silme
    if len(words) > 8:
        num_words_to_keep = int(len(words) * 0.8)
        kept_words = random.sample(words, num_words_to_keep)
        augmented_texts.append(' '.join(kept_words))
        augmented_labels.append(label)
    
    return augmented_texts, augmented_labels

# Model Sınıfı
class ArabicSentimentModel:
    def __init__(self):
        """Model parametrelerini başlat"""
        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 5)
        )
        self.sgd_classifier = None
        self.lr_classifier = None
        self.deep_model = None
    
    def create_deep_model(self, input_dim):
        """Derin öğrenme modelini oluştur"""
        regularizer = keras.regularizers.l2(0.001)
        
        model = keras.Sequential([
            keras.layers.Dense(1024, input_dim=input_dim, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(512, activation='relu',
                                kernel_regularizer=regularizer),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=regularizer),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        return model
    
    def train(self, X, y):
        """3-fold cross validation ile model eğitimini gerçekleştir"""
        # 3-fold cross validation için KFold nesnesi oluştur
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/3 başlatılıyor...")
            
            # Eğitim ve validasyon verilerini ayır
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # TF-IDF dönüşümü
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_val_tfidf = self.vectorizer.transform(X_val)
            
            # SGD Classifier eğitimi
            self.sgd_classifier = SGDClassifier(
                loss='modified_huber',
                alpha=0.0001,
                penalty='elasticnet',
                max_iter=2000,
                random_state=42
            )
            self.sgd_classifier.fit(X_train_tfidf, y_train)
            
            # Logistic Regression eğitimi
            self.lr_classifier = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='saga',
                max_iter=2000
            )
            self.lr_classifier.fit(X_train_tfidf, y_train)
            
            # Derin öğrenme modeli eğitimi
            self.deep_model = self.create_deep_model(X_train_tfidf.shape[1])
            
            # Callback'leri tanımla
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-5,
                verbose=1
            )
            
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                f'best_model_fold_{fold}.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            
            # Modeli eğit
            history = self.deep_model.fit(
                X_train_tfidf.toarray(),
                y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_val_tfidf.toarray(), y_val),
                callbacks=[reduce_lr, model_checkpoint],
                verbose=1
            )
            
            # Ensemble tahminler
            sgd_pred = self.sgd_classifier.predict_proba(X_val_tfidf)[:, 1]
            lr_pred = self.lr_classifier.predict_proba(X_val_tfidf)[:, 1]
            deep_pred = self.deep_model.predict(X_val_tfidf.toarray()).ravel()
            
            # Ağırlıklı ensemble
            sgd_acc = self.sgd_classifier.score(X_val_tfidf, y_val)
            lr_acc = self.lr_classifier.score(X_val_tfidf, y_val)
            deep_acc = self.deep_model.evaluate(X_val_tfidf.toarray(), y_val)[1]
            
            weights = np.array([sgd_acc, lr_acc, deep_acc])
            weights = weights / weights.sum()
            
            ensemble_pred = (
                weights[0] * sgd_pred +
                weights[1] * lr_pred +
                weights[2] * deep_pred
            )
            
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            fold_accuracy = accuracy_score(y_val, ensemble_pred_binary)
            fold_accuracies.append(fold_accuracy)
            
            print(f"Fold {fold} Doğruluk: {fold_accuracy:.4f}")
            
            # Her bir sınıflandırıcının performansını yazdır
            print(f"SGD Doğruluk: {sgd_acc:.4f}")
            print(f"LR Doğruluk: {lr_acc:.4f}")
            print(f"Deep Learning Doğruluk: {deep_acc:.4f}")
        
        # Ortalama doğruluk değerini hesapla
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print(f"\nOrtalama Doğruluk: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
        
        return mean_accuracy

# Ana çalıştırma kodu
data = load_dataset()
X = data['text'].apply(preprocess_arabic_text)
y = data['label']

model = ArabicSentimentModel()

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF dönüşümü
X_train_tfidf = model.vectorizer.fit_transform(X_train)
X_test_tfidf = model.vectorizer.transform(X_test)

# TF-IDF vectorizer'ı kaydet
joblib.dump(model.vectorizer, 'models/tfidf_vectorizer.joblib')

# SGD Classifier eğitimi ve kaydı
model.sgd_classifier = SGDClassifier(
    loss='modified_huber',
    alpha=0.0001,
    penalty='elasticnet',
    max_iter=2000,
    random_state=42
)
model.sgd_classifier.fit(X_train_tfidf, y_train)
joblib.dump(model.sgd_classifier, 'models/sgd_classifier.joblib')

# Logistic Regression eğitimi ve kaydı
model.lr_classifier = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='saga',
    max_iter=2000
)
model.lr_classifier.fit(X_train_tfidf, y_train)
joblib.dump(model.lr_classifier, 'models/logistic_regression.joblib')

# Derin öğrenme modeli eğitimi ve kaydı
model.deep_model = model.create_deep_model(X_train_tfidf.shape[1])

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'models/best_deep_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Modeli eğit
history = model.deep_model.fit(
    X_train_tfidf.toarray(),
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_tfidf.toarray(), y_test),
    callbacks=[reduce_lr, model_checkpoint],
    verbose=1
)

# Eğitim geçmişi grafiklerini çiz
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

# Test seti üzerinde performans değerlendirmesi
sgd_pred = model.sgd_classifier.predict_proba(X_test_tfidf)[:, 1]
lr_pred = model.lr_classifier.predict_proba(X_test_tfidf)[:, 1]
deep_pred = model.deep_model.predict(X_test_tfidf.toarray()).ravel()

# Ağırlıklı ensemble
sgd_acc = model.sgd_classifier.score(X_test_tfidf, y_test)
lr_acc = model.lr_classifier.score(X_test_tfidf, y_test)
deep_acc = model.deep_model.evaluate(X_test_tfidf.toarray(), y_test)[1]

weights = np.array([sgd_acc, lr_acc, deep_acc])
weights = weights / weights.sum()

# Ağırlıkları kaydet
np.save('models/ensemble_weights.npy', weights)

ensemble_pred = (
    weights[0] * sgd_pred +
    weights[1] * lr_pred +
    weights[2] * deep_pred
)

ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
final_accuracy = accuracy_score(y_test, ensemble_pred_binary)

# Karmaşıklık matrisi oluştur ve görselleştir
cm = confusion_matrix(y_test, ensemble_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.savefig('models/confusion_matrix.png')
plt.close()

# Sınıflandırma raporu
classification_rep = classification_report(y_test, ensemble_pred_binary)

print("\nTest Seti Sonuçları:")
print(f"SGD Doğruluk: {sgd_acc:.4f}")
print(f"LR Doğruluk: {lr_acc:.4f}")
print(f"Deep Learning Doğruluk: {deep_acc:.4f}")
print(f"Ensemble Doğruluk: {final_accuracy:.4f}")

print("\nKarmaşıklık Matrisi:")
print(cm)

print("\nSınıflandırma Raporu:")
print(classification_rep)

# Model ağırlıklarını yazdır
print("\nEnsemble Ağırlıkları:")
print(f"SGD Ağırlık: {weights[0]:.4f}")
print(f"LR Ağırlık: {weights[1]:.4f}")
print(f"Deep Learning Ağırlık: {weights[2]:.4f}")

# Her bir modelin ayrı ayrı performans metriklerini hesapla
for model_name, predictions in [
    ("SGD", model.sgd_classifier.predict(X_test_tfidf)),
    ("Logistic Regression", model.lr_classifier.predict(X_test_tfidf)),
    ("Deep Learning", (model.deep_model.predict(X_test_tfidf.toarray()) > 0.5).astype(int))
]:
    print(f"\n{model_name} Detaylı Metrikleri:")
    print(classification_report(y_test, predictions))