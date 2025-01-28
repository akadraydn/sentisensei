# SentiSensei: Arapça Metin Analizi Projesi

Bu proje, Arapça metinler üzerinde duygu analizi ve kategori sınıflandırması yapan bir web uygulamasıdır. Projenin web arayüzüne [sentisensei.com] adresinden ulaşabilirsiniz.

## Sistem Gereksinimleri

- macOS veya Windows işletim sistemi
- Python 3.10.11
- pip (Python paket yöneticisi)

## Kurulum Adımları

### macOS için:

1. **Python Kurulumu:**
   - [Python'un resmi sitesinden](https://www.python.org/downloads/) Python 3.10.11 sürümünü indirin
   - Kurulum sırasında "Add Python 3.10 to PATH" seçeneğini işaretleyin
   - Kurulumdan sonra terminali açıp Python sürümünü kontrol edin:
   ```bash
   python3 --version
   ```

2. **VSCode Python Yorumlayıcı Ayarları:**
   - VSCode'u açın
   - Klavyeden `Cmd + Shift + P` tuşlarına basın
   - Açılan komut paletine "Python: Select Interpreter" yazın ve Enter'a basın
   - Listeden Python 3.10.11 sürümünü seçin
   - VSCode'un sağ alt köşesinde seçilen Python sürümünün "3.10.11" olduğunu doğrulayın

3. **Proje Dosyalarını İndirin:**
   ```bash
   git clone https://github.com/akadraydn/SentiSensei.git
   cd SentiSensei
   ```

3. **Sanal Ortam Oluşturun ve Aktifleştirin:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

### Windows için:

1. **Python Kurulumu:**
   - [Python'un resmi sitesinden](https://www.python.org/downloads/) Python 3.10.11 sürümünü indirin
   - Kurulum sırasında "Add Python 3.10 to PATH" seçeneğini işaretleyin
   - Kurulumdan sonra Komut İstemi'ni açıp Python sürümünü kontrol edin:
   ```cmd
   python --version
   ```

2. **VSCode Python Yorumlayıcı Ayarları:**
   - VSCode'u açın
   - Klavyeden `Ctrl + Shift + P` tuşlarına basın
   - Açılan komut paletine "Python: Select Interpreter" yazın ve Enter'a basın
   - Listeden Python 3.10.11 sürümünü seçin
   - VSCode'un sağ alt köşesinde seçilen Python sürümünün "3.10.11" olduğunu doğrulayın

3. **Proje Dosyalarını İndirin:**
   ```cmd
   git clone https://github.com/akadraydn/SentiSensei.git
   cd SentiSensei
   ```

3. **Sanal Ortam Oluşturun ve Aktifleştirin:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

### Her İki İşletim Sistemi için Ortak Adımlar:

4. **Gerekli Paketleri Yükleyin:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Yüklenen önemli paket sürümleri:
   - Flask==3.0.2
   - Flask-CORS==5.0.0
   - Werkzeug==3.1.1
   - NumPy==1.26.4
   - Pandas==2.1.4
   - Scikit-learn==1.5.2
   - TensorFlow-macos==2.16.2 (macOS için)
   - TensorFlow-metal==1.1.0 (macOS için)
   - Keras==3.8.0
   - NLTK==3.8.1
   - Joblib==1.4.2

5. **Model Dosyalarını İndirin:**
   Aşağıdaki model dosyalarını indirip `models` klasörüne kopyalayın:
   - [arabic_classifier.keras](https://drive.google.com/file/d/1p6HNek8N61STgwG6wQ1aSlITQRq9wLgd/view?usp=sharing)
   - [best_deep_model.keras](https://drive.google.com/file/d/1Q18NEeIzLq8zaiqJLPz-XYr8ICpdeqOX/view?usp=sharing)
   - [tfidf_vectorizer.joblib](https://drive.google.com/file/d/1LUintLQGEWq3pg67_8HC3mki1eyXOfi5/view?usp=sharing)
   - [tokenizer.joblib](https://drive.google.com/file/d/1gcEnb8kiaO6RsJ0X_q51qJh-xFmzaMOf/view?usp=sharing)
   - [label_encoder.joblib](https://drive.google.com/file/d/1DSfvEjsJBEary3kQ6CBUnJvs0ZQpHW6K/view?usp=sharing)
   - [sgd_classifier.joblib](https://drive.google.com/file/d/1JEEc9Z-_zPRAisvnx62KzJ5b1Mm1Hl19/view?usp=sharing)
   - [logistic_regression.joblib](https://drive.google.com/file/d/1YhQp9Kb2omyVMEcAuQjSQIgpNaz1vhzD/view?usp=sharing)
   - [ensemble_weights.npy](https://drive.google.com/file/d/1Uy7QjNnwaLRUrvvNLbhMWtuavcvACiYJ/view?usp=sharing)

   Not: Model dosyalarının boyutları:
   - best_deep_model.keras: ~148MB
   - arabic_classifier.keras: ~13MB
   - Diğer dosyalar: <5MB

6. **API'yi Başlatın:**
   ```bash
   python3 flask-api.py  # macOS için
   python flask-api.py   # Windows için
   ```
   API başarıyla başlatıldığında "Running on http://127.0.0.1:5002" mesajını göreceksiniz.

## Port Kullanımı

API varsayılan olarak 5002 portunda çalışır. Eğer port kullanımda olduğuna dair hata alırsanız:

### macOS için:
```bash
# Portu kullanan uygulamayı bulun
lsof -i :5002

# Uygulamayı durdurun (PID_NUMARASI yerine üstteki komuttan aldığınız PID'yi yazın)
kill -9 PID_NUMARASI
```

### Windows için:
```cmd
# Portu kullanan uygulamayı bulun
netstat -ano | findstr :5002

# Uygulamayı durdurun (PID_NUMARASI yerine üstteki komuttan aldığınız PID'yi yazın)
taskkill /PID PID_NUMARASI /F
```

## Sorun Giderme

1. **TensorFlow Hataları (macOS):**
   - Apple Silicon Mac kullanıyorsanız, TensorFlow'un özel sürümlerini kullanmanız gerekir:
   ```bash
   pip uninstall tensorflow tensorflow-macos tensorflow-metal
   pip install tensorflow-macos tensorflow-metal
   ```

2. **ModuleNotFoundError Hataları:**
   - Sanal ortamın aktif olduğundan emin olun
   - Paketleri yeniden yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Dosyası Hataları:**
   - `models` klasöründeki tüm dosyaların doğru konumda olduğunu kontrol edin
   - Dosya boyutlarını kontrol edin:
     - best_deep_model.keras: ~148MB
     - arabic_classifier.keras: ~13MB
     - Diğer dosyalar: <5MB

4. **NLTK Veri Hatası:**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Test İçin Örnek Metinler

Aşağıdaki Arapça metinleri test amaçlı kullanabilirsiniz:

1. Spor:
   ```
   فاز فريق برشلونة بالمباراة النهائية في دوري أبطال أوروبا
   ```
   (Barcelona takımı UEFA Şampiyonlar Ligi final maçını kazandı)

2. Teknoloji:
   ```
   أطلقت شركة آبل هاتفها الذكي الجديد بميزات متطورة
   ```
   (Apple şirketi gelişmiş özelliklerle yeni akıllı telefonunu piyasaya sürdü)

3. Sağlık:
   ```
   يساعد النوم الجيد على تقوية جهاز المناعة في الجسم
   ```
   (İyi uyku vücudun bağışıklık sistemini güçlendirmeye yardımcı olur)

4. Finans:
   ```
   ارتفعت أسعار النفط في الأسواق العالمية بشكل ملحوظ
   ```
   (Küresel piyasalarda petrol fiyatları önemli ölçüde yükseldi)

5. Kültür:
   ```
   افتتح متحف جديد يعرض القطع الأثرية النادرة من العصر القديم
   ```
   (Antik dönemden nadir eserlerin sergilendiği yeni bir müze açıldı)

6. Din:
   ```
   يجتمع المسلمون في شهر رمضان للصلاة والصيام
   ```
   (Müslümanlar Ramazan ayında ibadet ve oruç için bir araya gelirler)

7. Politika:
   ```
   عقد مجلس الأمن الدولي اجتماعاً طارئاً لمناقشة الأزمة
   ```
   (BM Güvenlik Konseyi krizi görüşmek üzere acil toplantı düzenledi)

8. Eğitim:
   ```
   تقدم الجامعة منحاً دراسية للطلاب المتفوقين في العلوم
   ```
   (Üniversite, bilimde başarılı öğrencilere burs veriyor)

9. Çevre:
   ```
   تتخذ الحكومة إجراءات صارمة للحد من التلوث البيئي
   ```
   (Hükümet çevre kirliliğini azaltmak için sıkı önlemler alıyor)

10. Ekonomi:
    ```
    حقق الاقتصاد المحلي نمواً ملحوظاً في الربع الأخير
    ```
    (Yerel ekonomi son çeyrekte önemli bir büyüme kaydetti)

Metinleri kopyalayıp web arayüzündeki metin kutusuna yapıştırarak analiz edebilirsiniz.

## Önemli Notlar

- macOS Apple Silicon (M1/M2) kullanıcıları için özel TensorFlow sürümleri gereklidir
- Python sürümü kesinlikle 3.10.11 olmalıdır
- Sanal ortam kullanımı zorunludur
- API çalışırken Terminal/Komut İstemi penceresini kapatmayın
- Sistem sadece Arapça metinler için optimize edilmiştir


