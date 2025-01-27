# SentiSensei: Arapça Metin Analizi Projesi

Bu proje, Arapça metinler üzerinde duygu analizi ve kategori sınıflandırması yapan bir web uygulamasıdır. Projenin web arayüzüne [sentisensei.com] adresinden ulaşabilirsiniz.

## Sistem Gereksinimleri

- macOS işletim sistemi
- Python 3.9
- pip (Python paket yöneticisi)

## Kurulum Adımları

1. **Python Kurulumu:**
   - Eğer bilgisayarınızda Python 3.9 kurulu değilse, [Python'un resmi sitesinden](https://www.python.org/downloads/) indirip kurabilirsiniz
   - Kurulumdan sonra terminali açıp aşağıdaki komutu yazarak Python'un kurulu olduğunu doğrulayın:
   ```bash
   python3 --version
   ```

2. **Proje Dosyalarını İndirin:**
   - [Bu linkten](https://github.com/akadraydn/SentiSensei) projeyi ZIP olarak indirin
   - İndirdiğiniz ZIP dosyasını çıkartın
   - Terminal'i açın ve çıkarttığınız klasöre gidin:
   ```bash
   cd İNDİRDİĞİNİZ_KLASÖRÜN_YOLU
   ```

3. **Sanal Ortam Oluşturun:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Gerekli Paketleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Model Dosyalarını İndirin:**
   - [Bu Google Drive linkinden](https://drive.google.com/drive/folders/1xzU4M7fmLlHNCSS4S66QjnKym0puYlXJ?usp=sharing) model dosyalarını indirin
   - İndirdiğiniz ZIP dosyasını çıkartın ve içindeki tüm dosyaları `models` klasörüne kopyalayın

6. **API'yi Başlatın:**
   ```bash
   python flask-api.py
   ```
   Terminal'de "Running on http://127.0.0.1:5002" benzeri bir mesaj görmelisiniz.

## Kullanım

1. API'yi başlattıktan sonra web tarayıcınızdan [sentisensei.com] adresine gidin
2. Metin kutusuna analiz etmek istediğiniz Arapça metni girin
3. "Analiz Et" butonuna tıklayın
4. Sonuçlar otomatik olarak görüntülenecektir

## Port Kullanımı

API, 5002 portunda çalışacak şekilde ayarlanmıştır. Eğer bu port başka bir uygulama tarafından kullanılıyorsa, API başlatılamayacak ve bir hata mesajı göreceksiniz. Bu durumda:

1. Tüm Terminal pencerelerini kapatın
2. Bilgisayarınızı yeniden başlatın
3. Sadece bu uygulamayı çalıştırın

Alternatif olarak, macOS'ta şu komutla 5002 portunu kullanan uygulamayı bulup kapatabilirsiniz:

```bash
lsof -i :5002
```

Bu çıktıda görünen PID numarasını kullanarak uygulamayı kapatın:
```bash
kill -9 PID_NUMARASI
```

## Sorun Giderme

1. **"ImportError: numpy.core.multiarray failed to import" Hatası:**
   ```bash
   pip uninstall numpy
   pip uninstall tensorflow
   pip install numpy==1.23.5
   pip install tensorflow==2.12.0
   ```

2. **"ModuleNotFoundError" Hatası:**
   - Sanal ortamın aktif olduğundan emin olun
   - `requirements.txt` dosyasındaki tüm paketlerin yüklendiğini kontrol edin

3. **Bağlantı Hatası:**
   - Terminal'de API'nin çalışır durumda olduğunu kontrol edin
   - Bilgisayarınızın internet bağlantısını kontrol edin

4. **Model Dosyası Hatası:**
   - `models` klasöründe aşağıdaki dosyaların olduğunu kontrol edin:
     - arabic_classifier.keras
     - best_deep_model.keras
     - tfidf_vectorizer.joblib
     - tokenizer.joblib
     - label_encoder.joblib
     - sgd_classifier.joblib
     - logistic_regression.joblib
     - ensemble_weights.npy

5. **Port Kullanım Hatası:**
   - "Port 5002 is already in use" hatası alırsanız:
     1. Tüm Terminal pencerelerini kapatın
     2. Bilgisayarınızı yeniden başlatın
     3. Sadece bu uygulamayı çalıştırın

## Önemli Notlar

- API çalışır durumdayken Terminal penceresini kapatmayın
- Bilgisayarınızı yeniden başlattığınızda API'yi tekrar başlatmanız gerekecektir
- Sistem sadece Arapça metinler için çalışmaktadır
- Web arayüzündeki "Dosya Yükle" özelliği şu anda aktif değildir, lütfen metin kutusunu kullanın
- Analiz için örnek metinleri aşağıdaki "Test İçin Örnek Metinler" bölümünden kopyalayabilirsiniz

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


