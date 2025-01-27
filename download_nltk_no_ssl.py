import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    try:
        print("Punkt indiriliyor...")
        nltk.download('punkt', quiet=False)
        print("Stopwords indiriliyor...")
        nltk.download('stopwords', quiet=False)
        print("ISRI indiriliyor...")
        nltk.download('isri', quiet=False)
        print("İndirme tamamlandı!")
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    download_nltk_data() 