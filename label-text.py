import pandas as pd
from transformers import pipeline

# Sentiment analizini gerçekleştirmek için pipeline oluştur
model_name = 'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment'
sa = pipeline('sentiment-analysis', model=model_name)

# Excel dosyasını oku
input_file_path = "/Users/akadraydn/Desktop/web-scraping/dataset/news_content.xlsx"
df = pd.read_excel(input_file_path)

# Metinleri 512 token'lık parçalara bölüp analiz et
def label_sentiment(text):
    if not isinstance(text, str):
        return "nötr"

    sentiments = []
    for i in range(0, len(text), 512):
        chunk = text[i:i+512]
        result = sa(chunk)[0]['label']
        sentiments.append(result)

    # Parçaların sonuçlarını birleştir
    if sentiments.count("positive") > sentiments.count("negative"):
        return "pos"
    else:
        return "neg"

# Her bir metni etiketle
df['sentiment'] = df['content'].apply(label_sentiment)

# Sonuçları kaydet
output_file_path = "/Users/akadraydn/Desktop/web-scraping/dataset/etiketlenmis_metinler.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Etiketlenmiş veriler başarıyla kaydedildi: {output_file_path}")
