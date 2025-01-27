import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_dataset():
    # Excel dosyasını oku
    df = pd.read_excel('neg-pos-dataset/original_dataset.xlsx')
    
    # Sadece gerekli sütunları al
    df = df[['rating', 'review_description']]
    
    # Sütun isimlerini değiştir
    df.columns = ['sentiment', 'content']
    
    # 0 değerlerini içeren satırları sil
    df = df[df['sentiment'] != 0]
    
    # 1 ve -1 değerlerini pos ve neg olarak değiştir
    df['sentiment'] = df['sentiment'].map({1: 'pos', -1: 'neg'})
    
    # Pozitif ve negatif yorumları ayır
    pos_df = df[df['sentiment'] == 'pos']
    neg_df = df[df['sentiment'] == 'neg']
    
    # Eğitim ve test verilerini ayır (%80 eğitim, %20 test)
    pos_train, pos_test = train_test_split(pos_df['content'], test_size=0.2, random_state=42)
    neg_train, neg_test = train_test_split(neg_df['content'], test_size=0.2, random_state=42)
    
    # Kayıt dizini
    save_dir = '/Users/akadraydn/Desktop/SentiSensei/neg-pos-dataset'
    
    # Dosyaları kaydet
    pos_train.to_csv(os.path.join(save_dir, 'train_pos.tsv'), 
                     index=False, 
                     header=False,
                     sep='\t')
    pos_test.to_csv(os.path.join(save_dir, 'test_pos.tsv'), 
                    index=False, 
                    header=False,
                    sep='\t')
    neg_train.to_csv(os.path.join(save_dir, 'train_neg.tsv'), 
                     index=False, 
                     header=False,
                     sep='\t')
    neg_test.to_csv(os.path.join(save_dir, 'test_neg.tsv'), 
                    index=False, 
                    header=False,
                    sep='\t')
    
    print("Veri seti işlendi ve dosyalara ayrıldı:")
    print(f"Pozitif eğitim örneği sayısı: {len(pos_train)}")
    print(f"Pozitif test örneği sayısı: {len(pos_test)}")
    print(f"Negatif eğitim örneği sayısı: {len(neg_train)}")
    print(f"Negatif test örneği sayısı: {len(neg_test)}")
    print(f"\nDosyalar şuraya kaydedildi: {save_dir}")
    print("- train_pos.tsv")
    print("- test_pos.tsv")
    print("- train_neg.tsv")
    print("- test_neg.tsv")

if __name__ == "__main__":
    preprocess_dataset() 

    