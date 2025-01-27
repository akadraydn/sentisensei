from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
import os
import pandas as pd

# Chrome seçeneklerini ayarlama
options = Options()

# Reklam engelleme ayarları
options.add_argument('--disable-notifications')
options.add_argument('--disable-popup-blocking')
options.add_argument('--disable-advertisements')

# User-Agent ekleme
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

# uBlock Origin CRX dosyasının yolunu belirtme
ublock_path = "/Users/akadraydn/Downloads/CJPALHDLNBPAFIAMEJDNHCPHJBKEIAGM_1_62_0_0.crx"
if os.path.exists(ublock_path):
    options.add_extension(ublock_path)

# Tarayıcıyı başlatma
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# URL'nin ilk parçası
url_first_part = "https://www.bbc.com/arabic/topics/cv2xyrnr8dnt?page="

# Sonuçları saklamak için bir liste
all_news_contents = []

# Toplam sekme sayısı
total_page = 589

try:
    # Her sayfadaki haber başlıklarını dolaşma
    for page_num in range(1, total_page + 1):
        main_url = url_first_part + str(page_num)
        driver.get(main_url)

        # Sayfanın tam olarak yüklenmesini beklemek için
        WebDriverWait(driver, 60).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ul.bbc-k6wdzo li.bbc-t44f9r")))

        # Sayfadaki tüm haber linklerini toplama
        news_links = []
        news_divs = driver.find_elements(By.CSS_SELECTOR, "ul.bbc-k6wdzo li.bbc-t44f9r")
        for news_div in news_divs:
            try:
                link = news_div.find_element(By.TAG_NAME, "a").get_attribute('href')
                news_links.append(link)
            except StaleElementReferenceException:
                print(f"StaleElementReferenceException occurred while collecting links on page {page_num}.")
                continue

        # Toplanan linklere giderek haber içeriğini çekme
        for link_url in news_links:
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    driver.get(link_url)
                    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.bbc-4wucq3.ebmt73l0")))

                    news_story_div = driver.find_element(By.CSS_SELECTOR, "div.bbc-4wucq3.ebmt73l0 p.bbc-1gjryo4.e17g058b0")
                    paragraphs = news_story_div.find_elements(By.CSS_SELECTOR, "p.bbc-1gjryo4.e17g058b0 b")
                    content = "\n".join([para.text for para in paragraphs if para.text.strip() != ""])
                    all_news_contents.append({"url": link_url, "content": content})
                    print(f"Successfully scraped: {link_url}")
                    break
                except (StaleElementReferenceException, TimeoutException) as e:
                    retry_count += 1
                    print(f"Error {e} occurred for URL: {link_url}. Retrying {retry_count}/{max_retries}...")
                    if retry_count == max_retries:
                        print(f"Skipping URL: {link_url} after {max_retries} retries.")
                except Exception as e:
                    print(f"Unexpected error occurred for URL: {link_url}: {e}")
                    break

except KeyboardInterrupt:
    print("\nProgram kullanıcı tarafından durduruldu. Toplanan verileri kaydetmeye çalışıyorum...")
finally:
    driver.quit()
    # Toplanan veri miktarını kontrol etme
    print("\nToplanan haber sayısı:", len(all_news_contents))
    if all_news_contents:
        print("İlk haber örneği:", all_news_contents[0])
        
        # Excel'e yazma işlemi
        try:
            df = pd.DataFrame(all_news_contents)
            output_excel = "/Users/akadraydn/Desktop/web-scraping/veri.xlsx"
            df.to_excel(output_excel, index=False, engine='openpyxl')
            print(f"Veriler başarıyla {output_excel} dosyasına kaydedildi!")
        except Exception as e:
            print(f"Excel'e yazma hatası: {str(e)}")
            print("Alternatif olarak CSV formatında kaydetmeyi deniyorum...")
            try:
                output_csv = "/Users/akadraydn/Desktop/web-scraping/veri.csv"
                df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                print(f"Veriler başarıyla {output_csv} dosyasına kaydedildi!")
            except Exception as e:
                print(f"CSV'ye yazma hatası: {str(e)}")
    else:
        print("Kaydedilecek veri bulunamadı!")


