import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Ścieżka do ChromeDrivera
service = Service(
    "C:/Users/Dżejnek/Desktop/hackaton/chromedriver-win64/chromedriver.exe")

# Konfiguracja opcji Chrome
options = Options()
options.headless = False  # Możesz ustawić True, jeśli chcesz, żeby przeglądarka działała w tle

# Tworzenie instancji WebDrivera
driver = webdriver.Chrome(service=service, options=options)

# Otwórz stronę Google Grafika
print("Ładowanie strony...")
driver.get("https://www.google.com/search?hl=pl&tbm=isch&q=zupa+grzybowa")

# Czekaj, aby strona mogła się załadować
time.sleep(3)

# Przewijanie strony w dół, aby załadować więcej obrazów
print("Przewijanie strony, aby załadować obrazy...")
body = driver.find_element(By.TAG_NAME, 'body')
for _ in range(3):  # Przewiń stronę 3 razy
    body.send_keys(Keys.END)
    time.sleep(2)

# Znalezienie wszystkich obrazów
images = driver.find_elements(By.TAG_NAME, 'img')

# Stworzenie folderu, jeśli jeszcze nie istnieje
folder_path = "C:/Users/Dżejnek/Desktop/hackaton/dataset/zupa grzybowa"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Pobieranie i zapisywanie obrazów
for i, image in enumerate(images):
    try:
        # Sprawdzanie URL
        img_url = image.get_attribute('src')

        if img_url:
            # Sprawdzamy, czy URL to dane w base64 (które nie są fizycznym plikiem)
            if img_url.startswith('data:image'):
                print(f"Obraz {i + 1} jest w formacie base64, pomijam.")
                continue

            # Sprawdzamy, czy URL zaczyna się od 'https://'
            if img_url.startswith('https://'):
                try:
                    # Pobieranie obrazka
                    img_data = requests.get(img_url).content
                    img_name = f"image_{i + 1}.jpg"
                    img_path = os.path.join(folder_path, img_name)

                    # Zapisz obraz na dysku
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    print(f"Pobrano obraz: {img_name}")
                except Exception as e:
                    print(f"Błąd przy pobieraniu obrazu {i + 1}: {str(e)}")
            else:
                print(f"Obraz {i + 1} ma nieprawidłowy URL: {img_url}")
        else:
            print(f"Nie znaleziono URL obrazu {i + 1}.")

    except Exception as e:
        print(f"Błąd przy pobieraniu obrazu {i + 1}: {str(e)}")

# Zamknięcie przeglądarki
print("Zamknięcie przeglądarki...")
driver.quit()
