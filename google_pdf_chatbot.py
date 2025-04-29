import os
import time
import google.generativeai as genai
from PyPDF2 import PdfReader
from typing import List, Dict
import threading
from langchain.text_splitter import RecursiveCharacterTextSplitter

class GooglePDFChatbot:
    def __init__(self, pdf_path: str, api_key: str):
        """Initialize the PDF chatbot with the given PDF file path and Google API key."""
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.text_chunks = []
        self.model = None
        self.chat = None
        self.last_request_time = 0
        self.min_request_interval = 4  # 4 saniye bekleme süresi (15 RPM için)
        self.stop_countdown = False
        self.requests_in_last_minute = 0
        self.last_minute_start = time.time()
        self.next_available_time = 0
        
    def show_countdown(self, total_seconds: int, message: str):
        """Bekleme süresini sayaç şeklinde göster."""
        self.stop_countdown = False
        for i in range(total_seconds, 0, -1):
            if self.stop_countdown:
                break
            print(f"\r{message} {i} saniye kaldı...", end="", flush=True)
            time.sleep(1)
        print("\r" + " " * 50 + "\r", end="", flush=True)  # Sayaç satırını temizle
        
    def load_pdf(self) -> bool:
        """Load and process the PDF file."""
        try:
            print("PDF yükleniyor ve parçalara ayrılıyor...")
            reader = PdfReader(self.pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # Metni daha küçük parçalara ayır
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Daha küçük parçalar
                chunk_overlap=100,  # Daha az örtüşme
                length_function=len
            )
            self.text_chunks = text_splitter.split_text(text)
            print(f"PDF başarıyla yüklendi ve {len(self.text_chunks)} parçaya ayrıldı.")
            return True
        except Exception as e:
            print(f"PDF yükleme hatası: {str(e)}")
            return False
            
    def setup_model(self):
        """Set up the Google Gemini model."""
        try:
            print("Model yükleniyor...")
            # API'yi yapılandır
            genai.configure(api_key=self.api_key)
            
            # En son Gemini modelini kullan
            self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
            self.chat = self.model.start_chat(history=[])
            print("Model başarıyla yüklendi.")
            return True
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            print("Lütfen API key'inizin doğru olduğundan ve internet bağlantınızın çalıştığından emin olun.")
            return False
            
    def initialize(self) -> bool:
        """Initialize all components of the chatbot."""
        if not self.load_pdf():
            return False
        if not self.setup_model():
            return False
        return True
        
    def wait_for_rate_limit(self):
        """Rate limit için bekleme süresi."""
        current_time = time.time()
        
        # Dakika başına istek sayısını kontrol et
        if current_time - self.last_minute_start >= 60:
            self.requests_in_last_minute = 0
            self.last_minute_start = current_time
            
        if self.requests_in_last_minute >= 15:  # 15 RPM limiti
            wait_time = 60 - (current_time - self.last_minute_start)
            if wait_time > 0:
                print(f"\nDakika başına istek limiti aşıldı. {wait_time:.0f} saniye bekleyin...")
                countdown_thread = threading.Thread(
                    target=self.show_countdown,
                    args=(int(wait_time), "Dakika başına istek limiti için bekleniyor:")
                )
                countdown_thread.start()
                time.sleep(wait_time)
                self.stop_countdown = True
                countdown_thread.join()
                self.requests_in_last_minute = 0
                self.last_minute_start = time.time()
        
        # İstekler arası minimum bekleme süresi
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            wait_time = int(self.min_request_interval - time_since_last_request)
            print(f"\nİstekler arası bekleme: {wait_time} saniye")
            countdown_thread = threading.Thread(
                target=self.show_countdown,
                args=(wait_time, "İstekler arası bekleme:")
            )
            countdown_thread.start()
            time.sleep(wait_time)
            self.stop_countdown = True
            countdown_thread.join()
        
        self.last_request_time = time.time()
        self.requests_in_last_minute += 1
        self.next_available_time = self.last_request_time + self.min_request_interval
        
    def find_relevant_chunk(self, question: str) -> str:
        """Soruyu en çok ilgilendiren metin parçasını bul."""
        # Sorudaki anahtar kelimeleri bul
        question_lower = question.lower()
        keywords = [word for word in question_lower.split() if len(word) > 3]  # 3 karakterden uzun kelimeler
        
        # Her parça için puan hesapla
        best_chunk = self.text_chunks[0]
        best_score = 0
        
        for chunk in self.text_chunks:
            chunk_lower = chunk.lower()
            score = 0
            
            # Anahtar kelime eşleşmelerini say
            for keyword in keywords:
                if keyword in chunk_lower:
                    score += 1
            
            # Eğer bu parça daha iyi eşleşiyorsa güncelle
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        return best_chunk
        
    def handle_quota_error(self, error_message: str) -> bool:
        """Token kotası aşıldığında bekleme süresini ayarla."""
        try:
            # Hata mesajından bekleme süresini çıkar
            import re
            retry_delay = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_message)
            if retry_delay:
                wait_time = int(retry_delay.group(1))
                countdown_thread = threading.Thread(
                    target=self.show_countdown,
                    args=(wait_time, "Token kotası aşıldı. Bekleniyor:")
                )
                countdown_thread.start()
                time.sleep(wait_time)
                self.stop_countdown = True
                countdown_thread.join()
                return True
        except:
            pass
        return False
        
    def ask_question(self, question: str, max_retries: int = 3) -> str:
        """Ask a question to the chatbot."""
        retry_count = 0
        while retry_count < max_retries:
            try:
                if not self.text_chunks or not self.model:
                    return "Chatbot düzgün başlatılamadı. Lütfen kurulumu kontrol edin."
                
                # Rate limit için bekle
                self.wait_for_rate_limit()
                
                # İlgili metin parçasını bul
                relevant_chunk = self.find_relevant_chunk(question)
                    
                # PDF içeriğini ve soruyu birleştir
                prompt = f"""Aşağıdaki metin bir PDF dosyasından alınmıştır. Lütfen bu metne dayanarak soruyu yanıtlayın.

PDF İçeriği:
{relevant_chunk}

Soru: {question}

Lütfen cevabınızı Türkçe olarak verin ve mümkün olduğunca kısa ve öz tutun. Sadece sorunun cevabına odaklanın, gereksiz detaylardan kaçının. Maksimum 2-3 cümle ile cevaplayın."""
                
                # Soruyu gönder ve cevabı al
                response = self.chat.send_message(prompt)
                
                # Bir sonraki soru için kalan süreyi göster
                current_time = time.time()
                if current_time < self.next_available_time:
                    wait_time = int(self.next_available_time - current_time)
                    print(f"\nBir sonraki soru için {wait_time} saniye bekleyin.")
                
                return response.text
                
            except Exception as e:
                error_message = str(e)
                print(f"\nSoru işleme hatası: {error_message}")
                
                # Token kotası aşıldıysa bekle ve tekrar dene
                if "quota" in error_message.lower():
                    if self.handle_quota_error(error_message):
                        print("\nTekrar deniyor...")
                        retry_count += 1
                        continue
                    else:
                        print("\nAPI kotası aşıldı. Lütfen bir süre bekleyin ve tekrar deneyin.")
                return f"Hata oluştu: {error_message}"
        
        return "Maksimum deneme sayısına ulaşıldı. Lütfen daha sonra tekrar deneyin."

def main():
    # Google API key'inizi buraya girin
    api_key = "API KEY"
    pdf_path = "pdf_path"
    
    print(f"PDF dosyası: {pdf_path}")
    
    chatbot = GooglePDFChatbot(pdf_path, api_key)
    if not chatbot.initialize():
        print("Chatbot başlatılamadı. Lütfen kurulumu kontrol edin.")
        return
        
    print("\nChatbot hazır! Sorularınızı sorabilirsiniz.")
    print("Çıkmak için 'quit' yazın.")
    print("Not: Dakikada en fazla 15 soru sorabilirsiniz.")
    print("Not: Her soru arasında 4 saniye bekleme süresi vardır.")
    print("Not: Token kotası aşılırsa otomatik olarak bekleyecektir.")
    
    while True:
        question = input("\nSorunuz: ")
        if question.lower() == 'quit':
            break
            
        answer = chatbot.ask_question(question)
        print(f"\nCevap: {answer}")

if __name__ == "__main__":
    main() 