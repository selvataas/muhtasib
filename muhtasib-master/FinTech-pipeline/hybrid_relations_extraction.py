#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proje: 5411 – Relation Types Çıkarma (Hybrid: Embedding + LLM)
Amaç: Embedding ile benzer maddeleri bul, LLM'e sadece ilgili maddeleri gönder
"""

import os
import json
import csv
import psycopg2
import google.generativeai as genai
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


# ======================================================
# Config
# ======================================================
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
SIMILARITY_THRESHOLD = 0.92  
TOP_K_SIMILAR = 5  


# ======================================================
#  DB Connection
# ======================================================

def get_db_connection():
    """PostgreSQL bağlantısı kur"""
    conn = psycopg2.connect(
        host="192.168.81.57",
        port=5433,
        database="mevzuat",
        user="stas",
        password="MGmnAWJQGaxrAZHK"
    )
    conn.set_session(autocommit=True)
    return conn


def get_article_text_by_number(article_no: str) -> str:
    """5411 sayılı Kanunun ilgili madde metnini döndür"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.content
        FROM articles a
        LEFT JOIN versionarticle v ON a.id = v.article_id
        WHERE a.law_id = 596
          AND CAST(a.number AS TEXT) = %s
          AND (a.abolition IS NULL OR a.abolition = false)
          AND (a.temporary IS NULL OR a.temporary = false);
    """, (article_no,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row and row[0] else ""


# ======================================================
#  CU Load
# ======================================================

def load_compliance_units(cu_file_path: str) -> List[Dict[str, Any]]:
    with open(cu_file_path, "r", encoding="utf-8") as f:
        cu_list = json.load(f)
    return [cu for cu in cu_list if cu["cu_id"].startswith("CU_5411_")]


def parse_article_number_from_cu_id(cu_id: str) -> str:
    """CU_5411_22/1 → 22"""
    try:
        return cu_id.split("_", 2)[2].split("/")[0]
    except Exception:
        return None


# ======================================================
#  EMBEDDING MODEL
# ======================================================

class EmbeddingAnalyzer:
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"🤖 Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.cache_file = f"embeddings_5411_articles_{model_name.replace('/', '_')}.pkl"
    
    def create_article_embeddings(self, article_texts: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Madde metinleri için embedding oluştur"""
        
        if os.path.exists(self.cache_file):
            print(f" Cache'den yükleniyor: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                if set(article_texts.keys()).issubset(set(cache_data.keys())):
                    return {k: cache_data[k] for k in article_texts.keys()}
        
        print(" Madde embeddings oluşturuluyor...")
        embeddings = {}
        
        article_list = list(article_texts.items())
        texts = [text for _, text in article_list]
        
        encoded = self.model.encode(texts, show_progress_bar=True, batch_size=8)
        
        for i, (article_no, _) in enumerate(article_list):
            embeddings[article_no] = encoded[i]
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f" Cache kaydedildi: {self.cache_file}")
        
        return embeddings
    
    def find_similar_articles(self, article_no: str, embeddings: Dict[str, np.ndarray], 
                            top_k: int = TOP_K_SIMILAR, threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
        """Bir maddeye benzer maddeleri bul"""
        
        if article_no not in embeddings:
            return []
        
        target_embedding = embeddings[article_no].reshape(1, -1)
        similar_articles = []
        
        for other_article, other_embedding in embeddings.items():
            if other_article == article_no:
                continue
            
            similarity = cosine_similarity(target_embedding, other_embedding.reshape(1, -1))[0][0]
            
            if similarity >= threshold:
                similar_articles.append((other_article, float(similarity)))
        
        similar_articles.sort(key=lambda x: x[1], reverse=True)
        return similar_articles[:top_k]


# ======================================================
# GEMINI MODEL
# ======================================================

def setup_llm():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(" GEMINI_API_KEY ortam değişkenini ayarla!")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-2.5-pro")


# ======================================================
#  PROMPT TEMPLATE
# ======================================================

PROMPT_TEMPLATE = """
Proje Promptu: 5411 – Relation Types Çıkarma (Hybrid Analiz)

Aşağıdaki görevi uygula. Türkçe yanıt ver.
Çıktıyı, Excel'e doğrudan kopyalanabilecek, tab-separated formatta üret.

Rol:
Finansal uyum uzmanı bir hukuk analisti gibi çalış. 5411 sayılı Kanun maddeleri arasında (ve gerektiğinde diğer kanun/maddelere) ilişki (relation) etiketlemesi yap.

İlişki Tipleri:
- refer (Reference): Kaynak madde/fıkra, yorum/uygulama için başka bir madde/fıkra veya kanuna atıf yapıyor (bilgi/definisyona ihtiyaç).
- exclude (Exclusion): Kaynak madde/fıkra koşulları sağlanırsa, hedef madde/fıkra uygulanmaz/bertaraf edilir.
- only_include (Exclusive Applicability): Kaynak madde/fıkra koşulları sağlanırsa uygulama kapsamı yalnızca hedef madde/fıkralarla sınırlandırılır.
- should_include (Mandatory Inclusion): Kaynak madde/fıkra, hedef madde/fıkranın kısıtına uymayı açıkça zorunlu kılar (genel kuralı veya başka kuralı bağlayıcı şekilde devreye sokar).

Not: Diğer dört kategoriye girmeyen, ancak belirtilmesi gereken özel durumlar veya notlar için kullanılır.

Girdi
Fıkra Tanımı: Girdi metninde, aralarında boşluk bırakılarak yazılmış her bir paragraf ayrı bir "fıkra" olarak kabul edilir ve buna göre numaralandırılır.

law_no_source: 5411
madde_no: {madde_no}
madde_metni:
{madde_text}

BENZER MADDELER (Embedding ile bulundu):
{similar_articles_context}

Çıktı FORMATI (Sekmeyle Ayrılmış Metin - TSV):
Her madde için fıkra bazında analiz yap ve sonucu aşağıdaki sütun yapısına uygun, tek bir satır olarak üret. Sütunlar arasında ayraç olarak sekme (`\\t`) kullan. Her fıkra yeni bir satırda olmalıdır.

Sütunlar:
Fıkra No\\tReferences\\tExclude\\tonly include\\tshould include\\tNot

Kurallar:
- Çıktı dosyasına başlık satırı ekle.
- Her fıkra için tek bir satır oluştur.
- Sütunlar arasında ayraç olarak sadece sekme (`\\t`) kullan.
- References ilişkisi için çıktıyı yapısal formatta ver: `kanun_no: [Numara] ([Adı]), madde_no: [Numara], fıkra_no: [Numara]`. Metinde belirtilmeyen alanlara "belirtilmemiştir" yaz.
- Diğer sütunlarda bulgu yoksa "Yok" yaz.
- BENZER MADDELER bölümünü dikkate al: Eğer madde metni bu maddelere atıf yapıyorsa, ilişkileri belirt.

ÖNEMLİ KISITLAMALAR - ÇOK KATIYIZ:
- SADECE metinde AÇIKÇA ve DOĞRUDAN belirtilen ilişkileri yaz.
- Yoruma dayalı, dolaylı veya çıkarım gerektiren ilişkileri ASLA EKLEME.
- "References": SADECE metinde madde/fıkra numarası GEÇİYORSA yaz (örn: "8 inci madde", "23. madde").
- "Exclude": SADECE "...uygulanmaz", "...hariç", "...istisna" gibi AÇIK ifadeler varsa.
- "only_include": SADECE "...ile sınırlı", "...sadece" gibi AÇIK kısıtlama varsa.
- "should_include": SADECE "...uygulanır", "...tabi olur" gibi AÇIK zorunluluk varsa.
- "Not": Neredeyse HİÇBİR ZAMAN kullanma. Sadece çok özel durumlar için.
- Şüphe varsa MUTLAKA "Yok" yaz.
- HEDEF: Çoğu fıkrada sadece "Yok" olmalı. Bu NORMALDIR ve BEKLENENDİR.

Örnek çıktı biçimi (SADECE AÇIK İLİŞKİLER):
Fıkra No\\tReferences\\tExclude\\tonly include\\tshould include\\tNot
1\\tkanun_no: 5411, madde_no: 8, fıkra_no: 1, bent_no: a,b,c,d\\tYok\\tYok\\tYok\\tYok
2\\tkanun_no: 5411, madde_no: 23, fıkra_no: 1\\tYok\\tYok\\tYok\\tYok
3\\tYok\\tYok\\tYok\\tYok\\tYok
4\\tYok\\tYok\\tYok\\tYok\\tYok

NOT: Çoğu fıkrada sadece "References" veya hiç ilişki olmayacak. Bu NORMAL ve BEKLENENDİR.
"""


# ======================================================
#  HYBRID ANALYSIS
# ======================================================

def analyze_article_with_hybrid(model, embedding_analyzer, article_no: str, madde_text: str, 
                                embeddings: Dict[str, np.ndarray]) -> str:
    """Hybrid: Embedding ile benzer maddeleri bul, LLM'e context olarak gönder"""
    
    similar_articles = embedding_analyzer.find_similar_articles(article_no, embeddings)
    
    similar_context = ""
    if similar_articles:
        similar_context = "Embedding analizi ile bulunan benzer maddeler:\n"
        for sim_article, sim_score in similar_articles:
            similar_context += f"- Madde {sim_article} (benzerlik: {sim_score:.3f})\n"
    else:
        similar_context = "Bu maddeye benzer başka madde bulunamadı (embedding threshold: {})".format(SIMILARITY_THRESHOLD)
    
    
    prompt = PROMPT_TEMPLATE.format(
        madde_no=article_no,
        madde_text=madde_text.strip(),
        similar_articles_context=similar_context
    )
    
    try:
        response = model.generate_content(prompt)
        output_text = response.text.strip()
        tqdm.write(f"   Madde {article_no} işlendi (benzer: {len(similar_articles)} madde)")
        return output_text
    except Exception as e:
        tqdm.write(f"   [HATA] Madde {article_no} işlenemedi: {e}")
        return ""


# ======================================================
#  OUTPUT FORMATTERS
# ======================================================

def parse_tsv_line(tsv_line: str) -> Dict[str, str]:
    """TSV satırını parse et ve dictionary'ye çevir"""
    parts = tsv_line.split('\t')
    if len(parts) >= 6:
        return {
            'fikra_no': parts[0].strip(),
            'references': parts[1].strip(),
            'exclude': parts[2].strip(),
            'only_include': parts[3].strip(),
            'should_include': parts[4].strip(),
            'not': parts[5].strip()
        }
    return {}


def save_as_tsv(results: List[Dict], output_path: str):
    """Sonuçları TSV formatında kaydet"""
    print(f"\n TSV dosyası oluşturuluyor: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        writer.writerow(['Madde No', 'CU IDs', 'Fıkra No', 'References', 'Exclude', 
                        'Only Include', 'Should Include', 'Not'])
        
        for result in results:
            madde_no = result['madde_no']
            cu_ids = ', '.join(result['cu_ids'])
            tsv_output = result['tsv_output']
            
            # TSV çıktısını satırlara böl
            lines = tsv_output.strip().split('\n')
            for line in lines:
                # Başlık satırını atla
                if line.startswith('Fıkra No'):
                    continue
                if line.strip():
                    parsed = parse_tsv_line(line)
                    if parsed:
                        writer.writerow([
                            madde_no,
                            cu_ids,
                            parsed.get('fikra_no', ''),
                            parsed.get('references', ''),
                            parsed.get('exclude', ''),
                            parsed.get('only_include', ''),
                            parsed.get('should_include', ''),
                            parsed.get('not', '')
                        ])
    
    print(f" TSV dosyası kaydedildi: {output_path}")


def save_as_csv(results: List[Dict], output_path: str):
    """Sonuçları CSV formatında kaydet"""
    print(f"\n CSV dosyası oluşturuluyor: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['Madde No', 'CU IDs', 'Fıkra No', 'References', 'Exclude', 
                        'Only Include', 'Should Include', 'Not'])
        
        for result in results:
            madde_no = result['madde_no']
            cu_ids = ', '.join(result['cu_ids'])
            tsv_output = result['tsv_output']
            
            lines = tsv_output.strip().split('\n')
            for line in lines:
                if line.startswith('Fıkra No'):
                    continue
                if line.strip():
                    parsed = parse_tsv_line(line)
                    if parsed:
                        writer.writerow([
                            madde_no,
                            cu_ids,
                            parsed.get('fikra_no', ''),
                            parsed.get('references', ''),
                            parsed.get('exclude', ''),
                            parsed.get('only_include', ''),
                            parsed.get('should_include', ''),
                            parsed.get('not', '')
                        ])
    
    print(f" CSV dosyası kaydedildi: {output_path}")


def save_as_json(results: List[Dict], output_path: str):
    """Sonuçları JSON formatında kaydet"""
    print(f"\n JSON dosyası oluşturuluyor: {output_path}")
    
    json_data = []
    
    for result in results:
        madde_no = result['madde_no']
        cu_ids = result['cu_ids']
        tsv_output = result['tsv_output']
        madde_text = result['madde_text']
        
        madde_obj = {
            'madde_no': madde_no,
            'cu_ids': cu_ids,
            'madde_text': madde_text,
            'fikralar': []
        }
        
        lines = tsv_output.strip().split('\n')
        for line in lines:
            if line.startswith('Fıkra No'):
                continue
            if line.strip():
                parsed = parse_tsv_line(line)
                if parsed:
                    madde_obj['fikralar'].append(parsed)
        
        json_data.append(madde_obj)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f" JSON dosyası kaydedildi: {output_path}")


# ======================================================
#  MAIN FUNCTION
# ======================================================

def run_5411_hybrid_relation_extraction(cu_file_path: str, output_path: str):
    print(" CU listesi yükleniyor...")
    cu_list = load_compliance_units(cu_file_path)
    print(f" {len(cu_list)} CU bulundu.")

    article_to_cus = {}
    for cu in cu_list:
        cu_id = cu["cu_id"]
        article_no = parse_article_number_from_cu_id(cu_id)
        if not article_no:
            continue
        if article_no not in article_to_cus:
            article_to_cus[article_no] = []
        article_to_cus[article_no].append(cu_id)

    print(f" {len(article_to_cus)} benzersiz madde bulundu.")

    print(" Madde metinleri yükleniyor...")
    article_texts = {}
    for article_no in article_to_cus.keys():
        madde_text = get_article_text_by_number(article_no)
        if madde_text.strip():
            article_texts[article_no] = madde_text
        else:
            print(f" Madde {article_no} metni boş, atlanıyor.")

    print(f" {len(article_texts)} madde metni yüklendi.")

    print(" Embedding analizi başlıyor...")
    embedding_analyzer = EmbeddingAnalyzer(EMBEDDING_MODEL)
    embeddings = embedding_analyzer.create_article_embeddings(article_texts)
    print(f" {len(embeddings)} madde için embedding oluşturuldu.")


    print("\n Benzerlik Analizi Özeti:")
    total_similar = 0
    article_keys = sorted(embeddings.keys(), key=lambda x: int(x))
    
    for article_no in tqdm(article_keys, desc="🔍 Benzerlik hesaplanıyor", unit="madde"):
        similar = embedding_analyzer.find_similar_articles(article_no, embeddings)
        if similar:
            total_similar += len(similar)
            tqdm.write(f"  Madde {article_no}: {len(similar)} benzer madde")
    
    print(f"\n Toplam {total_similar} benzerlik ilişkisi bulundu.")
    print(f" LLM'e her madde için ortalama {total_similar/len(embeddings):.1f} benzer madde context'i gönderilecek.\n")

    print(" Gemini modeli yükleniyor...")
    model = setup_llm()
    print(" Model hazır.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    article_list = sorted([a for a in article_to_cus.keys() if a in article_texts], key=lambda x: int(x))
    
    print(f"\n Hybrid analiz başlıyor ({len(article_list)} madde)...")
    
    all_results = []
    
    with open(output_path, "w", encoding="utf-8") as out:
        for article_no in tqdm(article_list, desc="📊 Maddeler işleniyor", unit="madde"):
            cu_ids = article_to_cus[article_no]
            madde_text = article_texts[article_no]

            tsv_output = analyze_article_with_hybrid(
                model, embedding_analyzer, article_no, madde_text, embeddings
            )
            
            if tsv_output:
                out.write(f"\n### Madde {article_no} (İlgili CU'lar: {', '.join(cu_ids)})\n")
                out.write(tsv_output + "\n")
                
                # Sonuçları parse et ve sakla
                all_results.append({
                    'madde_no': article_no,
                    'cu_ids': cu_ids,
                    'tsv_output': tsv_output,
                    'madde_text': madde_text
                })

            time.sleep(1.0)

    base_path = output_path.rsplit('.', 1)[0]
    tsv_path = f"{base_path}.tsv"
    save_as_tsv(all_results, tsv_path)
    
    csv_path = f"{base_path}.csv"
    save_as_csv(all_results, csv_path)
    
    json_path = f"{base_path}.json"
    save_as_json(all_results, json_path)

    print(f"\n İşlem tamamlandı!")
    print(f" TXT çıktısı: {output_path}")
    print(f" TSV çıktısı: {tsv_path}")
    print(f" CSV çıktısı: {csv_path}")
    print(f" JSON çıktısı: {json_path}")


# ======================================================
#  RUN BLOCK
# ======================================================

if __name__ == "__main__":
    run_5411_hybrid_relation_extraction(
        cu_file_path="/Users/selva/Downloads/cu/muhtesib/data/compliance_unit.json",
        output_path="/Users/selva/Downloads/cu/muhtesib/FinTech-pipeline/5411_relation_types_hybrid.txt"
    )

