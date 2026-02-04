#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import psycopg2
import json
import re
import os
import time
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd

load_dotenv()

# ---------------------
# Config
# ---------------------
@dataclass
class Config:
    db_host: str = "192.168.81.57"
    db_port: int = 5433
    db_name: str = "mevzuat"
    db_user: str = "stas"
    db_password: str = "MGmnAWJQGaxrAZHK"
    output_csv: str = "cu_from_db.csv"
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.1
    max_tokens: int = 2000
    api_delay: float = 0.5
    
    def __post_init__(self):
        # Çoklu kanun ve madde seçimi
        self.laws_and_articles = {
            596: {  # 5411 Bankacılık Kanunu
                "name": "5411 Bankacılık Kanunu",
                "articles": [22, 23, 24, 29, 30, 31, 32, 42, 43, 55, 56, 37, 38, 41, 62, 73, 76, (76, 'A'), 95, 96, 148, 159]
            },
            470: {  # 6362 Sermaye Piyasası Kanunu
                "name": "6362 Sermaye Piyasası Kanunu",
                "articles": [15, 16, 32, 90, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113]
            },
            459: {  # 6493 Ödeme ve Menkul Kıymet Mutabakat Sistemleri Kanunu
                "name": "6493 Ödeme ve Menkul Kıymet Mutabakat Sistemleri Kanunu",
                "articles": [12, 14, (14, 'A'), 18, 23, 27, 28, 31, 34, 35]
            },
            471: {  # 6361 Finansal Kiralama, Faktoring, Finansman ve Tasarruf Finansman Şirketleri Kanunu
                "name": "6361 Finansal Kiralama, Faktoring, Finansman ve Tasarruf Finansman Şirketleri Kanunu",
                "articles": [14, 15, 30, 31, 32, 38, 39, 40, 47, 48, 50]
            },
            1018: {  # 213 Vergi Usul Kanunu
                "name": "213 Vergi Usul Kanunu",
                "articles": [253, 256, 359, 148, 149, 30]
            },
            442: {  # 6698 Kişisel Verilerin Korunması Kanunu
                "name": "6698 Kişisel Verilerin Korunması Kanunu",
                "articles": [4, 5, 8, 9, 12, 28]
            },
            576: {  # 5549 Suç Gelirlerinin Aklanmasının Önlenmesi Hakkında Kanun
                "name": "5549 Suç Gelirlerinin Aklanmasının Önlenmesi Hakkında Kanun",
                "articles": [2, 3, 4, 5, 7, 8, 22, 27, 28]
            },
            467: {  # 6415 Terörizmin Finansmanının Önlenmesi Hakkında Kanun
                "name": "6415 Terörizmin Finansmanının Önlenmesi Hakkında Kanun",
                "articles": [3, 4, 5, 7, 8]
            },
            370: {  # 7262 Kitle İmha Silahlarının Yayılmasının Finansmanının Önlenmesine İlişkin Kanun
                "name": "7262 Kitle İmha Silahlarının Yayılmasının Finansmanının Önlenmesine İlişkin Kanun",
                "articles": [3, 4, 5, 6, 7]
            },
            632: {  # 5237 Türk Ceza Kanunu
                "name": "5237 Türk Ceza Kanunu",
                "articles": [188, 191, 157, 158, 220, 235, 241, 247, 250, 252, 282, 314, 315]
            },
            625: {  # 5271 Ceza Muhakemesi Kanunu
                "name": "5271 Ceza Muhakemesi Kanunu",
                "articles": [46, 160, 161, 128, 135, 140, 206]
            },
            614: {  # 5326 Kabahatler Kanunu
                "name": "5326 Kabahatler Kanunu",
                "articles": [3, 16, 17, 22, 23, 24]
            },
            8788: {  # 39158 Dijital Bankaların Faaliyet Esasları İle Servis Modeli Bankacılığı Hakkında Yönetmelik
                "name": "39158 Dijital Bankaların Faaliyet Esasları İle Servis Modeli Bankacılığı Hakkında Yönetmelik",
                "articles": [3, 4, 8, 10, 11, 12, 16]
            },
            8780: {  # 39171 Sır Niteliğindeki Bilgilerin Paylaşılması Hakkında Yönetmelik
                "name": "39171 Sır Niteliğindeki Bilgilerin Paylaşılması Hakkında Yönetmelik",
                "articles": [4, 5, 6, 7]
            },
            11800: {  # 34495 Finansal Piyasalarda Manipülasyon ve Yanıltıcı İşlemler Hakkında Yönetmelik
                "name": "34495 Finansal Piyasalarda Manipülasyon ve Yanıltıcı İşlemler Hakkında Yönetmelik",
                "articles": [4, 5, 6]
            },
            18454: {  # 10750 Bankaların Kurumsal Yönetim İlkelerine İlişkin Yönetmelik
                "name": "10750 Bankaların Kurumsal Yönetim İlkelerine İlişkin Yönetmelik",
                "articles": [4]
            },
            18465: {  # 10732 Finansal Holding Şirketleri Hakkında Yönetmelik
                "name": "10732 Finansal Holding Şirketleri Hakkında Yönetmelik",
                "articles": [5, 7, 8, 9, 12, 13, 16]
            },
            10457: {  # 38797 Varlık Yönetim Şirketlerinin Kuruluş ve Faaliyet Esasları Yönetmeliği
                "name": "38797 Varlık Yönetim Şirketlerinin Kuruluş ve Faaliyet Esasları Yönetmeliği",
                "articles": [10, 12, 15, 16]
            },
            16720: {  # 15481 Bankaların Destek Hizmeti Almalarına İlişkin Yönetmelik
                "name": "15481 Bankaların Destek Hizmeti Almalarına İlişkin Yönetmelik",
                "articles": [4, 6, 7, 8, 9]
            },
            14760: {  # 22598 Bankaların Kıymetli Maden Alım Satımına İlişkin Usul ve Esaslar Hakkında Yönetmelik
                "name": "22598 Bankaların Kıymetli Maden Alım Satımına İlişkin Usul ve Esaslar Hakkında Yönetmelik",
                "articles": [3, 4]
            },
            15522: {  # 20645 Bankaların Bağımsız Denetimi Hakkında Yönetmelik
                "name": "20645 Bankaların Bağımsız Denetimi Hakkında Yönetmelik",
                "articles": [4, 5, 11, 13, 17, 18, 20, 21, 22]
            },
            18456: {  # 10747 Bankaların Muhasebe Uygulamalarına ve Belgelerin Saklanmasına İlişkin Usul ve Esaslar Hakkında Yönetmelik
                "name": "10747 Bankaların Muhasebe Uygulamalarına ve Belgelerin Saklanmasına İlişkin Usul ve Esaslar Hakkında Yönetmelik",
                "articles": [6, 8, 9]
            },
            16033: {  # 19864 Bankaların İç Sistemleri ve İçsel Sermaye Yeterliliği Değerlendirme Süreci Hakkında Yönetmelik
                "name": "19864 Bankaların İç Sistemleri ve İçsel Sermaye Yeterliliği Değerlendirme Süreci Hakkında Yönetmelik",
                "articles": [5, 6, 7, 8, 10, 22]
            },
            14954: {  # 21278 Bankalarca Yapılacak Repo Ve Ters Repo İşlemlerine İlişkin Esaslar Hakkında Yönetmelik
                "name": "21278 Bankalarca Yapılacak Repo Ve Ters Repo İşlemlerine İlişkin Esaslar Hakkında Yönetmelik",
                "articles": [4, 6]
            },
            18406: {  # 11180 Banka Kartları ve Kredi Kartları Hakkında Yönetmelik
                "name": "11180 Banka Kartları ve Kredi Kartları Hakkında Yönetmelik",
                "articles": [15, 16, 17, 22]
            },
            8650: {  # 39368 Finansal Kiralama, Faktoring, Finansman ve Tasarruf Finansman Şirketlerince Kullanılacak Uzaktan Kimlik Tespiti Yöntemlerine ve Elektronik Ortamda Sözleşme İlişkisinin Kurulmasına İlişkin Yönetmelik
                "name": "39368 Finansal Kiralama, Faktoring, Finansman ve Tasarruf Finansman Şirketlerince Kullanılacak Uzaktan Kimlik Tespiti Yöntemlerine ve Elektronik Ortamda Sözleşme İlişkisinin Kurulmasına İlişkin Yönetmelik",
                "articles": [5, 6, 7, 8, 9]
            },
            11228: {  # 38500 Tasarruf Finansman Şirketlerinin Kuruluş ve Faaliyet Esasları Hakkında Yönetmelik
                "name": "38500 Tasarruf Finansman Şirketlerinin Kuruluş ve Faaliyet Esasları Hakkında Yönetmelik",
                "articles": [13, 14, 15, 16, 17, 19, 20]
            },
            15613: {  # 20506 Faktoring İşlemlerinde Uygulanacak Usul ve Esaslar Hakkında Yönetmelik
                "name": "20506 Faktoring İşlemlerinde Uygulanacak Usul ve Esaslar Hakkında Yönetmelik",
                "articles": [5, 9, 12]
            },
            16259: {  # 19152 Finansal Kiralama, Faktoring, Finansman ve Tasarruf Finansman Şirketlerinin Muhasebe Uygulamaları ile Finansal Tabloları Hakkında Yönetmelik
                "name": "19152 Finansal Kiralama, Faktoring, Finansman ve Tasarruf Finansman Şirketlerinin Muhasebe Uygulamaları ile Finansal Tabloları Hakkında Yönetmelik",
                "articles": [5, 6, 7, 8]
            },
            16403: {  # 18312 Finansal Kiralama, Faktoring ve Finansman Şirketlerinin Kuruluş ve Faaliyet Esasları Hakkında Yönetmelik
                "name": "18312 Finansal Kiralama, Faktoring ve Finansman Şirketlerinin Kuruluş ve Faaliyet Esasları Hakkında Yönetmelik",
                "articles": [12, 13, 14, 15, 16, 18, 19, 20, 21]
            },
            13789: {  # 31395 Finansal Kiralama, Faktoring ve Finansman Şirketlerinin Bilgi Sistemlerinin Yönetimine ve Denetimine İlişkin Tebliğ
                "name": "31395 Finansal Kiralama, Faktoring ve Finansman Şirketlerinin Bilgi Sistemlerinin Yönetimine ve Denetimine İlişkin Tebliğ",
                "articles": [5, 6, 7, 8, 9, 11, 14, 16]
            },
            8833: {  # 39080 Ödeme Hizmetleri ve Elektronik Para İhracı ile Ödeme Kuruluşları ve Elektronik Para Kuruluşları Hakkında Yönetmelik
                "name": "39080 Ödeme Hizmetleri ve Elektronik Para İhracı ile Ödeme Kuruluşları ve Elektronik Para Kuruluşları Hakkında Yönetmelik",
                "articles": [26, 27, 28, 29, 32, 41, 42, 53, 69]
            },
            11501: {  # 39081 Ödeme ve Elektronik Para Kuruluşlarının Bilgi Sistemleri ile Ödeme Hizmeti Sağlayıcılarının Ödeme Hizmetleri Alanındaki Veri Paylaşım Servislerine İlişkin Tebliğ
                "name": "39081 Ödeme ve Elektronik Para Kuruluşlarının Bilgi Sistemleri ile Ödeme Hizmeti Sağlayıcılarının Ödeme Hizmetleri Alanındaki Veri Paylaşım Servislerine İlişkin Tebliğ",
                "articles": [5, 6, 7, 8, 9, 12, 20, 22]
            },
            18505: {  # 10522 Bankacılık Düzenleme ve Denetleme Kurumu Tarafından Yapılacak Denetime İlişkin Usul ve Esaslar Hakkında Yönetmelik
                "name": "10522 Bankacılık Düzenleme ve Denetleme Kurumu Tarafından Yapılacak Denetime İlişkin Usul ve Esaslar Hakkında Yönetmelik",
                "articles": [5, 6, 7, 8, 9, 10, 11, 12]
            }
        }
        
        # Sadece seçili kanunları işle (boşsa tümü işlenir)
        # Örn: enabled_laws = [470] sadece 6362'yi işler
        self.enabled_laws = None  # Tüm kanunlar işlenir
        # self.enabled_laws = [470]  # Sadece 6362 işler
        #self.enabled_laws = [596]  # Sadece 5411 işler
        # self.enabled_laws = [459]  # Sadece 6493 işler
        # self.enabled_laws = [471]  # Sadece 6361 işler
        # self.enabled_laws = [1018]  # Sadece 213 işler
        # self.enabled_laws = [442]  # Sadece 6698 işler
        # self.enabled_laws = [576]  # Sadece 5549 işler
        # self.enabled_laws = [467]  # Sadece 6415 işler
        # self.enabled_laws = [370]  # Sadece 7262 işler
        # self.enabled_laws = [632]  # Sadece 5237 işler
        # self.enabled_laws = [625]  # Sadece 5271 işler
        # self.enabled_laws = [614]  # Sadece 5326 işler
        # self.enabled_laws = [8788]  # Sadece 39158 işler
        # self.enabled_laws = [8780]  # Sadece 39171 işler
        # self.enabled_laws = [11800]  # Sadece 34495 işler
        # self.enabled_laws = [18454]  # Sadece 10750 işler
        # self.enabled_laws = [18465]  # Sadece 10732 işler
        # self.enabled_laws = [10457]  # Sadece 38797 işler
        # self.enabled_laws = [16720]  # Sadece 15481 işler
        # self.enabled_laws = [14760]  # Sadece 22598 işler
        # self.enabled_laws = [15522]  # Sadece 20645 işler
        # self.enabled_laws = [18456]  # Sadece 10747 işler
        # self.enabled_laws = [16033]  # Sadece 19864 işler
        # self.enabled_laws = [14954]  # Sadece 21278 işler
        # self.enabled_laws = [18406]  # Sadece 11180 işler
        # self.enabled_laws = [8650]  # Sadece 39368 işler
        # self.enabled_laws = [11228]  # Sadece 38500 işler
        # self.enabled_laws = [15613]  # Sadece 20506 işler
        # self.enabled_laws = [16259]  # Sadece 19152 işler
        # self.enabled_laws = [16403]  # Sadece 18312 işler
        # self.enabled_laws = [13789]  # Sadece 31395 işler
        # self.enabled_laws = [8833]  # Sadece 39080 işler
        # self.enabled_laws = [11501]  # Sadece 39081 işler
        # self.enabled_laws = [18505]  # Sadece 10522 işler
        
        if self.enabled_laws:
            self.laws_and_articles = {k: v for k, v in self.laws_and_articles.items() if k in self.enabled_laws}

# ---------------------
# DB Helper
# ---------------------
def get_db_connection(cfg: Config):
    conn = psycopg2.connect(
        host=cfg.db_host,
        port=cfg.db_port,
        database=cfg.db_name,
        user=cfg.db_user,
        password=cfg.db_password
    )
    conn.set_session(autocommit=True)
    return conn

def get_maddeler_from_db(cursor, law_id):
    """Veritabanından belirtilen kanunun tüm ana maddelerini çek"""
    cursor.execute(f"""
        SELECT DISTINCT number
        FROM articles
        WHERE law_id = {law_id}
        AND number IS NOT NULL
        AND CAST(number AS TEXT) NOT LIKE '%.%'
        ORDER BY number::int;
    """)
    main_articles = [row[0] for row in cursor.fetchall()]
    
    # Extra=true olan maddeleri (76/A gibi) ekle
    cursor.execute(f"""
        SELECT DISTINCT number
        FROM articles
        WHERE law_id = {law_id}
        AND extra = true
        ORDER BY number::int;
    """)
    extra_articles = [row[0] for row in cursor.fetchall()]
    
    # Combine main articles with extra articles as tuples
    result = []
    for num in main_articles:
        result.append(num)
        # Bu numaranın extra versiyonu var mı?
        if any(e == num for e in extra_articles):
            result.append((num, 'A'))
    
    return result

def sort_maddeler(madde_list):
    """76, 76/A gibi maddeler için custom sort"""
    def madde_key(m):
        if isinstance(m, tuple):
            return (int(m[0]), m[1])
        m_str = str(m)
        if '/' in m_str:
            parts = m_str.split('/')
            return (int(parts[0]), parts[1])
        else:
            return (int(m_str), '')
    return sorted(madde_list, key=madde_key)

def extract_law_number_from_name(law_name):
    """Kanun adından gerçek kanun numarasını çıkar"""
    import re
    # Kanun adının başındaki sayıyı bul (örn: "5411 Bankacılık Kanunu" -> "5411")
    match = re.match(r'^(\d+)', law_name.strip())
    if match:
        return match.group(1)
    return None

def clean_cu_data(cu, law_id, law_no, source_article):
    """Gemini API'den gelen CU verisini düzelt ve temizle"""
    import json
    import re
    
    # source_article'ı string'e çevir
    if isinstance(source_article, tuple):
        # (76, 'A') -> '76/A'
        source_article_str = f"{source_article[0]}/{source_article[1]}"
    else:
        source_article_str = str(source_article)
    
    # cu_id düzelt veya yeniden oluştur
    if "cu_id" in cu:
        cu_id = cu["cu_id"]
        article_num = source_article_str.split('/')[0]
        
        # Eğer cu_id eksik, yanlış veya placeholder içeriyorsa, yeniden oluştur
        if not cu_id or 'XXXX' in cu_id or 'XXX' in cu_id or '_X/' in cu_id or '_X-' in cu_id or '_XX/' in cu_id or '_Y' in cu_id or 'UNKNOWN' in cu_id or cu_id == "CU_":
            # Yeniden oluştur
            fıkra_no = source_article_str.split('/')[-1] if '/' in source_article_str else "1"
            cu["cu_id"] = f"CU_{law_no}_{article_num}/{fıkra_no}"
        else:
            # Var olan cu_id'i temizle
            article_num = source_article_str.split('/')[0]
            
            # Tüm template ve belirsiz değişkenleri düzelt
            replacements = {
                "[kanun_no]": str(law_no),
                "{kanun_no}": str(law_no),
                "[madde_no]": str(article_num),
                "{madde_no}": str(article_num),
                "KANUN_NO": str(law_no),
                "MADDENO": str(article_num),
                "UNKNOWN_UNKNOWN": f"{law_no}_{article_num}",
                "unknown_unknown": f"{law_no}_{article_num}",
                "XXXX_X": f"{law_no}_{article_num}",
                "XXXX_XX": f"{law_no}_{article_num}",
                "XXXX_YY": f"{law_no}_{article_num}",
                "XXXX_YYY": f"{law_no}_{article_num}",
                "XXX_YYY": f"{law_no}_{article_num}",
                "XXX_RiskYonetimiSistemi": f"{law_no}_{article_num}",
                "RiskYonetimiSistemi": f"{article_num}",
                "_MusteriHaklari": f"_{article_num}",
                "_MaddeBelirsiz": f"_{article_num}",
                "MADDE_NO": str(article_num),
            }
            
            for old, new in replacements.items():
                cu_id = cu_id.replace(old, new)
            
            # Regex ile kalan XXXX, XXX, YYY, MusteriHaklari, MaddeBelirsiz gibi desenleri düzelt
            cu_id = re.sub(r'_X{2,}(?=[/_]|$)', f"_{article_num}", cu_id)
            cu_id = re.sub(r'_Y{2,}(?=[/_]|$)', f"_{article_num}", cu_id)
            cu_id = re.sub(r'_[A-Z][a-z]+(?=[/_]|$)', f"_{article_num}", cu_id)
            
            cu["cu_id"] = cu_id
    else:
        # cu_id yoksa yeniden oluştur
        article_num = source_article_str.split('/')[0]
        fıkra_no = source_article_str.split('/')[-1] if '/' in source_article_str else "1"
        cu["cu_id"] = f"CU_{law_no}_{article_num}/{fıkra_no}"
    
    # constraint'i düzelt - string ise array'e dönüştür
    if "constraint" in cu:
        constraint = cu["constraint"]
        if isinstance(constraint, str):
            try:
                if constraint.startswith("['") or constraint.startswith('["'):
                    constraint = constraint.replace("'", '"')
                cu["constraint"] = json.loads(constraint)
            except:
                cu["constraint"] = [constraint]
        elif not isinstance(constraint, list):
            cu["constraint"] = [str(constraint)]
    
    # is_executable boolean'a dönüştür
    if "is_executable" in cu:
        if isinstance(cu["is_executable"], str):
            cu["is_executable"] = cu["is_executable"].lower() == "true"
    
    return cu

def get_number_condition(madde_no):
    """Alt maddeleri (76/A gibi) yada ana maddeler (76 gibi) için condition oluştur"""
    if isinstance(madde_no, tuple):
        # (76, 'A') -> Madde 76'nın extra=true kaydı
        num, sub = madde_no
        return f"number = {num} AND extra = true"
    elif '/' in str(madde_no):
        return f"CAST(number AS TEXT) = '{madde_no}'"
    else:
        # Sadece tam eşleşme - alt maddeleri dahil etme
        return f"number = {madde_no}"

def fetch_article_with_clauses_and_subclauses(cursor, law_id, article_num):
    """Madde, fıkra ve bentleri birlikte çek"""
    madde_display = f"{article_num[0]}/{article_num[1]}" if isinstance(article_num, tuple) else article_num
    
    cursor.execute(f"""
        WITH ranked AS (
            SELECT id, number, name, temporary, abolition, extra, note,
                ROW_NUMBER() OVER (
                    ORDER BY
                        abolition NULLS FIRST,
                        CASE WHEN temporary = false OR temporary IS NULL THEN 0 ELSE 1 END,
                        CASE WHEN CAST(number AS TEXT) LIKE '%/%' THEN 0
                             WHEN extra = false OR extra IS NULL THEN 0
                             ELSE 1 END,
                        CASE WHEN note LIKE '%Geçici%' THEN 1 ELSE 0 END,
                        note NULLS FIRST,
                        id ASC
                ) as rn
            FROM articles
            WHERE law_id = {law_id} 
            AND {get_number_condition(article_num)}
            AND (abolition IS NULL OR abolition = false)
            AND (temporary IS NULL OR temporary = false)
            AND (extra IS NULL OR extra = false)
        )
        SELECT id, number, name
        FROM ranked
        WHERE rn = 1;
    """)
    
    ana_madde = cursor.fetchone()
    if not ana_madde:
        return None
    
    article_id = ana_madde[0]
    
    # Bu maddenin fıkralarını bul
    cursor.execute(f"""
        SELECT id, number, content, abolition
        FROM clauses
        WHERE article_id = {article_id}
        AND (abolition IS NULL OR abolition = false)
        ORDER BY CAST(number AS INTEGER);
    """)
    
    clauses = cursor.fetchall()
    full_data = []
    
    if not clauses:
        # Fıkra yoksa sadece madde name'i döndür
        full_data.append({
            'madde_no': article_num,
            'madde_display': madde_display,
            'article_id': article_id,
            'clause_num': None,
            'clause_content': None,
            'bentler': [],
            'full_text': ana_madde[2]  # name
        })
    else:
        # Her fıkra için bentleri çek
        for clause_id, clause_num, clause_content, clause_abolition in clauses:
            cursor.execute(f"""
                SELECT number, content
                FROM subclauses
                WHERE clause_id = {clause_id}
                AND (abolition IS NULL OR abolition = false)
                ORDER BY number;
            """)
            
            subclauses = cursor.fetchall()
            bentler = [{"bent_num": b[0], "bent_content": b[1]} for b in subclauses]
            
            # Full text: madde + fıkra + bentler
            bentler_text = ""
            if bentler:
                bentler_text = "\n" + "\n".join([f"({b['bent_num']}) {b['bent_content']}" for b in bentler])
            
            full_text = f"{ana_madde[2]}\nFıkra {clause_num}: {clause_content}{bentler_text}"
            
            full_data.append({
                'madde_no': article_num,
                'madde_display': madde_display,
                'article_id': article_id,
                'clause_num': clause_num,
                'clause_content': clause_content,
                'bentler': bentler,
                'full_text': full_text
            })
    
    return full_data

# ---------------------
# CU Extractor
# ---------------------
class ComplianceExtractorFromDB:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        api_key = os.getenv("GEMINI_API_KEY_1")
        if not api_key:
            raise ValueError("GEMINI_API_KEY_1 bulunamadı")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(cfg.model_name)

    def _parse_json(self, content: str):
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except:
            try:
                m = re.search(r'\{.*\}', content, re.DOTALL)
                if m:
                    return json.loads(m.group())
            except:
                pass
        return {}

    def extract_cus(self, text: str, kanun_no: str, madde_no: str, expected_clause_count: int = None):
        """Metinden CU çıkar - gemini_cu1.py promptu kullanarak"""
        if not text or not text.strip():
            return []

        # Template değişkenlerini değiştir
        cu_prompt_template = """
Sen finansal compliance uzmanı bir AI'sın. Görevin aşağıdaki metinden Compliance Units (CU) çıkarmaktır.
Bir maddenin birden fazla paragrafı varsa, bunlardan her birine “fıkra” denir. Fıkralar cümlelerden oluşur. Fıkraların altında bulunan a, b, c, ... gibi sıralama varsa bunlara “bent” denir.
TEMEL KURALLAR:
 
Sadece verilen madde metninden CU çıkar, başka madde üretme.
Subject alanı fıkrada geçen tam lafzı içermelidir. Eğer özne açıkça yazmıyorsa, üst cümleden veya önceki fıkradan gelen özne aynı şekilde tekrar et; yeni özne icat etme.
Her fıkra için mutlaka 1 CU üret; fıkrayı asla bölme. Fıkrayı ifade eden cümleler tek CU olarak ifade edilmeli.
Fıkradaki tüm yükümlülükleri tek CU'da topla, özne/subject mevzuat lafzına uygun olsun.
Contextual_info sadece madde lafzındaki bağlamı açıklasın, yorum veya dış mevzuat ekleme.
Hiçbir maddeyi ve maddenin fıkralarını CU oluştururken atlama. Bir maddenin ne kadar fıkrası var ise; o kadar CU oluşturmalısın. (10 fıkra var ise 10 CU oluşturulmalı)
Mülga (Yürürlükten kaldırılmış, hükümsüz) fıkra varsa CU üretme. Sadece "madde_no/fıkra_no:MÜLGADIR" çıktısı ver.
Fıkra bazlı tekil CU çıkar, atomize etme, madde dışına çıkma.
Boş alan bırakma. Condition hiçbir zaman "N/A" olamaz. (Aşağıdaki Condition kurallarına uy).
CU üretirken sadece verilen madde metnindeki hükümleri kullan.
 
YÜKÜMLÜLÜK İÇERMEYEN FIKRALAR İÇİN KESİN KURAL (N/A YASAĞI)
 
'constraint' alanı ASLA "N/A" veya boş olamaz. Bir fıkrada doğrudan uygulanabilir bir yükümlülük, görev veya yasak bulunmuyorsa, fıkranın niteliğini aşağıdaki gibi sınıflandırarak 'constraint' alanını DOLDURMAK ZORUNLUDUR:
Eğer fıkra bir tanım yapıyorsa: is_executable = false, constraint = ["Tanım hükmüdür."]
Eğer fıkra bir kuruma veya kişiye yetki veriyorsa (örn: "Kurul belirlemeye yetkilidir"): is_executable = false, constraint = ["Yetki hükmüdür."]
Eğer fıkra bir idari prosedürü veya yöntemi açıklıyorsa (örn: "Başvurular şu şekilde yapılır..."): is_executable = false, constraint = ["Usul hükmüdür."]
Eğer fıkra başka bir maddeye atıf yaparak o maddenin geçerli olduğunu belirtiyorsa: is_executable = false, constraint = ["Atıf hükmüdür."]
Bu kural, her fıkranın anlamsal olarak sınıflandırılmasını ve "N/A" gibi anlamsız çıktılardan kaçınılmasını garanti eder.
 
MADDE NUMARASI TANIMA KURALI:
 
Türk mevzuatında madde numaraları yalnızca sayılardan oluşmaz; aşağıdaki biçimlerin tamamı madde_no olarak kabul edilir ve fıkra numaralarıyla karıştırılmamalıdır:
cu_id = CU_[kanun_no]_[madde_no]/[fıkra_no] formatında olmalı. madde_no ve fıkra_no için aşağıdaki bilgileri dikkate al.
Türk mevzuatında madde_no yalnızca sayı olmayabilir: 1, 45/A, 76/A-1, 8 Mükerrer, Ek Madde 3, Geçici Madde 2/B vb. Madde_no ile fıkra_no’yu karıştırma. “/A”, “/B”, “Ek Madde”, “Geçici Madde”, “Mükerrer” ibareleri madde numarasının parçasıdır.
Ör: “76/A” → madde_no = “76A”; “Ek Madde 2” → “EkMadde2”, 1, 45/A, 76/A-1, 8 Mükerrer, Ek Madde 3, Geçici Madde 2/B = bunların hepsi kanun maddelerinin numaralandırma şeklidir, fıkra değildir.
 
Fıkra numaraları genellikle (1), (2), (3)... biçiminde parantez içinde yazılır.
Örneğin: “76/A” → madde_no = “76A” = 76/A maddesi demektir. "76/1" → madde_no/fıkra_no = 76. maddenin 1.fıkrası demektir. “Ek Madde 2” → madde_no = “EkMadde2”  Bu tür maddeler CU kimliğinde ayrı madde olarak gösterilir (örnek: CU_5411_76A/1 / CU_5411_76/1)
 
**GELİŞMİŞ FIKRA TANIMLAMA VE SAYIM KURALLARI**
 
1.  **Temel Kural:** Metindeki her paragraf bir fıkradır ve bir CU oluşturur.
2.  **Bent Sonrası Fıkralar:** Bir fıkradan sonra gelen (a), (b), (c)... veya 1., 2., 3... gibi alt bentler, ana fıkranın parçasıdır. Ancak bu bent bloğu bittikten sonra başlayan **yeni bir paragraf**, numarasız bile olsa, **yeni ve ayrı bir fıkradır** ve mutlaka sayıma dahil edilmelidir.
    * **ÖRNEK YAPI:**
        (5) Beşinci fıkra metni...
            a) alt bent...
            b) alt bent...
        (6) Altıncı fıkra metni... -> **BU YENİ BİR FIKRADIR, ATLANAMAZ!**
3.  **Hukuki Atıf ve Ek Metinleri:** `(Ek:26/6/2024-7518/10 md.)` veya `(Değişik:...)` gibi parantez içindeki ifadeler, bir fıkranın başlangıcında yer alsa bile, **o fıkranın varlığını ortadan kaldırmaz.** Bu ifadeleri yok say ve takip eden paragraf metnini yeni bir fıkra olarak işle.
4.  **Fazladan CU Üretme Yasağı:** Bir fıkrayı asla ikiye bölme. Metindeki doğal paragraf sonları tek sınırdır. Eğer bir paragraf çok uzunsa veya içinde birden fazla cümle varsa, bu tek bir fıkradır ve tek bir CU üretilmelidir. Fıkra sayısından fazla CU üretmek kesinlikle yasaktır.
 
ÖZET CU KİMLİĞİ FORMAT
 
cu_id = CU_[kanun_no]_[madde_no]/[fıkra_no] (bent varsa harf ekle: /[fıkra_no][bent_harfi]).
Ör: CU_5411_148/1a, CU_5411_148/1b, CU_5411_76A/1 / CU_5411_76/1)
 
GELİŞMİŞ CU KİMLİĞİ (cu_id) OLUŞTURMA KURALLARI
 
Temel Kural: Madde numarasındaki /A, /B, Ek Madde, Geçici Madde gibi ifadeler madde numarasının ayrılmaz parçasıdır ve madde_no alanına birleşik yazılır.Madde 76/A -> madde_no = 76A -> cu_id = CU_..._76A/1
Ek Madde 2 -> madde_no = EkMadde2 -> cu_id = CU_..._EkMadde2/1
Çok Parçalı Madde Numaraları: Bazı kanunlarda Madde 14/A altında ayrıca (1) numaralı fıkralar olabilir. Bu durumda da kural aynıdır. /A maddeye aittir, (1) fıkraya.Metin: Madde 14/A – (1) Birinci fıkra metni...
DOĞRU ÇIKTI: cu_id = CU_..._14A/1
YANLIŞ ÇIKTI: cu_id = CU_..._14/1 (Bu, ana Madde 14'ün 1. fıkrası ile karışıklığa yol açar!)
ID Çakışmasını Önleme: Ürettiğin cu_id'lerin tüm çıktı boyunca benzersiz olduğundan emin ol. Asla aynı cu_id'yi iki farklı fıkra için kullanma.
 
FIKRA NUMARALANDIRMA KURALI:
 
Her paragraf ve/veya (1),(2),(3)… numarası = 1 fıkra kabul edilir; hiçbir fıkrayı atlamadan her fıkra için CU oluştur.
Üretim sonunda “Üretilen CU sayısı = fıkra sayısı” olmalıdır.
Her kanun maddesinin her fıkrası ayrı bir CU (Compliance Unit) oluştur.
Örn: Madde altı(6) fıkradan oluşuyorsa tam altı(6) CU üretmelisin.
 
Eğer metinde açık fıkra numarası bulunmuyorsa, her paragrafı ayrı fıkra olarak değerlendir ve fıkra numarasını sıralı biçimde ata (örnek: 56/1, 56/2, 56/3). Fıkra numaraları sıralı ve eksiksiz olmalıdır.
CU kimlikleri, metnin paragraf yapısına göre deterministik biçimde üretilmelidir.
Her CU'nun madde_no ve fıkra_no değeri sıralı ve benzersiz olmalıdır (1, 2, 3 …). Madde_no veya fıkra_no tekrarı varsa bunu düzelt.
FIKRA SAYISI, NUMARALANDIRMA VE ZORUNLU TAMLIK KURALI
 
Bu görevde fıkra atlamak KESİNLİKLE yasaktır.
Her paragraf veya (1), (2), (3)... biçimindeki numaralı kısım = 1 fıkra.
Her madde için ürettiğin CU sayısı, kanun maddesindeki fıkra sayısıyla eşleşmelidir.
Sonuçta ürettiğin CU sayısını kontrol et ve “Toplam CU sayısı = fıkra sayısı” kuralına göre eksik varsa tamamla.
“veya” bağlacıyla başlayan uzun cümleler tek fıkradaysa, bunları ayırma.
Her fıkra için mutlaka 1 CU üret. Fıkrada yalnızca yetki, tanım veya usul varsa bile CU oluştur:
Yetki/tanım → is_executable = false, constraint = ["Yetki hükmüdür." veya "Tanım hükmüdür."]
 
CU üretimi bittikten sonra şu öz-kontrolü yap:“Üretilen CU sayısı” = “fıkra sayısı” olmalı.
Eğer azsa → eksik fıkralar için açıklayıcı CU ekle (boş alan bırakma).
ALT BENT KURALI:
 
Madde içinde alt bentler (a, b, c, …) varsa:Eğer bentler aynı konunun alt adımlarını veya tek bir yükümlülüğün parçalarını oluşturuyorsa, bunları aynı CU içinde topla (constraint dizisi olarak).
Eğer madde yalnızca tek fıkradan oluşuyor ve (a), (b), (c)... gibi bentlerle istisnalar sıralanmışsa, her bentteki yükümlülüğü veya istisnayı constraint dizisi içinde ayrı eleman olarak yaz.
Hiçbir madde, hiçbir fıkra ve hiçbir bent atlanmamalıdır.
COMPLIANCE UNIT(CU) KURALLARI: CU (Compliance Unit/Uyum Birimi) = Subject(Özne) + Condition(Koşul) + Constraint(Yükümlülük) + Contextual Information
 
SUBJECT KURALLARI:
 
Subject (Özne): Kural tarafından hukuken bağlanan kişi veya kurum(lar) (örneğin, “yönetim kurulu üyeleri" | "denetçiler", "üst düzey yöneticiler”, “Bankalar”, “Kurul (BDDK)”, “Denetim komitesi” vb.).
“Kurul”, “Kurum”, “Komite” gibi genel isimleri ilk geçtiği yerde kanunun maddesinden ve kanun numarasından anlaşılacağı şekilde genel isimlerin tam resmî adını belirt:Örn: 5411 sayılı kanunda “Kurul” → “Bankacılık Düzenleme ve Denetleme Kurulu (BDDK)” olabilir veya kanun maddesinin içeriğinden farklı bir kuruldan da söz ediliyor olabilir. Bunu kanun maddesindeki anlama göre tespit et.
Örn: 5411 sayılı kanunda “Kurum” → “Bankacılık Düzenleme ve Denetleme Kurumu (BDDK)” olabilir
Örn: 5411 sayılı kanunda “Denetim komitesi” → “Bankaların denetim komitesi (yönetim kuruluna bağlı)” olabilir
Cümlelerin başında özne (subject) tekrar edebilir.
 
 
CONDITION KURALLARI:
 
Condition (Koşul): Yükümlülüğün uygulanacağı belirli bağlam veya tetikleyici senaryo.
Condition asla “N/A” olamaz.
Eğer açık koşul yoksa, “Bu hüküm hangi durumda işletilir/denetlenir?” sorusunun cevabını yaz.
Condition fıkrada yer alan tetikleyici bir unsuru içermelidir (en az biri):
• Zaman: “yedi iş günü içinde”, “her altı ayda bir” vb.  • Olay/Eylem: “atama yapıldığında”, “risk tespit edildiğinde”, “bilanço kapatılmadan önce” vb.  • Durum: “mevzuata aykırılık bulunduğunda”, “oran %60’ı aşarken” vb.
 
Genel ifadeler tek başına yeterli değildir (“faaliyetlerini yürütürken” gibi). Gerekirse süreci belirt:
• Örn: “faaliyetlerini yürütürken” → “faaliyetlerini yürütürken finansal raporlama sürecinde” şeklinde.
 
İstisna/olumsuz koşullar (“hariç”, “ancak”, “dışında”) condition’da negatif şart olarak yazılmalıdır (“… olmadıkça”, “… hariç”).
Sayısal oran/süre/limitler condition/constraint içinde metindeki numeric biçimiyle korunur (örn. “%15”, “%60”, “10 iş günü”).
Birden fazla eylem varsa condition çoğul süreci kapsamalıdır:
• Yanlış: “Risk yönetimi sistemi kapsamında faaliyetlerini yürütürken”  • Doğru: “Risk politikalarını oluşturma, uygulama ve raporlama sürecinde”
 
Birden fazla sistem birlikte anılıyorsa condition hepsini kapsar:
• “İç kontrol, risk yönetimi ve iç denetim sistemleri kapsamında faaliyet yürütürken”
 
Condition, gerçek hayatta hangi olay, süreç veya veri durumunda bu yükümlülüğün devreye gireceğini açıkça belirtmelidir.
Condition alanında sayısal oran sınırlamaları (ör. %15, %60) geçiyorsa, tetikleyici olay “oran hesaplanırken” veya “pay edinilirken” şeklinde açıkça yazılmalıdır.Sayısal oran, süre veya limit içeren constraint'lerde numeric değer orijinal metindeki biçimiyle (ör. “%15”, “%60”) korunmalıdır; yuvarlama veya yorum yapılmaz.
Condition alanı, sistemsel bir izleme veya denetim uyarısını tetikleyebilecek şekilde tanımlanmalıdır.
Condition alanında “faaliyetlerini yürütürken” gibi genel ifadeler yalnızca başka seçenek kalmadığında kullanılabilir.
Eğer condition alanında “faaliyet süresince”, “her zaman” gibi genel ifadeler yer alıyorsa, bu genel bağlamın hangi eylem veya sürece ilişkin olduğu açıkça belirtilmelidir.
Örneğin: “faaliyet süresince” → “faaliyet süresince finansal raporlama süreçlerinde” veya “her zaman iç denetim faaliyetlerini yürütürken”.
 
Condition alanında “sırları bildikleri sürece” gibi belirsiz ifadeler kullanılmamalıdır;Eylemi başlatan tetikleyici durum (“görev sırasında sır öğrenildiğinde”, “bilgi paylaşımı yapıldığında” vb.) açıkça yazılmalıdır.
CONDITION ALANI ZORUNLULUĞU:
 
Her CU için hiçbir alan boş veya “N/A” olamaz.
Condition alanında asla “N/A” kullanılmamalıdır; her durumda bir tetikleyici olay veya süreç (“Kurum denetim yaptığında”, “bilgi talep edildiğinde”, “rapor düzenlendiğinde”) yazılmalıdır.
Fıkrada “hariç”, “istisna”, “dışında”, “ancak” gibi ifadeler bulunuyorsa, bu ibareler condition alanında negatif koşul olarak (“... olmadıkça”, “... hariç”) yazılmalıdır. Bu, Compliance-to-Code şemasındaki exception clause mantığına uygun şekilde condition’a entegre edilmelidir.
CONDITION ZENGİNLEŞTİRME VE VERİ TETİKLEYİCİ KURALI
 
Condition alanı, sistem tarafından denetlenebilir bir tetikleyici (trigger) görevi görmelidir. Bu nedenle şu unsurlardan en az biri yer almalıdır:
• Süreç: (“faaliyet yürütürken”, “raporlama yapılırken”, “fon transferi gerçekleştirilirken”)    • Sistem: (“iç kontrol sistemi kapsamında”, “risk değerlendirme sürecinde”)    • Olay: (“mutabakat yapılmadığında”, “uygunsuzluk tespit edildiğinde”)
 
Eğer metinde açık bir koşul yoksa, LLM şu soruyu kendine sormalıdır: “Bu hüküm hangi durumda işletilir veya denetlenir?” → Bu sorunun cevabı "Condition" alanına yazılmalıdır.
Böylece "Condition" alanı, uyarı sistemlerinde spesifik tetikleyici olarak kullanılabilir.
 
 
CONSTRAINT KURALI
 
Constraint(Kısıtlama/Yükümlülük): Kural tarafından getirilen zorunlu eylem, yasak veya gerekliliklerdir. (Örneğin, “15 gün önceden ilan eder”, “3 ayı aşamaz”, “paylarını azaltması yasaktır”).Bu alanda sayısal sınırlar ve yükümlülük bildiren durumlar (“gereklilik”, “zorunluluk”, “yükümlülük”, vb.) aynen korunur.
“constraint” her zaman JSON dizi (liste) olmalıdır. Tek cümle bile olsa şu şekilde yaz: ["..."].
Her constraint fiil tabanlı olmalı (“oluşturmakla yükümlüdür”, “bildirmekle sorumludur”, “yetkilidir” gibi).
İç sistem unsurları birlikte sayılmışsa her biri için ayrı constraint yaz.
Eğer bir fıkrada iç sistem unsurları (örneğin: iç kontrol, risk yönetimi, iç denetim vb. gibi) birlikte veya ardışık biçimde sayılmışsa, her birini ayrı constraint cümlesi olarak açıkla.
Bir fıkrada birden fazla yükümlülük, görev veya yetki varsa, her birini constraint dizisi içinde ayrı madde olarak yaz.
İç kontrol/denetim ilkeleri (tarafsızlık, bağımsızlık, meslekî özen gibi) ayrı ayrı constraint olmalıdır.
Her constraint özneyi açık biçimde içermeli ve fiille bitmelidir (“... oluşturmakla yükümlüdür”, “... uygulamakla yükümlüdür” vb.).
Şartlar:
 
Eğer fıkrada iç kontrol faaliyetlerinin kim tarafından yürütüleceği belirtilmişse (örneğin “yönetim kuruluna bağlı iç kontrol birimi tarafından yürütülür”), bu unsuru da constraint alanına ekle.
Eğer fıkrada iç denetim faaliyetleri için “tarafsızlık”, “bağımsızlık” veya “meslekî özen” gibi ilkelere atıf varsa, bu ilkelerin her biri ayrı constraint olarak yazılmalıdır.
Eğer fıkrada aynı özneye ait birden fazla fiil geçiyorsa (“… kurmakla yükümlüdür.”, “… işletmekle yükümlüdür.”, “… raporlamakla yükümlüdür.”,“oluşturmakla yükümlüdür." "uygulamakla yükümlüdür." vb. gibi), bu fiillerin her biri ayrı constraint olarak yazılmalıdır.
Eğer bir fıkrada iç sistem unsurları (örneğin: iç kontrol, risk yönetimi, iç denetim vb. gibi) birlikte veya ardışık biçimde sayılmışsa, her birini ayrı constraint cümlesi olarak açıkla.Bu tür ilkeler, genel açıklama cümlesi içine sıkıştırılamaz; ayrı ve yürütülebilir nitelikte constraint ifadeleri halinde yer almalıdır.
Eğer fıkrada belirli bir süre, tarih veya bildirim yükümlülüğü varsa (“en geç 7 iş günü içinde”, “her altı ayda bir”, “faaliyete başlamadan önce” gibi), bu ifadeyi açıkça condition alanında belirt.
Eğer Kurul”, “Kurum”, “Komite” gibi genel isimler yer alıyorsa ilk geçtiği yerde kanunun maddesinden ve kanun numarasından anlaşılacağı şekilde genel isimlerin tam resmî adını belirt:Örn: 5411 sayılı kanunda “Kurul” → “Bankacılık Düzenleme ve Denetleme Kurulu (BDDK)” olabilir veya kanun maddesinin içeriğinden farklı bir kuruldan da söz ediliyor olabilir. Bunu kanun maddesindeki anlama göre tespit et.
Örn: 5411 sayılı kanunda “Kurum” → “Bankacılık Düzenleme ve Denetleme Kurumu (BDDK)” olabilir
Örn: 5411 sayılı kanunda “Denetim komitesi” → “Bankaların denetim komitesi (yönetim kuruluna bağlı)” olabilir
Eğer “Kurul” gibi genel bir kurum adı constraint içinde geçiyorsa, bu kurumun tam adı (“Bankacılık Düzenleme ve Denetleme Kurulu – BDDK”) parantez içinde veya açık biçimde belirtilmelidir.
Eğer fıkrada birden fazla yükümlülük, görev, yasak veya yetki varsa; her biri ayrı bir dizi elemanı olarak yazılmalıdır.
Doğru örnek:  "constraint": [     "Bankalar iç kontrol sistemi kurmakla yükümlüdür.",      "Bankalar risk yönetimi sistemi işletmekle yükümlüdür."   ]   Yanlış örnek:  "constraint": "['Bankalar iç kontrol sistemi kurmakla yükümlüdür.', 'Bankalar risk yönetimi sistemi işletmekle yükümlüdür.']"
 
Tek bir yükümlülük olsa bile "constraint" bir liste içinde tek eleman olarak yazılmalıdır.
(Örnek: "constraint": ["Bankalar iç kontrol sistemi kurmakla yükümlüdür."])
 
Her "constraint" cümlesi fiil tabanlı, yürütülebilir olmalı ve nokta (.) ile bitmelidir.
4. CONTEXTUAL_INFORMATION
 
Contextual information (Bağlamsal Bilgi): Uyum Biriminin doğru şekilde yorumlanması veya uygulanması için gerekli olan, ancak "Subject", "Condition" veya "Constraint" alanlarına girmeyen tamamlayıcı bilgilerdir.Contextual Information bölümünde; tanımlar, hesaplama yöntemleri, belirli ek veya forma yapılan atıflar gibi detaylar yer alır.
"Subject", "Condition" veya "Constraint" alanlarına doğrudan dahil edilebilecek bilgiler buraya yazılmaz.
Eğer bir fıkrada hem yetki hem yükümlülük içeren ifadeler varsa (ör. “Kurul belirler ve uygulatır”), yalnızca yükümlülük kısmı constraint’e alınmalı, yetki kısmı contextual_info’ya yazılmalıdır.
CU ÇIKARTMA KURALLARI:
 
Her fıkradan SADECE 1 CU çıkar. Fıkraları cümle cümle ayırmak YASAK.
CU output formatı JSON olmalı.
"Constraint" mutlaka JSON array of string formatında olmalıdır.
Örnek:  "constraint": ["(a) Yönetim kurulu yıllık plan yapar", "(b) Faaliyet raporu hazırlar", "(c) Plan ve raporu bakanlığa sunar"]
 
Listeyi string içine alma; her madde bağımsız string olmalı.
Tek cümle bile olsa constraint köşeli parantez içinde tek eleman olarak yazılmalıdır.(Örnek: "constraint": ["Bankalar yıllık rapor hazırlar."])
5. EXECUTABLE BELİRLEME KURALI:
 
Her kanun hükmü analiz edilirken:Açık/örtük görev, yükümlülük, yasak, sınır varsa → is_executable = true
Sadece yetki/izin/idari usul/prosedür tanımı varsa → is_executable = false
“Kurul belirler”, “Bakanlık düzenler”, “Kurum onaylar” gibi idari yetki cümleleri → is_executable = false
CONDITION ZENGİNLEŞTİRME VE VERİ TETİKLEYİCİ KURALI
 
Condition alanı, sistem tarafından denetlenebilir bir tetikleyici (trigger) görevi görmelidir
Bu nedenle şu unsurlardan en az biri yer almalıdır:    • Süreç: (“faaliyet yürütürken”, “raporlama yapılırken”, “fon transferi gerçekleştirilirken”)    • Sistem: (“iç kontrol sistemi kapsamında”, “risk değerlendirme sürecinde”)    • Olay: (“mutabakat yapılmadığında”, “uygunsuzluk tespit edildiğinde”)
 
Eğer metinde açık bir koşul yoksa, LLM şu soruyu kendine sormalıdır:
“Bu hüküm hangi durumda işletilir veya denetlenir?”  Bu sorunun cevabı condition alanına yazılmalıdır.
 
Böylece condition alanı, uyarı sistemlerinde spesifik tetikleyici olarak kullanılabilir.
OUTPUT CHECK: Her CU’nun condition alanı tetikleyici olay, zaman veya durum içeriyor mu?
Eğer içermiyorsa, “Eksik tetikleyici” uyarısı ver ve condition’u revize et.
 
ÇOKLU SÜREÇ VE EYLEM TETİKLEYİCİ KURALI
 
Eğer fıkrada birden fazla eylem veya fiil (ör. “oluşturmak, uygulamak, raporlamak”, “kurmak, işletmek”, “hazırlamak, sunmak”) yer alıyorsa,
condition bu fiillerin her birini kapsayacak şekilde çoğul biçimde yazılmalıdır. Örnek: Yanlış: "Risk yönetimi sistemi kapsamında faaliyetlerini yürütürken" Doğru: "Risk politikalarını oluşturma, uygulama ve raporlama sürecinde"
 
Eğer fıkrada birden fazla sistem (ör. iç kontrol, risk yönetimi, iç denetim) birlikte geçiyorsa,
condition alanı bu sistemleri birlikte içerecek biçimde yazılmalıdır. Örnek: Yanlış: "İç sistemler kapsamında faaliyet yürütürken" Doğru: "İç kontrol, risk yönetimi ve iç denetim sistemleri kapsamında faaliyet yürütürken"
 
Eğer fıkrada başka bir maddeye, yönetmeliğe veya kanuna açık atıf bulunuyorsa (“... 38 inci madde uyarınca”, “... Yönetmelikte belirtilen esaslara göre”), contextual_info alanında bu atıf belirtilmeli ve relation_type alanına “refer to” yazılmalıdır. Eğer fıkra başka bir CU’nun uygulanmasını zorunlu kılıyorsa, relation_type = "should include" olarak not düşülmelidir.
ATIFLAR VE İLİŞKİLER
 
Başka madde/kanun/yönetmeliğe açık atıf varsa contextual_info’da belirt (örn. “37 nci maddeye istinaden…”). (İlişki alanı gerekiyorsa: relation_type: "refer to"; başka CU’nun uygulanmasını zorunlu kılıyorsa “should include”.)
HARD STOP – TAMLIK ZORUNLULUĞU
 
Tek bir fıkrayı ve benti bile atlamak kesinlikle yasaktır.
Son doğrulama: “fıkra sayısı = CU sayısı (ve bentli fıkralarda bent sayısı = bent-CU sayısı)”. Sağlanmıyorsa eksikleri üretip düzelt.
Son doğrulama: Her CU’nun condition alanı tetikleyici olay, zaman veya durum içeriyor mu? → Eğer içermiyorsa, “Eksik tetikleyici” uyarısı ver ve condition’u revize et.
MUTLAK KURAL: FIKRA ATLAMAYI ÖNLEME MEKANİZMASI
Bu görevdeki en kritik hata, bir fıkrayı atlamaktır. Bunu önlemek için aşağıdaki adımları harfiyen uygula:
1. **Adım 1: Fıkra Sayımı (Ön Analiz):** CU üretimine başlamadan önce, sana verilen metindeki toplam fıkra (paragraf) sayısını say ve bu sayıyı hafızanda tut. Bu senin "hedef CU sayısı" olacak.
2. **Adım 2: Sıralı Üretim:** Metindeki her bir fıkra için sırayla, 1'den başlayarak hedef sayıya ulaşana kadar CU üret. Hiçbirini atlama.
3. **Adım 3: Son Kontrol ve Tamamlama (Zorunlu):** Tüm CU'ları ürettikten sonra, ürettiğin toplam CU sayısını Adım 1'de belirlediğin fıkra sayısıyla karşılaştır. * **Eğer sayılar eşit değilse, bu bir hatadır.**
Eksik olan fıkraları hemen tespit et ve onlar için derhal, diğer kurallara uygun şekilde CU'ları üret. * **Unutma:** Tanım, yetki veya usul belirten fıkralar atlanmaya en müsait olanlardır. Bu bir hata olur. Bu tür fıkralar için `is_executable = false` olarak ayarlanmış bir CU üretmek zorunludur.
 
 
**DEMİR KURAL: 2 AŞAMALI DÜŞÜNCE ZİNCİRİ (CHAIN-OF-THOUGHT) PROTOKOLÜ**
Görevi, aşağıdaki 2 Aşamalı protokole göre yürütmek zorundasın. Bu protokol, en başta metnin yapısını doğru anlamanı ve fıkra atlamamanı garanti eder.
 
**AŞAMA 1: YAPI ANALİZİ VE PLANLAMA (ÖN ÇIKTI)**
CU üretimine başlamadan ÖNCE, sana verilen metni analiz et ve aşağıdaki formatta bir YAPI ANALİZ PLANI oluştur. Bu plan, senin metni nasıl anladığını gösterecek.
 
--- YAPI ANALİZ PLANI ---
Madde Metni: [Metnin ilk 20 kelimesi...]
Tespit Ettiğim Fıkra Yapısı:
- Fıkra 1: (Numarası: [varsa (1), yoksa inferred]) - Başlangıç: "[Fıkranın ilk 5 kelimesi...]"
- Fıkra 2: (Numarası: [varsa (2), yoksa inferred]) - Başlangıç: "[Fıkranın ilk 5 kelimesi...]"
- Fıkra 3: (Numarası: [varsa a), yoksa inferred]) - Başlangıç: "[Fıkranın ilk 5 kelimesi...]"
- ... (tüm fıkralar için devam et)
Toplam Tespit Edilen Fıkra Sayısı: [Yukarıda listelediğin toplam sayı]
Hedeflenen CU Sayısı: [Yukarıdaki sayının aynısı]
Plan: Şimdi bu yapıya göre, SADECE [tespit ettiğin sayı] adet CU üreteceğim. Fıkra atlamayacağım ve fazladan CU üretmeyeceğim.
--- YAPI ANALİZ PLANI SONU ---
 
**AŞAMA 2: CU ÜRETİMİ (NİHAİ ÇIKTI)**
Yukarıda oluşturduğun YAPI ANALİZ PLANI'na %100 sadık kalarak, `compliance_units` JSON çıktısını üret. Planında belirttiğin fıkra sayısı kadar CU üret.
JSON FORMAT:
{{{{
"compliance_units": [
{{{{
"cu_id": "CU_{kanun_no}_{madde_no}_[fıkra_no]",
"subject": "Kim? (mevzuat lafzına uygun özne, sade biçimde)",
"condition": "Ne zaman? (koşullar, genel faaliyet bağlamı dahil, tetikleyici durum açık olmalı)",
"constraint": ["...","..."],
"contextual_info": "Sadece madde lafzındaki bağlam",
"is_executable": true
}}}}
]
}}}}
{{{{
"compliance_units": [
{{{{
"cu_id": "CU_{kanun_no}_{madde_no}_[fıkra_no]",
"subject": "Kim? (mevzuat lafzına uygun özne, sade biçimde)",
"condition": "Ne zaman? (koşullar, genel faaliyet bağlamı dahil, tetikleyici durum açık olmalı)",
"constraint": ["...","..."],
"contextual_info": "Sadece madde lafzındaki bağlam",
"is_executable": false
}}}}
]
}}}}
Metin: {text}

ÖNEMLİ: Bu metinden CU çıkarırken:
- Kanun numarası: {kanun_no}
- Madde numarası: {madde_no}
- cu_id formatı: CU_{kanun_no}_{madde_no}/[fıkra_no] şeklinde olmalı
{clause_info}
"""
        
        # HTML etiketlerini temizle
        import re
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Fıkra sayısı bilgisini ekle
        if expected_clause_count:
            clause_info = f"\n\n🚨 **ZORUNLU KURAL: Bu maddede TAM OLARAK {expected_clause_count} fıkra vardır!**\n"
            clause_info += f"🚨 **Sen MUTLAKA {expected_clause_count} adet CU üreteceksin!**\n"
            clause_info += f"🚨 **{expected_clause_count}'den az veya fazla CU üretirsen YANLIŞ yapmış olursun!**\n"
            clause_info += f"🚨 **Her fıkra = 1 CU kuralına kesinlikle uy!**\n\n"
        else:
            clause_info = ""
        
        # Template değişkenlerini gerçek değerlerle değiştir ve fıkra sayısı bilgisini ekle
        final_prompt = cu_prompt_template.format(
            kanun_no=kanun_no,
            madde_no=madde_no,
            text=clean_text[:3000] if len(clean_text) > 3000 else clean_text,
            clause_info=clause_info
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Timeout ve retry ayarları ile API çağrısı
                response = self.model.generate_content(
                    final_prompt,
                    request_options={'timeout': 120}  # 120 saniye timeout
                )
                content = response.text or ""
                result = self._parse_json(content)
                return result.get("compliance_units", [])
            except Exception as e:
                print(f"⚠️ API Hatası (Deneme {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 saniye bekle
                    print(f"   {wait_time} saniye bekleniyor...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Maksimum deneme sayısına ulaşıldı, bu madde atlanıyor.")
                    return []

# ---------------------
# Main Process
# ---------------------
def main():
    cfg = Config()
    
    extractor = ComplianceExtractorFromDB(cfg)
    all_cus = []
    
    conn = get_db_connection(cfg)
    cursor = conn.cursor()
    
    total_articles = sum(len(v['articles']) for v in cfg.laws_and_articles.values())
    
    print("📊 Çoklu kanun CU çıkarımı başlıyor...")
    print(f"✅ İşlenecek kanun sayısı: {len(cfg.laws_and_articles)}")
    print(f"✅ İşlenecek toplam madde sayısı: {total_articles}\n")
    
    with tqdm(total=total_articles, desc="CU Extraction", unit="madde") as pbar:
        for law_id, law_info in cfg.laws_and_articles.items():
            law_name = law_info['name']
            selected_articles = law_info['articles']
            
            print(f"\n🏛️ {law_name} ({law_id}) - {len(selected_articles)} madde işleniyor...")
            
            # Gerçek kanun numarasını çıkar
            actual_law_number = extract_law_number_from_name(law_name)
            if not actual_law_number:
                print(f"⚠️ Kanun numarası çıkarılamadı: {law_name}")
                continue
                
            for madde_no in sort_maddeler(selected_articles):
                madde_display = f"{madde_no[0]}/{madde_no[1]}" if isinstance(madde_no, tuple) else madde_no
                
                try:
                    # Veritabanı bağlantısını kontrol et ve gerekirse yeniden kur
                    try:
                        cursor.execute("SELECT 1")
                    except:
                        print("   🔄 Veritabanı bağlantısı yeniden kuruluyor...")
                        cursor.close()
                        conn.close()
                        conn = get_db_connection(cfg)
                        cursor = conn.cursor()
                    
                    # Madde + fıkra + bentleri çek
                    article_data_list = fetch_article_with_clauses_and_subclauses(cursor, law_id, madde_no)
                    
                    if article_data_list:
                        # Tüm fıkraları birleştir ve tek seferde API'ye gönder
                        full_article_text = "\n\n".join([data['full_text'] for data in article_data_list])
                        
                        # Fıkra sayısını hesapla
                        clause_count = len(article_data_list)
                        
                        # CU çıkar - gerçek kanun numarasını ve fıkra sayısını kullan
                        cus = extractor.extract_cus(full_article_text, actual_law_number, madde_display, clause_count)
                        
                        print(f"   📊 Madde {madde_display}: {len(cus)} CU çıkarıldı (beklenen: {clause_count})")
                        
                        if cus:
                            for cu in cus:
                                # CU verisini temizle ve düzelt - gerçek kanun numarasını kullan
                                cu = clean_cu_data(cu, law_id, actual_law_number, madde_no)
                                cu["law_id"] = law_id
                                cu["law_name"] = law_name
                                cu["source_article"] = madde_display
                                cu["extracted_at"] = datetime.now().isoformat()
                            
                            all_cus.extend(cus)
                        time.sleep(cfg.api_delay)
                    
                except Exception as e:
                    print(f"   ❌ Madde {madde_display} işlenirken hata: {str(e)}")
                
                pbar.update(1)
    
    cursor.close()
    conn.close()
    
    # Sonuçları CSV'ye kaydet
    if all_cus:
        df = pd.DataFrame(all_cus)
        df.to_csv(cfg.output_csv, index=False, encoding="utf-8")
        print(f"\n✅ Toplam {len(all_cus)} CU çıkarıldı!")
        print(f"📁 Dosya kaydedildi: {cfg.output_csv}")
        print(f"\n📊 Kanunlar itibariyle CU dağılımı:")
        for law_id, law_info in cfg.laws_and_articles.items():
            count = len([cu for cu in all_cus if cu.get('law_id') == law_id])
            print(f"  - {law_info['name']}: {count} CU")
    else:
        print("❌ Hiç CU çıkarılamadı!")

if __name__ == "__main__":
    main()
