#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np


load_dotenv()

import openai
try:
    from google import genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from zai import ZaiClient
    ZAI_AVAILABLE = True
except ImportError:
    ZAI_AVAILABLE = False

# Model Configuration
AVAILABLE_MODELS = {
    "openai": {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini", 
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-3.5-turbo": "gpt-3.5-turbo"
    },
    "google": {
        "gemini-pro": "gemini-2.5-pro",
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro-exp": "gemini-2.0-flash-exp"
    },
    "anthropic": {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307"
    },
    "nebius": {
        "glm-4.5": "glm-4.5",
        "GLM-4.5": "glm-4.5",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "zai-org/GLM-4.5": "zai-org/GLM-4.5"
    }
}


# Enhanced Config
@dataclass
class Config:
    input_excel: str
    sheet: int
    output_csv: str
    quality_report_csv: str
    model_provider: str = "openai"
    model_name: str = "gpt-4o"
    min_constraint_words: int = 3
    max_constraint_words: int = 20
    min_actionable_score: float = 0.6
    min_feasibility_score: float = 0.7
    batch_size: int = 10
    temperature: float = 0.1
    max_tokens: int = 2000
    api_delay: float = 0.3
    max_retries: int = 3
    retry_delay: float = 5.0
    # New parameters from paper
    use_few_shot: bool = True
    max_examples: int = 3
    use_cot_reasoning: bool = True
    reasoning_depth: str = "detailed"  
    difficulty_levels: List[str] = None

    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ["simple", "medium", "difficult"]

def read_config(path: str) -> Config:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        input_excel=raw["input_excel"],
        sheet=int(raw.get("sheet", 0)),
        output_csv=raw["output_csv"],
        quality_report_csv=raw.get("quality_report_csv", "quality_report.csv"),
        model_provider=raw.get("model_provider", "openai"),
        model_name=raw.get("model_name", "gpt-4o"),
        min_constraint_words=int(raw.get("min_constraint_words", 3)),
        max_constraint_words=int(raw.get("max_constraint_words", 20)),
        min_actionable_score=float(raw.get("min_actionable_score", 0.6)),
        min_feasibility_score=float(raw.get("min_feasibility_score", 0.7)),
        batch_size=int(raw.get("batch_size", 5)),
        temperature=float(raw.get("temperature", 0.1)),
        max_tokens=int(raw.get("max_tokens", 4000)),
        api_delay=float(raw.get("api_delay", 1.0)),
        max_retries=int(raw.get("max_retries", 3)),
        retry_delay=float(raw.get("retry_delay", 5.0)),
        use_few_shot=bool(raw.get("use_few_shot", True)),
        max_examples=int(raw.get("max_examples", 3)),
        use_cot_reasoning=bool(raw.get("use_cot_reasoning", True)),
        reasoning_depth=raw.get("reasoning_depth", "detailed")
    )

# Compliance Patterns (from paper)
COMPLIANCE_PATTERNS = {
    "threshold_check": {
        "keywords": ["eşik", "üzerinde", "altında", "%", "oran", "oranı", "fazla", "az"],
        "template": "df[column] >= threshold",
        "complexity": "simple"
    },
    "time_constraint": {
        "keywords": ["gün", "ay", "yıl", "süre", "tarih", "içinde", "kadar", "önce", "sonra"],
        "template": "datetime comparison logic",
        "complexity": "medium"
    },
    "prohibition": {
        "keywords": ["yasaklanır", "yapamaz", "edemez", "olmaz", "yasak", "mümkün değil"],
        "template": "not (condition)",
        "complexity": "simple"
    },
    "disclosure": {
        "keywords": ["açıklama", "bildirim", "duyuru", "ilan", "yayın", "kamuoyu"],
        "template": "disclosure validation logic",
        "complexity": "medium"
    },
    "approval_required": {
        "keywords": ["onay", "izin", "karar", "gerekir", "zorunlu", "mecburi"],
        "template": "approval check logic",
        "complexity": "medium"
    },
    "quantitative_limit": {
        "keywords": ["limit", "sınır", "maksimum", "minimum", "en fazla", "en az"],
        "template": "quantitative boundary check",
        "complexity": "simple"
    },
    "temporal_sequence": {
        "keywords": ["ardından", "sonrası", "öncesi", "aynı anda", "beraber"],
        "template": "temporal ordering logic",
        "complexity": "difficult"
    },
    "conditional_obligation": {
        "keywords": ["durumunda", "halinde", "koşulunda", "şartı ile"],
        "template": "if condition then obligation",
        "complexity": "medium"
    }
}


# API Helper Functions
def call_api_with_retry(api_func, cfg: Config, *args, **kwargs):
    """Enhanced API call with retry logic"""
    last_exception = None
    
    for attempt in range(cfg.max_retries):
        try:
            if attempt > 0:
                print(f"🔄 Retry {attempt}/{cfg.max_retries} - {cfg.retry_delay}s bekliyor...")
                time.sleep(cfg.retry_delay)
            else:
                time.sleep(cfg.api_delay)
            
            result = api_func(*args, **kwargs)
            return result
            
        except Exception as e:
            last_exception = e
            error_str = str(e)
            
            if "429" in error_str or "quota" in error_str.lower():
                print(f"API quota aşıldı. {cfg.retry_delay * 2}s bekliyor...")
                time.sleep(cfg.retry_delay * 2)
                continue
            elif "rate" in error_str.lower():
                print(f"Rate limit. {cfg.retry_delay}s bekliyor...")
                time.sleep(cfg.retry_delay)
                continue
            else:
                print(f"API hatası (deneme {attempt + 1}): {error_str[:100]}...")
                if attempt < cfg.max_retries - 1:
                    time.sleep(cfg.retry_delay)
                    continue
    
    raise last_exception


# Enhanced Compliance Extractor
class ComplianceExtractorEnhanced:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.setup_client()
        self.good_examples = []  # For few-shot learning
        self.pattern_stats = {pattern: 0 for pattern in COMPLIANCE_PATTERNS.keys()}
        
    def setup_client(self):
        """API client'ı ayarla"""
        if self.cfg.model_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable gerekli!")
            self.client = openai.OpenAI(api_key=api_key)
            
        elif self.cfg.model_provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ValueError("pip install google-genai gerekli!")
            api_key = os.getenv("GOOGLE_API_KEY") 
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable gerekli!")
            # Yeni Google Genai SDK
            os.environ["GEMINI_API_KEY"] = api_key
            self.client = genai.Client()
            model_name = AVAILABLE_MODELS["google"][self.cfg.model_name]
            self.model_name = model_name
            
        elif self.cfg.model_provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("pip install anthropic gerekli!")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable gerekli!")
            self.client = anthropic.Anthropic(api_key=api_key)
            
        elif self.cfg.model_provider == "nebius":
            api_key = os.getenv("NEBIUS_API_KEY")
            if not api_key:
                raise ValueError("NEBIUS_API_KEY environment variable gerekli!")
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.studio.nebius.ai/v1/"
            )
            
        print(f"Model: {self.cfg.model_provider}:{self.cfg.model_name}")

    def match_compliance_patterns(self, cu: Dict[str, Any]) -> Tuple[List[str], str]:
        """Match compliance patterns and determine complexity"""
        constraint_text = cu.get('constraint', '').lower()
        condition_text = cu.get('condition', '').lower()
        full_text = f"{constraint_text} {condition_text}"
        
        matched_patterns = []
        complexity_scores = []
        
        for pattern_name, pattern_data in COMPLIANCE_PATTERNS.items():
            keywords = pattern_data["keywords"]
            if any(keyword in full_text for keyword in keywords):
                matched_patterns.append(pattern_name)
                complexity_scores.append(pattern_data["complexity"])
        
        # Determine overall difficulty
        if not complexity_scores:
            difficulty = "simple"
        elif "difficult" in complexity_scores:
            difficulty = "difficult"
        elif "medium" in complexity_scores:
            difficulty = "medium"
        else:
            difficulty = "simple"
        
        return matched_patterns, difficulty

    def create_few_shot_examples(self) -> str:
        """Create few-shot examples from good CUs following paper structure"""
        if not self.good_examples or not self.cfg.use_few_shot:
            return self._get_default_paper_examples()
        
        examples = []
        for i, example in enumerate(self.good_examples[:self.cfg.max_examples], 1):
            examples.append(f"""
PAPER-STYLE ÖRNEK {i} (Compliance-to-Code standardı):
SUBJECT (Özne): {example.get('subject', '')}
CONDITION (Koşul): {example.get('condition', '')}
CONSTRAINT (Kısıtlama): {example.get('constraint', '')}
CONTEXTUAL INFO: {example.get('contextual_info', '')}
EXECUTABLE: {example.get('is_executable', False)}
DIFFICULTY: {example.get('difficulty_level', '')}
PATTERNS: {', '.join(example.get('matched_patterns', []))}
REASONING: {'; '.join(example.get('reasoning_steps', [])[:3])}
""")
        
        return "\n".join(examples)
    
    def _get_default_paper_examples(self) -> str:
        """Default examples following Compliance-to-Code paper structure"""
        return """
PAPER-STYLE ÖRNEK 1 (Compliance-to-Code standardı):
SUBJECT (Özne): Sermayesi paylara bölünmüş komandit şirketler
CONDITION (Koşul): Şirketin halka açık olması ve %5 eşiğini aştığında
CONSTRAINT (Kısıtlama): Kamuya açıklama yapması zorunludur
CONTEXTUAL INFO: SPK düzenlemelerine göre, 3 işgünü içinde
EXECUTABLE: True
DIFFICULTY: medium
PATTERNS: threshold_check, disclosure
REASONING: Eşik kontrolü yapılabilir; DataFrame ile açıklama durumu takip edilebilir

PAPER-STYLE ÖRNEK 2 (Compliance-to-Code standardı):  
SUBJECT (Özne): Yönetim kurulu üyeleri
CONDITION (Koşul): Karar alma toplantılarında çıkar çatışması durumunda
CONSTRAINT (Kısıtlama): Oylamaya katılamaz ve toplantıdan çıkmalıdır
CONTEXTUAL INFO: TTK 393. madde, çıkar çatışması tanımı için bakınız
EXECUTABLE: True
DIFFICULTY: simple
PATTERNS: prohibition, conditional_obligation
REASONING: Boolean logic ile çıkar çatışması kontrolü; oylama katılım kontrolü
"""

    def extract_cus_with_cot(self, text: str, kanun_no: str, madde_no: str) -> List[Dict[str, Any]]:
        """Enhanced CU extraction with CoT reasoning - following the paper"""
        
        if not text.strip() or len(text) < 50:
            return []
        
        few_shot_examples = self.create_few_shot_examples()
        
        # Enhanced prompt following the "Compliance-to-Code" paper methodology (arXiv:2505.19804)
        cu_prompt = f"""
Sen financial compliance uzmanı bir AI'sın. "Compliance-to-Code: Enhancing Financial Compliance Checking via Code Generation" makalesindeki metodolojiye göre Türk düzenleyici metinleri Compliance Units (CU) olarak yapılandır.

{few_shot_examples}

{"=" * 80}
COMPLIANCE-TO-CODE METHODOLOGY - PAPER STANDARDı (arXiv:2505.19804):

TEMEL PRENSİPLER:
1. Her düzenleyici hüküm dört mantıksal bileşene ayrılır
2. Her CU yürütülebilir Python koduna dönüştürülebilir olmalıdır
3. Chain-of-Thought reasoning ile yapılandırılmış analiz
4. Validation logic için DataFrame operations kullanılabilmelidir

CHAIN-OF-THOUGHT REASONING - ADIM ADIM ANALİZ:

ADIM 1: ACTIONABLE HÜKÜM TESPİTİ
- Düzenleyici metinde hangi YÜRÜTÜLEBILIR (executable) yükümlülükler var?
- Modal verbs: gerekir, zorunda, yasaklanır, must, shall, yapmalı, etmeli, olmaz
- Sayısal thresholds: %X, Y gün, Z TL, maksimum, minimum, eşik değerleri
- Zaman constraints: tarih sınırları, süre kısıtları, dönemsel gereksinimler
- Onay/izin gereksinimleri: karar alınması, bildirimin yapılması

ADIM 2: DÖRT BİLEŞEN YAPISAL AYRIŞTIRMA (Paper'daki ana metodoloji)
Her actionable hüküm için:

A) SUBJECT (Özne) - "Kim?"
   - Düzenlemeye tabi olan taraf: şirket, yönetici, pay sahibi, kurul
   - Sorumlu birim: yönetim kurulu, denetim komitesi, genel kurul
   - Kapsamdaki varlıklar: belirli büyüklük, tür veya statüdeki kuruluşlar

B) CONDITION (Koşul) - "Ne zaman/hangi durumda?"
   - Tetikleme koşulları: belirli olaylar, eşik aşımları
   - Sayısal kriterler: finansal oranlar, tutarlar, yüzdeler
   - Durumsal koşullar: statü değişiklikleri, zaman kriterleri
   - Kombinasyonlar: birden fazla koşulun aynı anda sağlanması

C) CONSTRAINT (Kısıtlama) - "Ne yapılacak/yasaklanacak?"
   - Net eylem tanımı (maksimum 20 kelime)
   - Pozitif obligation: yapılması gereken eylem
   - Negatif prohibition: yasaklanan eylem  
   - Quantitative limits: sayısal sınırlamalar

D) CONTEXTUAL INFO (Bağlamsal Bilgi) - "Nasıl/ek bilgiler?"
   - Tanımlar ve açıklamalar
   - İstisnalar ve muafiyetler
   - Referans dokümanlar
   - Hesaplama metodolojileri

ADIM 3: CODE GENERATION FEASİBİLİTY KONTROLÜ
- Bu hüküm Python DataFrame operasyonları ile kontrol edilebilir mi?
- Input/output parametreleri açık olarak tanımlanabilir mi?
- Validation logic kurulabilir mi? (if-then-else, mathematical operations)
- Compliance violation detection mantığı yazılabilir mi?

ADIM 4: COMPLEXITY CLASSIFICATION (Paper standardı)
- SIMPLE: Tek koşul, basit threshold check, direct if-else logic
- MEDIUM: 2-3 koşul, mathematical calculations, date/time operations  
- DIFFICULT: Karmaşık boolean logic, multiple interdependencies, temporal sequencing

ADIM 5: INTER-UNIT RELATIONSHIP ANALYSIS
- refer_to: Bu CU başka CU'ya veya dokümana referans veriyor
- exclude: Bu CU başka CU'nun uygulamasını geçersiz kılıyor
- only_include: Sadece bu CU geçerli (exclusivity)
- should_include: Bu CU başka CU'nun uygulanmasını gerektiriyor (dependency)

PAPER'DAKİ KRİTİK GEREKLER:
✓ Sadece CODE'a çevrilebilir, EXECUTABLE kuralları çıkar
✓ Her CU'nun Python validation function'ı yazılabilir olmalı
✓ DataFrame operations ile compliance checking yapılabilmeli
✓ Test edilebilir input/output parametreleri olmalı

COMPLIANCE-TO-CODE JSON STANDARD (Paper formatı):
{{
  "reasoning_process": [
    "Adım 1: Actionable hüküm tespiti - X adet yürütülebilir kural bulundu",
    "Adım 2: Dört bileşen ayrıştırması - Subject/Condition/Constraint/ContextualInfo yapılandırıldı", 
    "Adım 3: Code generation feasibility - Python DataFrame operasyonları ile kontrol edilebilirlik değerlendirildi",
    "Adım 4: Complexity classification - Paper standardına göre zorluk seviyeleri belirlendi",
    "Adım 5: Inter-unit relationship analysis - CU'lar arası bağımlılıklar analiz edildi"
  ],
  "compliance_units": [
    {{
      "subject": "düzenlemeye tabi olan taraf/varlık (kim?)",
      "condition": "tetikleme koşulları, sayısal eşikler, durumsal kriterler (ne zaman?)", 
      "constraint": "net eylem tanımı - pozitif obligation veya negatif prohibition (ne yapılacak? max 20 kelime)",
      "contextual_info": "tanımlar, istisnalar, referans dokümanlar, hesaplama metodolojileri",
      "is_executable": true,
      "code_generation_feasible": true,
      "dataframe_operations_possible": true,
      "validation_logic_defined": true,
      "difficulty_level": "simple/medium/difficult",
      "matched_patterns": ["threshold_check", "disclosure", "prohibition"],
      "reasoning_steps": [
        "Paper analizi: Bu hüküm neden actionable ve executable",
        "Dört bileşen ayrıştırması: Subject-Condition-Constraint-ContextualInfo",
        "Code feasibility: Hangi DataFrame operations kullanılacak",
        "Validation logic: Compliance violation detection mantığı"
      ],
      "relations": [
        {{"type": "refer_to", "target_cu": "CU_X_Y_Z", "description": "paper standardı inter-unit relationship"}}
      ],
      "feasibility_score": 0.85,
      "paper_compliance_score": 0.90
    }}
  ]
}}

Kanun: {kanun_no}, Madde: {madde_no}
Metin: {text[:2500]}
        """
        
        try:
            if self.cfg.model_provider == "openai":
                def openai_api_call():
                    response = self.client.chat.completions.create(
                        model=AVAILABLE_MODELS["openai"][self.cfg.model_name],
                        messages=[
                            {"role": "system", "content": "Sen 'Compliance-to-Code: Enhancing Financial Compliance Checking via Code Generation' (arXiv:2505.19804) makalesindeki metodolojiye göre düzenleyici metinleri analiz eden uzman bir AI'sın. Her hükmü Subject-Condition-Constraint-ContextualInfo dört bileşenine ayır ve yürütülebilir Python koduna dönüştürülebilir Compliance Units çıkar. Chain-of-thought reasoning kullan ve sadece JSON formatında cevap ver."},
                            {"role": "user", "content": cu_prompt}
                        ],
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens
                    )
                    return response.choices[0].message.content
                
                content = call_api_with_retry(openai_api_call, self.cfg)
                
            elif self.cfg.model_provider == "google":
                def google_api_call():
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=cu_prompt
                    )
                    return response.text
                
                content = call_api_with_retry(google_api_call, self.cfg)
                
            elif self.cfg.model_provider == "anthropic":
                def anthropic_api_call():
                    response = self.client.messages.create(
                        model=AVAILABLE_MODELS["anthropic"][self.cfg.model_name],
                        max_tokens=self.cfg.max_tokens,
                        temperature=self.cfg.temperature,
                        messages=[
                            {"role": "user", "content": cu_prompt}
                        ]
                    )
                    return response.content[0].text
                
                content = call_api_with_retry(anthropic_api_call, self.cfg)
                
            elif self.cfg.model_provider == "nebius":
                def nebius_api_call():
                    response = self.client.chat.completions.create(
                        model=AVAILABLE_MODELS["nebius"][self.cfg.model_name],
                        messages=[
                            {"role": "system", "content": "Sen 'Compliance-to-Code: Enhancing Financial Compliance Checking via Code Generation' (arXiv:2505.19804) makalesindeki metodolojiye göre düzenleyici metinleri analiz eden uzman bir AI'sın. Her hükmü Subject-Condition-Constraint-ContextualInfo dört bileşenine ayır ve yürütülebilir Python koduna dönüştürülebilir Compliance Units çıkar."},
                            {"role": "user", "content": cu_prompt}
                        ],
                        max_tokens=self.cfg.max_tokens,
                        temperature=self.cfg.temperature
                    )
                    return response.choices[0].message.content
                
                content = call_api_with_retry(nebius_api_call, self.cfg)
            
            # JSON parse
            result = self._parse_json_response(content)
            
            # Process CUs
            processed_cus = []
            if "compliance_units" in result:
                reasoning_process = result.get("reasoning_process", [])
                
                for i, cu_data in enumerate(result["compliance_units"], 1):
                    # Match patterns and determine difficulty
                    matched_patterns, auto_difficulty = self.match_compliance_patterns(cu_data)
                    
                    cu = {
                        "cu_id": f"CU_{kanun_no}_{madde_no}_{i}",
                        "source_row": None,
                        "kanun_no": kanun_no,
                        "madde_no": madde_no,
                        "subject": str(cu_data.get("subject", ""))[:100],
                        "condition": str(cu_data.get("condition", ""))[:300],
                        "constraint": str(cu_data.get("constraint", ""))[:150],
                        "contextual_info": str(cu_data.get("contextual_info", ""))[:200],
                        "is_executable": bool(cu_data.get("is_executable", False)),
                        "code_generation_feasible": bool(cu_data.get("code_generation_feasible", cu_data.get("is_executable", False))),
                        "dataframe_operations_possible": bool(cu_data.get("dataframe_operations_possible", cu_data.get("is_executable", False))),
                        "validation_logic_defined": bool(cu_data.get("validation_logic_defined", cu_data.get("is_executable", False))),
                        "difficulty_level": cu_data.get("difficulty_level", auto_difficulty),
                        "matched_patterns": matched_patterns,
                        "reasoning_steps": cu_data.get("reasoning_steps", [])[:5],
                        "relations": cu_data.get("relations", [])[:3],
                        "feasibility_score": float(cu_data.get("feasibility_score", 0.0)),
                        "paper_compliance_score": float(cu_data.get("paper_compliance_score", cu_data.get("feasibility_score", 0.0))),
                        "reasoning_process": reasoning_process,
                        "extracted_at": datetime.now().isoformat(),
                        "model_used": f"{self.cfg.model_provider}:{self.cfg.model_name}",
                        "paper_methodology": "Compliance-to-Code (arXiv:2505.19804)"
                    }
                    
                    # Update pattern statistics
                    for pattern in matched_patterns:
                        if pattern in self.pattern_stats:
                            self.pattern_stats[pattern] += 1
                    
                    processed_cus.append(cu)
            
            return processed_cus
            
        except Exception as e:
            print(f"Enhanced CU extraction hatası: {str(e)[:100]}")
            return []

    def validate_cu_quality_enhanced(self, cu: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced quality validation following paper standards"""
        
        constraint = cu.get("constraint", "").lower()
        condition = cu.get("condition", "").lower()
        subject = cu.get("subject", "").lower()
        
        # Basic keyword analysis
        modal_verbs = ["gerekir", "zorunda", "yasaklanır", "must", "shall", "yapmalı", "etmeli", "olmaz"]
        quantitative_patterns = [r'\d+\s*(gün|ay|yıl|%|tl|euro|dolar|adet)', r'maksimum|minimum|en\s+(az|fazla)']
        temporal_patterns = ["tarih", "süre", "zaman", "dönem", "içinde", "kadar", "önce", "sonra"]
        
        has_modal = any(verb in constraint for verb in modal_verbs)
        has_quantitative = any(re.search(pattern, constraint + " " + condition) for pattern in quantitative_patterns)
        has_temporal = any(word in constraint + " " + condition for word in temporal_patterns)
        
        # Paper's complexity assessment
        difficulty = cu.get("difficulty_level", "simple")
        complexity_score = {"simple": 0.3, "medium": 0.6, "difficult": 0.9}.get(difficulty, 0.3)
        
        # Structure completeness (paper metric)
        structure_score = 0.0
        if cu.get("subject"): structure_score += 0.25
        if cu.get("condition"): structure_score += 0.25
        if cu.get("constraint"): structure_score += 0.25
        if cu.get("contextual_info"): structure_score += 0.25
        
        # Domain specificity (financial compliance terms)
        domain_terms = ["açıklama", "bildirim", "onay", "izin", "yasak", "oran", "limit", "eşik", "süre"]
        domain_score = sum(1 for term in domain_terms if term in constraint + " " + condition) / len(domain_terms)
        
        # Feasibility score (from LLM or calculated)
        feasibility_score = cu.get("feasibility_score", 0.0)
        if feasibility_score == 0.0:
            # Calculate if not provided
            feasibility_score = (
                (0.4 if has_modal else 0.0) +
                (0.3 if has_quantitative else 0.0) +
                (0.3 if has_temporal else 0.0)
            )
        
        # Overall actionable score (paper's main metric)
        actionable_score = min(1.0, (
            feasibility_score * 0.4 +
            structure_score * 0.2 +
            complexity_score * 0.2 +
            domain_score * 0.2
        ))
        
        # Paper compliance score (Compliance-to-Code specific)
        paper_compliance_score = min(1.0, (
            feasibility_score * 0.3 +
            structure_score * 0.3 +
            (1.0 if cu.get("code_generation_feasible", False) else 0.0) * 0.2 +
            (1.0 if cu.get("dataframe_operations_possible", False) else 0.0) * 0.2
        ))
        
        # Pass@1 equivalent (executable and high quality following paper standards)
        pass_at_1 = (
            cu.get("is_executable", False) and
            cu.get("code_generation_feasible", False) and
            actionable_score >= self.cfg.min_actionable_score and
            feasibility_score >= self.cfg.min_feasibility_score and
            paper_compliance_score >= 0.7
        )
        
        return {
            "is_actionable": actionable_score >= self.cfg.min_actionable_score,
            "actionable_score": actionable_score,
            "feasibility_score": feasibility_score,
            "paper_compliance_score": paper_compliance_score,
            "structure_score": structure_score,
            "complexity_score": complexity_score,
            "domain_score": domain_score,
            "has_modal_verbs": has_modal,
            "has_quantitative": has_quantitative,
            "has_temporal": has_temporal,
            "pass_at_1": pass_at_1,
            "reasoning": f"Paper: {paper_compliance_score:.2f}, Modal: {has_modal}, Quant: {has_quantitative}, Temp: {has_temporal}, Struct: {structure_score:.2f}",
            "model_used": f"{self.cfg.model_provider}:{self.cfg.model_name}"
        }
    
    def update_good_examples(self, cus: List[Dict[str, Any]]):
        """Update good examples for few-shot learning"""
        if not self.cfg.use_few_shot:
            return
            
        # Filter high-quality CUs
        good_cus = [
            cu for cu in cus 
            if cu.get("actionable_score", 0) >= 0.8 and 
               cu.get("feasibility_score", 0) >= 0.8 and
               cu.get("is_executable", False)
        ]
        
        # Add to examples, maintain max limit
        self.good_examples.extend(good_cus)
        
        # Sort by quality and keep only the best
        self.good_examples.sort(key=lambda x: x.get("actionable_score", 0), reverse=True)
        self.good_examples = self.good_examples[:self.cfg.max_examples * 2]  # Keep extra for variety

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Enhanced JSON parsing"""
        try:
            # Clean JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Try direct parse
            return json.loads(content)
        except:
            try:
                # Find JSON with regex
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        
        return {}

# -------------------------
# Paper Metrics Calculator
# -------------------------

def calculate_paper_metrics(cus: List[Dict[str, Any]], cfg: Config) -> Dict[str, float]:
    """Calculate metrics following the paper's evaluation"""
    
    if not cus:
        return {
            "total_cus": 0,
            "pass_at_1": 0.0,
            "pass_at_5": 0.0,  # Simulated
            "structural_completeness": 0.0,
            "executable_ratio": 0.0,
            "domain_specificity": 0.0,
            "avg_reasoning_steps": 0.0,
            "avg_actionable_score": 0.0,
            "avg_feasibility_score": 0.0,
            "difficulty_distribution": {},
            "pattern_distribution": {}
        }
    
    # Pass@1 - first attempt success rate
    pass_at_1_count = sum(1 for cu in cus if cu.get('pass_at_1', False))
    pass_at_1 = pass_at_1_count / len(cus)
    
    # Pass@5 simulation (slightly higher than Pass@1)
    pass_at_5 = min(1.0, pass_at_1 * 1.3)
    
    # Structural completeness
    complete_structure = sum(1 for cu in cus if all([
        cu.get('subject'), cu.get('condition'), cu.get('constraint')
    ])) / len(cus)
    
    # Executable ratio
    executable_ratio = sum(1 for cu in cus if cu.get('is_executable', False)) / len(cus)
    
    # Domain specificity
    domain_specificity = sum(cu.get('domain_score', 0) for cu in cus) / len(cus)
    
    # Difficulty distribution
    difficulty_dist = {}
    for level in cfg.difficulty_levels:
        difficulty_dist[level] = sum(1 for cu in cus if cu.get('difficulty_level') == level) / len(cus)
    
    # Pattern distribution
    all_patterns = []
    for cu in cus:
        all_patterns.extend(cu.get('matched_patterns', []))
    
    pattern_dist = {}
    for pattern in COMPLIANCE_PATTERNS.keys():
        pattern_dist[pattern] = all_patterns.count(pattern) / len(all_patterns) if all_patterns else 0.0
    
    return {
        "total_cus": len(cus),
        "pass_at_1": pass_at_1,
        "pass_at_5": pass_at_5,
        "structural_completeness": complete_structure,
        "executable_ratio": executable_ratio,
        "domain_specificity": domain_specificity,
        "avg_reasoning_steps": sum(len(cu.get('reasoning_steps', [])) for cu in cus) / len(cus),
        "avg_actionable_score": sum(cu.get('actionable_score', 0) for cu in cus) / len(cus),
        "avg_feasibility_score": sum(cu.get('feasibility_score', 0) for cu in cus) / len(cus),
        "difficulty_distribution": difficulty_dist,
        "pattern_distribution": pattern_dist
    }

# Enhanced Processing Functions
def extract_regulation_info(text: str, excel_kanun_no: str = None, excel_madde_no: str = None) -> Tuple[str, str]:
    """Enhanced regulation info extraction with sub-clause support
    
    Args:
        text: Text to extract from
        excel_kanun_no: Pre-extracted kanun number from Excel (optional) 
        excel_madde_no: Pre-extracted madde number from Excel (optional)
        
    Returns:
        Tuple of (kanun_no, madde_no) where madde_no may include sub-clause like "24/1"
    """

    kanun_patterns = [
        r'(\d+)\s*sayılı',
        r'Kanun\s*No[:\.\s]*(\d+)',
        r'(\d+)\s*no\'?lu',
    ]
    
    madde_patterns = [
        r'[Mm]adde\s*[:\.\s]*(\d+)\s*/?\s*\(\s*(\d+)\s*\)',
        r'[Mm]adde\s*[:\.\s]*(\d+)',
        r'md\.\s*(\d+)\s*/?\s*\(\s*(\d+)\s*\)',
        r'md\.\s*(\d+)',
        r'Art\.\s*(\d+)\s*/?\s*\(\s*(\d+)\s*\)',
        r'Art\.\s*(\d+)',
    ]
    
    sub_clause_patterns = [
        r'(?:birinci|ikinci|üçüncü|dördüncü|beşinci|altıncı|yedinci|sekizinci|dokuzuncu|onuncu)\s*fıkra',  
        r'(\d+)\s*\.\s*fıkra',  
        r'(\d+)\s*(?:üncü|inci|uncu|nci)\s*fıkra',  
        r'\(\s*(\d+)\s*\)',  
    ]
    
    ordinal_map = {
        'birinci': '1', 'ikinci': '2', 'üçüncü': '3', 'dördüncü': '4', 'beşinci': '5',
        'altıncı': '6', 'yedinci': '7', 'sekizinci': '8', 'dokuzuncu': '9', 'onuncu': '10'
    }
    
    if excel_kanun_no and str(excel_kanun_no) != 'nan':
        kanun_no = str(excel_kanun_no)
    else:
        kanun_no = "UNK"
        for pattern in kanun_patterns:
            match = re.search(pattern, text)
            if match:
                kanun_no = match.group(1)
                break
    
    if excel_madde_no and str(excel_madde_no) != 'nan':
        main_article = str(excel_madde_no)
    else:
        main_article = "1"
    
    madde_no = main_article  
    
    found_explicit = False
    for pattern in madde_patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) >= 2 and match.group(2):
                madde_no = f"{match.group(1)}/{match.group(2)}"
                found_explicit = True
                break
    
    if not found_explicit:
        sub_clause_found = False
        for sub_pattern in sub_clause_patterns:
            sub_match = re.search(sub_pattern, text)
            if sub_match:
                if 'fıkra' in sub_pattern and not sub_match.group(0).replace(' ', '').isdigit():
                    for ordinal, number in ordinal_map.items():
                        if ordinal in sub_match.group(0):
                            madde_no = f"{main_article}/{number}"
                            sub_clause_found = True
                            break
                else:
                    if len(sub_match.groups()) > 0:
                        sub_clause_num = sub_match.group(1)
                        if sub_clause_num and sub_clause_num.isdigit() and 1 <= int(sub_clause_num) <= 20:
                            madde_no = f"{main_article}/{sub_clause_num}"
                            sub_clause_found = True
                            break
                if sub_clause_found:
                    break
    
    return kanun_no, madde_no

def validate_cu_structure_enhanced(cu: Dict[str, Any], cfg: Config) -> Dict[str, bool]:
    """Enhanced CU structure validation"""
    constraint_words = len(cu.get("constraint", "").split())
    
    return {
        "has_subject": bool(cu.get("subject", "").strip()),
        "has_condition": bool(cu.get("condition", "").strip()),
        "has_constraint": bool(cu.get("constraint", "").strip()),
        "has_contextual_info": bool(cu.get("contextual_info", "").strip()),
        "constraint_length_ok": cfg.min_constraint_words <= constraint_words <= cfg.max_constraint_words,
        "has_reasoning": len(cu.get("reasoning_steps", [])) >= 1,
        "has_patterns": len(cu.get("matched_patterns", [])) >= 1,
        "has_relations": len(cu.get("relations", [])) >= 0,  # Optional
        "is_executable": cu.get("is_executable", False),
        "feasibility_ok": cu.get("feasibility_score", 0) >= cfg.min_feasibility_score,
        "actionable_ok": cu.get("actionable_score", 0) >= cfg.min_actionable_score
    }

def process_excel_enhanced(cfg: Config) -> pd.DataFrame:
    """Enhanced Excel processing with paper methodology"""
    
    extractor = ComplianceExtractorEnhanced(cfg)
    
    print(f"Excel dosyası okunuyor: {cfg.input_excel}")
    df = pd.read_excel(cfg.input_excel, sheet_name=cfg.sheet)
    df = df.reset_index(drop=True)   
    
    valid_rows = []
    for idx, row in df.iterrows():
        text = " | ".join([str(row[col]) for col in df.columns if pd.notna(row[col])])
        if len(text) < 50:
            print(f"Uyarı: Çok kısa madde (len={len(text)}). Yine de işleniyor.")
        excel_kanun = row.get('Kanun_Numarasi', None)
        excel_madde = row.get('Madde_Numarasi', None)
        valid_rows.append((idx, text, excel_kanun, excel_madde))
    
    print(f"İşlenecek satır sayısı: {len(valid_rows)}")
    
    all_cus = []
    processed_count = 0
    batch_results = []
    
    with tqdm(total=len(valid_rows), desc=f"🔄 Enhanced CU Extraction ({cfg.model_provider})", unit="row") as pbar:
        
        for batch_start in range(0, len(valid_rows), cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, len(valid_rows))
            batch = valid_rows[batch_start:batch_end]
            
            batch_cus = []
            
            for idx, text, excel_kanun, excel_madde in batch:
                    
                kanun_no, madde_no = extract_regulation_info(text, excel_kanun, excel_madde)
                
                try:
                    cus = extractor.extract_cus_with_cot(text, kanun_no, madde_no)
                    
                    if cus:
                        for cu in cus:
                            cu["source_row"] = idx
                            quality = extractor.validate_cu_quality_enhanced(cu)
                            cu.update(quality)
                        
                        batch_cus.extend(cus)
                    
                    processed_count += 1
                    pbar.update(1)
                    
                    if cus:
                        avg_score = sum(cu.get('actionable_score', 0) for cu in cus) / len(cus)
                        pbar.set_postfix({
                            "CUs": len(cus),
                            "Total": len(all_cus) + len(batch_cus),
                            "Avg Score": f"{avg_score:.2f}",
                            "Pass@1": sum(1 for cu in cus if cu.get('pass_at_1', False))
                        })
                    
                except Exception as e:
                    print(f"Satır {idx+1} hatası: {str(e)[:50]}")
                    pbar.update(1)
                    continue
            
            if batch_cus:
                extractor.update_good_examples(batch_cus)
                all_cus.extend(batch_cus)
                batch_results.append(len(batch_cus))
            
            if batch_cus:
                batch_metrics = calculate_paper_metrics(batch_cus, cfg)
                print(f"\n Batch {len(batch_results)} Metrics:")
                print(f"   CUs: {len(batch_cus)}, Pass@1: {batch_metrics['pass_at_1']:.2%}")
    
    # Final pattern statistics
    print(f"\n Pattern İstatistikleri:")
    for pattern, count in extractor.pattern_stats.items():
        if count > 0:
            print(f"   {pattern}: {count}")
    
    return pd.DataFrame(all_cus)

def generate_enhanced_quality_report(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Generate comprehensive quality report following paper metrics"""
    if df.empty:
        return pd.DataFrame()
    
    quality_data = []
    
    for _, cu in df.iterrows():
        validations = validate_cu_structure_enhanced(cu.to_dict(), cfg)
        
        quality_data.append({
            "cu_id": cu.get("cu_id"),
            "source_row": cu.get("source_row"),
            "kanun_no": cu.get("kanun_no"),
            "madde_no": cu.get("madde_no"),
            "model_used": cu.get("model_used"),
            
            "subject": str(cu.get("subject", ""))[:50],
            "condition": str(cu.get("condition", ""))[:50],
            "constraint": str(cu.get("constraint", ""))[:50],
            "contextual_info": str(cu.get("contextual_info", ""))[:50],
            
            "difficulty_level": cu.get("difficulty_level"),
            "matched_patterns": ", ".join(cu.get("matched_patterns", [])),
            "relations_count": len(cu.get("relations", [])),
            
            "constraint_word_count": len(str(cu.get("constraint", "")).split()),
            "reasoning_steps_count": len(cu.get("reasoning_steps", [])),
            
            "actionable_score": cu.get("actionable_score", 0.0),
            "feasibility_score": cu.get("feasibility_score", 0.0),
            "paper_compliance_score": cu.get("paper_compliance_score", 0.0),
            "structure_score": cu.get("structure_score", 0.0),
            "complexity_score": cu.get("complexity_score", 0.0),
            "domain_score": cu.get("domain_score", 0.0),
            
            "is_actionable": cu.get("is_actionable", False),
            "is_executable": cu.get("is_executable", False),
            "code_generation_feasible": cu.get("code_generation_feasible", False),
            "dataframe_operations_possible": cu.get("dataframe_operations_possible", False),
            "validation_logic_defined": cu.get("validation_logic_defined", False),
            "pass_at_1": cu.get("pass_at_1", False),
            "has_modal_verbs": cu.get("has_modal_verbs", False),
            "has_quantitative": cu.get("has_quantitative", False),
            "has_temporal": cu.get("has_temporal", False),
            
            "structure_valid": all([
                validations.get("has_subject", False),
                validations.get("has_constraint", False)
            ])
        })
    
    return pd.DataFrame(quality_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file path")
    args = parser.parse_args()
    
    cfg = read_config(args.config)
    
    print(" Enhanced Compliance-to-Code çıkarımı başlıyor...")
    print(f"    Model: {cfg.model_provider}:{cfg.model_name}")
    print(f"    Excel: {cfg.input_excel}")
    print(f"    Features: CoT={cfg.use_cot_reasoning}, Few-shot={cfg.use_few_shot}")
    print(f"    Thresholds: Actionable≥{cfg.min_actionable_score}, Feasibility≥{cfg.min_feasibility_score}")
    print()
    
    start_time = datetime.now()
    
    results_df = process_excel_enhanced(cfg)
    
    if len(results_df) == 0:
        print(" Hiç CU çıkarılamadı!")
        return
    
    if "is_actionable" in results_df.columns:
        actionable_df = results_df[
            (results_df["is_actionable"] == True) & 
            (results_df["actionable_score"] >= cfg.min_actionable_score)
        ].copy()
        
        pass_at_1_df = results_df[
            results_df["pass_at_1"] == True
        ].copy()
    else:
        actionable_df = results_df.copy()
        pass_at_1_df = results_df.copy()
    
    paper_metrics = calculate_paper_metrics(results_df.to_dict('records'), cfg)
    
    results_df.to_csv(cfg.output_csv, index=False, encoding="utf-8")
    
    quality_df = generate_enhanced_quality_report(results_df, cfg)
    quality_df.to_csv(cfg.quality_report_csv, index=False, encoding="utf-8")
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    print(f"\n COMPLIANCE-TO-CODE SONUÇLAR (arXiv:2505.19804 Standardı):")
    print(f"   Model: {cfg.model_provider}:{cfg.model_name}")
    print(f"   Toplam CU: {len(results_df)}")
    print(f"   Actionable CU: {len(actionable_df)} ({len(actionable_df)/len(results_df):.1%})")
    print(f"   Pass@1 CU: {len(pass_at_1_df)} ({paper_metrics['pass_at_1']:.1%})")
    print(f"   Pass@5 (sim): {paper_metrics['pass_at_5']:.1%}")
    if "paper_compliance_score" in results_df.columns:
        avg_paper_score = results_df["paper_compliance_score"].mean()
        print(f"   Avg Paper Compliance: {avg_paper_score:.2f}")
    if "code_generation_feasible" in results_df.columns:
        code_gen_ratio = results_df["code_generation_feasible"].sum() / len(results_df)
        print(f"   Code Generation Feasible: {code_gen_ratio:.1%}")
    
    print(f"\n PAPER METRICS:")
    print(f"   Executable Ratio: {paper_metrics['executable_ratio']:.1%}")
    print(f"   Structural Completeness: {paper_metrics['structural_completeness']:.1%}")
    print(f"   Domain Specificity: {paper_metrics['domain_specificity']:.2f}")
    print(f"   Avg Actionable Score: {paper_metrics['avg_actionable_score']:.2f}")
    print(f"   Avg Feasibility Score: {paper_metrics['avg_feasibility_score']:.2f}")
    print(f"   Avg Reasoning Steps: {paper_metrics['avg_reasoning_steps']:.1f}")
    
    print(f"\n ZORLUK DAĞILIMI:")
    for level, ratio in paper_metrics['difficulty_distribution'].items():
        print(f"   {level.capitalize()}: {ratio:.1%}")
    
    print(f"\n TOP PATTERN MATCHES:")
    sorted_patterns = sorted(paper_metrics['pattern_distribution'].items(), key=lambda x: x[1], reverse=True)
    for pattern, ratio in sorted_patterns[:5]:
        if ratio > 0:
            print(f"   {pattern}: {ratio:.1%}")
    
    print(f"\n Toplam süre: {elapsed:.1f} dakika")
    
    print(f"\n DOSYALAR KAYDEDİLDİ:")
    print(f"   Ana CSV: {cfg.output_csv}")
    print(f"   Kalite raporu: {cfg.quality_report_csv}")

if __name__ == "__main__":
    main()