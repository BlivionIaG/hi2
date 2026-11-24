# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Authored multilingual tasks: 14 languages, closed-form answers and translation refs.

The references are short, ordinary sentences written for this harness, so they are
guaranteed to be translations of the English source (the cached FLORES-200 copy on
the box is gated, and OPUS-100 test pairs are automatically mined and include
misaligned rows).  Closed-form answers are numbers or single colour words, which
gives exact-match scoring; the translation references give chrF plus a language-ID
check that the reply is not English.

Languages cover Latin, Cyrillic, Devanagari, Arabic and CJK scripts.
"""

LANGUAGES = [
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("zh", "Simplified Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
    ("pl", "Polish"),
    ("nl", "Dutch"),
]

TRANSLATIONS = [
    {
        "id": "train-seven",
        "en": "The train leaves at seven in the morning.",
        "refs": {
            "de": "Der Zug fährt um sieben Uhr morgens ab.",
            "fr": "Le train part à sept heures du matin.",
            "es": "El tren sale a las siete de la mañana.",
            "pt": "O trem parte às sete da manhã.",
            "ru": "Поезд отправляется в семь часов утра.",
            "zh": "火车在早上七点出发。",
            "ja": "列車は朝七時に出発します。",
            "ko": "기차는 아침 일곱 시에 출발합니다.",
            "ar": "يغادر القطار في الساعة السابعة صباحًا.",
            "hi": "ट्रेन सुबह सात बजे निकलती है।",
            "tr": "Tren sabah yedide kalkıyor.",
            "vi": "Tàu khởi hành lúc bảy giờ sáng.",
            "pl": "Pociąg odjeżdża o siódmej rano.",
            "nl": "De trein vertrekt om zeven uur 's ochtends.",
        },
    },
    {
        "id": "book-evening",
        "en": "She reads a book every evening before dinner.",
        "refs": {
            "de": "Sie liest jeden Abend vor dem Abendessen ein Buch.",
            "fr": "Elle lit un livre chaque soir avant le dîner.",
            "es": "Ella lee un libro cada noche antes de la cena.",
            "pt": "Ela lê um livro todas as noites antes do jantar.",
            "ru": "Она читает книгу каждый вечер перед ужином.",
            "zh": "她每天晚饭前都会读一本书。",
            "ja": "彼女は毎晩夕食の前に本を読みます。",
            "ko": "그녀는 저녁 식사 전에 매일 책을 읽습니다.",
            "ar": "تقرأ كتابًا كل مساء قبل العشاء.",
            "hi": "वह हर शाम रात के खाने से पहले एक किताब पढ़ती है।",
            "tr": "Her akşam yemekten önce bir kitap okur.",
            "vi": "Cô ấy đọc sách mỗi tối trước bữa ăn tối.",
            "pl": "Czyta książkę każdego wieczoru przed kolacją.",
            "nl": "Zij leest elke avond voor het avondeten een boek.",
        },
    },
]

QA = [
    {
        "id": "week-days",
        "kind": "numeric",
        "answer": "7",
        "prompts": {
            "de": "Wie viele Tage hat eine Woche? Antworte mit einer einzigen Zahl.",
            "fr": "Combien de jours compte une semaine ? Réponds avec un seul nombre.",
            "es": "¿Cuántos días tiene una semana? Responde con un solo número.",
            "pt": "Quantos dias tem uma semana? Responda com um único número.",
            "ru": "Сколько дней в неделе? Ответь одним числом.",
            "zh": "一周有几天？只回答一个数字。",
            "ja": "一週間は何日ですか？数字だけで答えてください。",
            "ko": "일주일은 며칠입니까? 숫자만 답하세요.",
            "ar": "كم عدد أيام الأسبوع؟ أجب برقم واحد فقط.",
            "hi": "एक सप्ताह में कितने दिन होते हैं? केवल एक संख्या में उत्तर दें।",
            "tr": "Bir haftada kaç gün vardır? Yalnızca bir sayı ile yanıtla.",
            "vi": "Một tuần có bao nhiêu ngày? Chỉ trả lời bằng một con số.",
            "pl": "Ile dni ma tydzień? Odpowiedz jedną liczbą.",
            "nl": "Hoeveel dagen heeft een week? Antwoord met één getal.",
        },
    },
    {
        "id": "nine-times-eight",
        "kind": "numeric",
        "answer": "72",
        "prompts": {
            "de": "Was ist neun mal acht? Antworte mit einer einzigen Zahl.",
            "fr": "Combien font neuf fois huit ? Réponds avec un seul nombre.",
            "es": "¿Cuánto es nueve por ocho? Responde con un solo número.",
            "pt": "Quanto é nove vezes oito? Responda com um único número.",
            "ru": "Сколько будет девять умножить на восемь? Ответь одним числом.",
            "zh": "九乘以八等于多少？只回答一个数字。",
            "ja": "九かける八はいくつですか？数字だけで答えてください。",
            "ko": "구 곱하기 팔은 얼마입니까? 숫자만 답하세요.",
            "ar": "كم يساوي تسعة في ثمانية؟ أجب برقم واحد فقط.",
            "hi": "नौ गुणा आठ कितना होता है? केवल एक संख्या में उत्तर दें।",
            "tr": "Dokuz çarpı sekiz kaç eder? Yalnızca bir sayı ile yanıtla.",
            "vi": "Chín nhân tám bằng bao nhiêu? Chỉ trả lời bằng một con số.",
            "pl": "Ile to jest dziewięć razy osiem? Odpowiedz jedną liczbą.",
            "nl": "Hoeveel is negen keer acht? Antwoord met één getal.",
        },
    },
    {
        "id": "sky-color",
        "kind": "word",
        "answers": {
            "de": ["blau"],
            "fr": ["bleu"],
            "es": ["azul"],
            "pt": ["azul"],
            "ru": ["голубой", "синий", "голубое"],
            "zh": ["蓝色", "蓝"],
            "ja": ["青", "青色"],
            "ko": ["파란색", "파랑", "하늘색"],
            "ar": ["أزرق", "زرقاء", "سماوي"],
            "hi": ["नीला", "नीली"],
            "tr": ["mavi"],
            "vi": ["xanh", "xanh dương"],
            "pl": ["niebieski", "niebieskie", "błękitny"],
            "nl": ["blauw"],
        },
        "prompts": {
            "de": "Welche Farbe hat ein klarer Himmel am Tag? Antworte mit einem Wort.",
            "fr": "De quelle couleur est un ciel clair le jour ? Réponds avec un mot.",
            "es": "¿De qué color es un cielo despejado de día? Responde con una palabra.",
            "pt": "De que cor é um céu limpo durante o dia? Responda com uma palavra.",
            "ru": "Какого цвета ясное небо днём? Ответь одним словом.",
            "zh": "晴朗的白天的天空是什么颜色？用一个词回答。",
            "ja": "晴れた昼間の空は何色ですか？一語で答えてください。",
            "ko": "맑은 낮 하늘은 무슨 색입니까? 한 단어로 답하세요.",
            "ar": "ما لون السماء الصافية في النهار؟ أجب بكلمة واحدة.",
            "hi": "साफ दिन के आसमान का रंग क्या है? एक शब्द में उत्तर दें।",
            "tr": "Açık bir gündüz gökyüzü ne renktir? Tek kelimeyle yanıtla.",
            "vi": "Bầu trời quang đãng vào ban ngày có màu gì? Trả lời bằng một từ.",
            "pl": "Jakiego koloru jest czyste niebo w dzień? Odpowiedz jednym słowem.",
            "nl": "Welke kleur heeft een heldere hemel overdag? Antwoord met één woord.",
        },
    },
]

SOURCE = {
    "name": "inkling-eval authored multilingual tasks",
    "license": "Apache-2.0",
    "note": "Sentences and questions written for this harness; no external translation corpus.",
}
