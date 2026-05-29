from __future__ import annotations

import math
import re
from collections import Counter

# Регулярные выражения намеренно простые:
# проект работает с английским текстом и не пытается быть универсальным NLP-парсером.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")
MODEL_EXCLUDED_FEATURES = {"word_count", "sentence_count"}
SHORT_TEXT_RATIO_DENOMINATOR = 8

# Ниже собраны компактные словари-маркеры.
# Это не полноценные CEFR-лексиконы, а инженерные эвристики,
# которые помогают модели отличать базовый, промежуточный и продвинутый регистр.
SUBORDINATORS = {
    "although",
    "because",
    "before",
    "despite",
    "if",
    "since",
    "unless",
    "whereas",
    "while",
}
DISCOURSE_CONNECTORS = {
    "besides",
    "consequently",
    "furthermore",
    "however",
    "moreover",
    "nevertheless",
    "therefore",
    "thus",
}
MODAL_VERBS = {
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
}
ACADEMIC_WORDS = {
    "abstract",
    "analysis",
    "approach",
    "assessment",
    "automation",
    "challenge",
    "concept",
    "consequently",
    "contemporary",
    "conventional",
    "context",
    "derive",
    "disparity",
    "economic",
    "efficiency",
    "evidence",
    "exacerbate",
    "expertise",
    "framework",
    "function",
    "interpretation",
    "method",
    "paradigm",
    "policy",
    "process",
    "proliferation",
    "proponent",
    "research",
    "socioeconomic",
    "significant",
    "structure",
    "theory",
    "unprecedented",
}
ADVANCED_DOCUMENT_WORDS = {
    "accountability",
    "aesthetic",
    "alternatives",
    "architecture",
    "armour",
    "assailed",
    "astrologers",
    "baroque",
    "benign",
    "blinkers",
    "broader",
    "candlesticks",
    "cathedral",
    "ceased",
    "charitable",
    "civic",
    "classical",
    "cliched",
    "conservative",
    "craftsmen",
    "decorative",
    "democratic",
    "dependency",
    "disproportionate",
    "donors",
    "dynamic",
    "equitable",
    "erode",
    "exerted",
    "fascinated",
    "feted",
    "forged",
    "generosity",
    "graceful",
    "imagination",
    "immovable",
    "innocuous",
    "institutions",
    "interior",
    "ironwork",
    "jeweller",
    "landscape",
    "liberated",
    "malleable",
    "manufacturers",
    "medium",
    "mismanagement",
    "oversight",
    "perpetuate",
    "philanthropy",
    "philanthropic",
    "practitioners",
    "priorities",
    "privileged",
    "profound",
    "promotion",
    "rigid",
    "sector",
    "seemingly",
    "sinuous",
    "spiral",
    "structural",
    "substitute",
    "systemic",
    "transparency",
    "unintended",
    "visual",
    "wares",
    "wrought",
}
CEFR_BASIC_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "always",
    "am",
    "an",
    "and",
    "animal",
    "are",
    "at",
    "baby",
    "bad",
    "be",
    "beautiful",
    "because",
    "bed",
    "big",
    "bike",
    "book",
    "bring",
    "brother",
    "but",
    "by",
    "call",
    "can",
    "cat",
    "city",
    "come",
    "day",
    "do",
    "dog",
    "drink",
    "eat",
    "every",
    "family",
    "father",
    "fish",
    "food",
    "for",
    "friend",
    "from",
    "give",
    "go",
    "good",
    "had",
    "happy",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "home",
    "house",
    "i",
    "in",
    "is",
    "it",
    "know",
    "like",
    "little",
    "live",
    "love",
    "make",
    "man",
    "many",
    "milk",
    "mother",
    "my",
    "name",
    "new",
    "nice",
    "not",
    "of",
    "old",
    "on",
    "one",
    "or",
    "people",
    "play",
    "read",
    "school",
    "see",
    "she",
    "sleep",
    "small",
    "soft",
    "some",
    "sun",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "time",
    "to",
    "today",
    "train",
    "travel",
    "two",
    "umbrella",
    "up",
    "very",
    "walk",
    "want",
    "was",
    "water",
    "we",
    "what",
    "when",
    "where",
    "who",
    "with",
    "woman",
    "work",
    "you",
    "your",
}
CEFR_INTERMEDIATE_WORDS = {
    "airport",
    "although",
    "besides",
    "changing",
    "enjoy",
    "faster",
    "journey",
    "landscape",
    "less",
    "personally",
    "prefer",
    "relax",
    "security",
    "stressful",
    "worry",
}
NOMINALIZATION_SUFFIXES = (
    "tion",
    "sion",
    "ment",
    "ity",
    "ness",
    "ance",
    "ence",
)
NOMINALIZATION_EXCEPTIONS = {
    "fashion",
    "question",
    "television",
    "version",
}


def tokenize(text: str) -> list[str]:
    """Разбивает текст на токены, сохраняя простые английские сокращения."""

    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def sentence_count(text: str) -> int:
    """Грубо оценивает число предложений по знакам конца фразы."""

    chunks = [chunk for chunk in SENTENCE_RE.split(text) if chunk.strip()]
    return max(1, len(chunks))


def count_syllables(word: str) -> int:
    """Приближённо считает слоги для readability-метрик."""

    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def smoothed_short_ratio(count: int, words: int) -> float:
    """Сглаживает доли на коротких текстах, чтобы единичные слова не искажали признак."""

    return count / max(words, SHORT_TEXT_RATIO_DENOMINATOR)


def short_text_reliability(words: int) -> float:
    """Оценивает, насколько вообще можно доверять долям на коротком тексте."""

    return min(1.0, words / SHORT_TEXT_RATIO_DENOMINATOR)


def readability_features(text: str) -> dict[str, float]:
    """Строит набор интерпретируемых признаков для модели и ответа API."""

    tokens = tokenize(text)
    words = max(1, len(tokens))
    sentences = sentence_count(text)
    chars = sum(len(token) for token in tokens)
    syllables = sum(count_syllables(token) for token in tokens)
    unique_words = len(set(tokens))
    long_words = sum(1 for token in tokens if len(token) >= 7)
    very_long_words = sum(1 for token in tokens if len(token) >= 10)
    subordinator_count = sum(1 for token in tokens if token in SUBORDINATORS)
    connector_count = sum(1 for token in tokens if token in DISCOURSE_CONNECTORS)
    modal_count = sum(1 for token in tokens if token in MODAL_VERBS)
    academic_count = sum(1 for token in tokens if token in ACADEMIC_WORDS)
    advanced_document_count = sum(1 for token in tokens if token in ADVANCED_DOCUMENT_WORDS)
    basic_cefr_count = sum(1 for token in tokens if token in CEFR_BASIC_WORDS)
    intermediate_cefr_count = sum(1 for token in tokens if token in CEFR_INTERMEDIATE_WORDS)
    nominalization_count = sum(
        1
        for token in tokens
        if (
            len(token) > 6
            and token not in NOMINALIZATION_EXCEPTIONS
            and token.endswith(NOMINALIZATION_SUFFIXES)
        )
    )
    punctuation_count = len(re.findall(r"[,;:()]", text))

    # Базовые количественные признаки удобнее считать один раз,
    # а затем переиспользовать и в модели, и в диагностике.
    avg_sentence_length = words / sentences
    avg_word_length = chars / words
    syllables_per_word = syllables / words
    lexical_diversity = unique_words / words
    long_word_ratio = smoothed_short_ratio(long_words, words)
    very_long_word_ratio = smoothed_short_ratio(very_long_words, words)
    subordinator_ratio = smoothed_short_ratio(subordinator_count, words)
    connector_ratio = smoothed_short_ratio(connector_count, words)
    modal_ratio = smoothed_short_ratio(modal_count, words)
    academic_word_ratio = smoothed_short_ratio(academic_count, words)
    advanced_document_word_ratio = smoothed_short_ratio(advanced_document_count, words)
    basic_cefr_word_ratio = basic_cefr_count / words
    intermediate_cefr_word_ratio = smoothed_short_ratio(intermediate_cefr_count, words)
    out_of_basic_cefr_ratio = (1.0 - basic_cefr_word_ratio) * short_text_reliability(words)
    nominalization_ratio = smoothed_short_ratio(nominalization_count, words)
    punctuation_density = smoothed_short_ratio(punctuation_count, words)

    # Индекс формальности не является академической метрикой;
    # это агрегатный инженерный сигнал для более официального и плотного стиля.
    formality_index = (
        0.45 * long_word_ratio
        + 0.25 * nominalization_ratio
        + 0.20 * academic_word_ratio
        + 0.20 * advanced_document_word_ratio
        + 0.10 * punctuation_density
    )
    flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * syllables_per_word

    return {
        "word_count": float(words),
        "sentence_count": float(sentences),
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "syllables_per_word": syllables_per_word,
        "lexical_diversity": lexical_diversity,
        "long_word_ratio": long_word_ratio,
        "very_long_word_ratio": very_long_word_ratio,
        "subordinator_ratio": subordinator_ratio,
        "connector_ratio": connector_ratio,
        "modal_ratio": modal_ratio,
        "academic_word_ratio": academic_word_ratio,
        "advanced_document_word_ratio": advanced_document_word_ratio,
        "basic_cefr_word_ratio": basic_cefr_word_ratio,
        "intermediate_cefr_word_ratio": intermediate_cefr_word_ratio,
        "out_of_basic_cefr_ratio": out_of_basic_cefr_ratio,
        "nominalization_ratio": nominalization_ratio,
        "punctuation_density": punctuation_density,
        "formality_index": formality_index,
        "flesch_reading_ease": flesch,
    }


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def readability_complexity_score(text: str) -> float:
    """Эвристическая оценка сложности на основе readability-признаков."""

    features = readability_features(text)
    flesch_component = _clamp((100.0 - features["flesch_reading_ease"]) / 80.0)
    sentence_component = _clamp((features["avg_sentence_length"] - 8.0) / 22.0)
    word_component = _clamp((features["avg_word_length"] - 3.5) / 4.0)
    syllable_component = _clamp((features["syllables_per_word"] - 1.2) / 1.2)
    long_word_component = _clamp(features["long_word_ratio"] * 2.0)
    syntax_component = _clamp(
        features["subordinator_ratio"] * 6.0
        + features["connector_ratio"] * 6.0
        + features["nominalization_ratio"] * 4.0
    )
    vocabulary_component = _clamp(
        features["out_of_basic_cefr_ratio"] * 0.45
        + features["intermediate_cefr_word_ratio"] * 2.0
        + features["academic_word_ratio"] * 8.0
        + features["advanced_document_word_ratio"] * 6.0
    )
    formality_component = _clamp(features["formality_index"] * 3.0)

    # Этот скор полезен как интерпретируемый baseline и диагностический ориентир.
    # Финальное предсказание в production теперь определяется обученной моделью.
    return _clamp(
        0.27 * flesch_component
        + 0.15 * sentence_component
        + 0.14 * word_component
        + 0.12 * syllable_component
        + 0.08 * long_word_component
        + 0.10 * syntax_component
        + 0.10 * vocabulary_component
        + 0.04 * formality_component
    )


def make_ngrams(tokens: list[str], max_n: int = 2) -> list[str]:
    """Строит unigram и bigram признаки для лёгкой текстовой модели."""

    ngrams: list[str] = []
    for n in range(1, max_n + 1):
        for index in range(0, len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[index : index + n]))
    return ngrams


def build_vocabulary(texts: list[str], max_features: int) -> list[str]:
    """Собирает частотный словарь n-грамм для последующей векторизации."""

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(make_ngrams(tokenize(text), max_n=2))
    return [token for token, _ in counts.most_common(max_features)]


def vectorize(text: str, vocabulary: list[str]) -> dict[str, float]:
    """Преобразует текст в объединённый набор readability- и n-грамм-признаков."""

    tokens = tokenize(text)
    ngrams = make_ngrams(tokens, max_n=2)
    counts = Counter(ngrams)
    total = max(1, len(ngrams))

    # Абсолютные длины исключаем из модельной части,
    # чтобы длинный текст сам по себе не считался автоматически сложным.
    features = {
        name: value
        for name, value in readability_features(text).items()
        if name not in MODEL_EXCLUDED_FEATURES
    }
    for term in vocabulary:
        features[f"tfidf::{term}"] = counts[term] / total
    return features


def scale_features(rows: list[dict[str, float]]) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    """Стандартизует признаки и сохраняет статистики для инференса."""

    keys = sorted({key for row in rows for key in row})
    stats: dict[str, dict[str, float]] = {}
    scaled_rows: list[dict[str, float]] = []
    for key in keys:
        values = [row.get(key, 0.0) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stats[key] = {"mean": mean, "std": math.sqrt(variance) or 1.0}
    for row in rows:
        scaled_rows.append(
            {key: (row.get(key, 0.0) - stats[key]["mean"]) / stats[key]["std"] for key in keys}
        )
    return scaled_rows, stats


def apply_scaling(row: dict[str, float], stats: dict[str, dict[str, float]]) -> dict[str, float]:
    """Применяет статистики стандартизации к одной строке признаков."""

    return {
        key: (row.get(key, 0.0) - value["mean"]) / value["std"]
        for key, value in stats.items()
    }
