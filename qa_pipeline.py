"""Reusable Extractive Question Answering pipeline for SQuAD v1.1.

Covers all four stages required by Project 15 (Natural Language Processing and Text Mining):
1. Data Loading & EDA helpers (Hugging Face `datasets` + NLTK)
2. Preprocessing (NLTK tokenization, TF-IDF vectorization, DistilBERT sliding-window alignment)
3. Model Implementation (Classical TF-IDF sentence retrieval baseline + DistilBERT QA)
4. Evaluation & Inference (Official SQuAD EM/F1 metrics, post-processing, top-k span extraction)
"""

from __future__ import annotations

import collections
import os
import re
import string
from typing import Any, Callable

import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULT_LOCAL_MODEL_DIR = os.path.join("saved_model", "distilbert-squad")
DEFAULT_HUB_FALLBACK_MODEL = "distilbert-base-cased-distilled-squad"


def ensure_nltk_resources(quiet: bool = True) -> None:
    """Download required NLTK tokenizers and stopwords if not already cached."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=quiet)


# ---------------------------------------------------------------------------
# Stage 1: Data Loading & Exploratory Data Analysis Helpers
# ---------------------------------------------------------------------------


def load_squad_dataset(
    train_samples: int | None = 4000,
    val_samples: int | None = 500,
    seed: int = 42,
) -> DatasetDict:
    """Load the SQuAD v1.1 dataset and optionally take reproducible subsets.

    Args:
        train_samples: Number of training examples to sample, or None/<=0 for full split.
        val_samples: Number of validation examples to sample, or None/<=0 for full split.
        seed: Random seed used when shuffling before slicing.

    Returns:
        DatasetDict containing `train` and `validation` splits.
    """
    raw_datasets = load_dataset("rajpurkar/squad")

    train_split = raw_datasets["train"]
    if train_samples is not None and train_samples > 0 and train_samples < len(train_split):
        train_split = train_split.shuffle(seed=seed).select(range(train_samples))

    val_split = raw_datasets["validation"]
    if val_samples is not None and val_samples > 0 and val_samples < len(val_split):
        val_split = val_split.shuffle(seed=seed).select(range(val_samples))

    return DatasetDict({"train": train_split, "validation": val_split})


def classify_question_type(question: str) -> str:
    """Classify a question into canonical WH-categories for EDA and error analysis."""
    cleaned = question.strip().lower()
    if not cleaned:
        return "other"

    # Extract leading alphabetic token (handles forms like "What's" -> "what").
    match = re.match(r"^([a-z]+)", cleaned)
    if not match:
        return "other"

    first_word = match.group(1)
    if first_word in {"what", "who", "when", "where", "which", "how"}:
        return first_word
    if first_word in {"whom", "whose"}:
        return "who"
    return "other"


def compute_eda_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Compute per-example length and question-type statistics using NLTK."""
    ensure_nltk_resources()
    records: list[dict[str, Any]] = []
    for row in dataset:
        context = row["context"]
        question = row["question"]
        answers = row["answers"]["text"]
        primary_answer = answers[0] if answers else ""

        context_words = nltk.word_tokenize(context)
        question_words = nltk.word_tokenize(question)
        answer_words = nltk.word_tokenize(primary_answer) if primary_answer else []

        records.append(
            {
                "id": row["id"],
                "title": row.get("title", ""),
                "context_chars": len(context),
                "context_words": len(context_words),
                "context_sentences": len(nltk.sent_tokenize(context)),
                "question_words": len(question_words),
                "answer_chars": len(primary_answer),
                "answer_words": len(answer_words),
                "question_type": classify_question_type(question),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stage 4 (Shared): Official SQuAD v1.1 Evaluation Metrics
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Lowercase text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if normalized prediction exactly matches ground truth, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between normalized prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: list[str],
) -> float:
    """Compute the maximum metric value across all valid SQuAD reference answers."""
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def evaluate_predictions(
    predictions: dict[str, str],
    references: Dataset | list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate predictions against SQuAD references using Exact Match (EM) and F1.

    Args:
        predictions: Mapping `{example_id: predicted_answer_text}`.
        references: Validation Dataset or iterable of dicts containing `id`, `answers`,
            and optionally `question` and `context`.

    Returns:
        Dictionary with percentage `exact_match`, percentage `f1`, `num_examples`,
        and `per_example` breakdown list.
    """
    per_example: list[dict[str, Any]] = []
    total_em = 0.0
    total_f1 = 0.0
    count = 0

    for row in references:
        ex_id = row["id"]
        ground_truths = row["answers"]["text"]
        pred_text = predictions.get(ex_id, "")

        em = metric_max_over_ground_truths(compute_exact_match, pred_text, ground_truths)
        f1 = metric_max_over_ground_truths(compute_f1, pred_text, ground_truths)

        total_em += em
        total_f1 += f1
        count += 1

        question_text = row.get("question", "")
        per_example.append(
            {
                "id": ex_id,
                "question": question_text,
                "question_type": classify_question_type(question_text),
                "context": row.get("context", ""),
                "prediction": pred_text,
                "ground_truths": ground_truths,
                "em": em * 100.0,
                "f1": f1 * 100.0,
            }
        )

    exact_match_pct = (100.0 * total_em / count) if count > 0 else 0.0
    f1_pct = (100.0 * total_f1 / count) if count > 0 else 0.0

    return {
        "exact_match": exact_match_pct,
        "f1": f1_pct,
        "num_examples": count,
        "per_example": per_example,
    }


# ---------------------------------------------------------------------------
# Stage 3.1: Classical Baseline (NLTK Sentence Tokenization + TF-IDF)
# ---------------------------------------------------------------------------


class TfidfSentenceBaseline:
    """Classical extractive QA baseline that retrieves the closest sentence via TF-IDF."""

    def __init__(self, stop_words: str | None = "english") -> None:
        self.stop_words = stop_words
        ensure_nltk_resources()

    @staticmethod
    def _sentence_spans(context: str) -> list[tuple[str, int, int]]:
        """Split context into sentences while preserving exact character offsets."""
        sentences = nltk.sent_tokenize(context)
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for sent in sentences:
            idx = context.find(sent, cursor)
            if idx == -1:
                idx = context.find(sent)
            if idx != -1:
                spans.append((sent, idx, idx + len(sent)))
                cursor = idx + len(sent)
            else:
                spans.append((sent, 0, len(sent)))
        return spans

    def predict(self, context: str, question: str) -> dict[str, Any]:
        """Return the highest-scoring sentence from `context` for `question`."""
        context = context or ""
        question = question or ""
        if not context.strip():
            return {"answer": "", "score": 0.0, "start_char": 0, "end_char": 0}

        spans = self._sentence_spans(context)
        if not spans:
            return {"answer": context.strip(), "score": 0.0, "start_char": 0, "end_char": len(context)}

        sentences = [s[0] for s in spans]
        corpus = sentences + [question]

        try:
            vectorizer = TfidfVectorizer(stop_words=self.stop_words)
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Fallback when stop-word filtering empties the vocabulary (e.g., short/symbol text).
            try:
                vectorizer = TfidfVectorizer(stop_words=None)
                tfidf_matrix = vectorizer.fit_transform(corpus)
            except ValueError:
                best_sent, start_c, end_c = spans[0]
                return {"answer": best_sent, "score": 0.0, "start_char": start_c, "end_char": end_c}

        sent_vectors = tfidf_matrix[:-1]
        question_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(question_vector, sent_vectors).flatten()

        best_idx = int(np.argmax(similarities))
        best_sent, start_c, end_c = spans[best_idx]
        return {
            "answer": best_sent,
            "score": float(similarities[best_idx]),
            "start_char": start_c,
            "end_char": end_c,
        }

    def evaluate_dataset(self, dataset: Dataset) -> tuple[dict[str, Any], dict[str, str]]:
        """Run sentence-retrieval baseline over a SQuAD split and compute EM/F1."""
        predictions: dict[str, str] = {}
        for row in dataset:
            result = self.predict(row["context"], row["question"])
            predictions[row["id"]] = result["answer"]
        metrics = evaluate_predictions(predictions, dataset)
        return metrics, predictions


# ---------------------------------------------------------------------------
# Stage 2 & 3.2: DistilBERT Sliding-Window Preprocessing & Post-Processing
# ---------------------------------------------------------------------------


def prepare_train_features(
    examples: dict[str, list[Any]],
    tokenizer: Any,
    max_length: int = 384,
    doc_stride: int = 128,
) -> dict[str, Any]:
    """Tokenize QA pairs with sliding windows and align character spans to token indices."""
    questions = [q.lstrip() for q in examples["question"]]
    pad_on_right = tokenizer.padding_side == "right"

    tokenized = tokenizer(
        questions if pad_on_right else examples["context"],
        examples["context"] if pad_on_right else questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        context_seq_id = 1 if pad_on_right else 0

        token_start_index = 0
        while sequence_ids[token_start_index] != context_seq_id:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_seq_id:
            token_end_index -= 1

        if not (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            tokenized["start_positions"].append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized["end_positions"].append(token_end_index + 1)

    return tokenized


def prepare_validation_features(
    examples: dict[str, list[Any]],
    tokenizer: Any,
    max_length: int = 384,
    doc_stride: int = 128,
) -> dict[str, Any]:
    """Tokenize validation examples with `example_id` and context-only offset mappings."""
    questions = [q.lstrip() for q in examples["question"]]
    pad_on_right = tokenizer.padding_side == "right"

    tokenized = tokenizer(
        questions if pad_on_right else examples["context"],
        examples["context"] if pad_on_right else questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []
    context_seq_id = 1 if pad_on_right else 0

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])
        tokenized["offset_mapping"][i] = [
            offset if sequence_ids[k] == context_seq_id else None
            for k, offset in enumerate(tokenized["offset_mapping"][i])
        ]

    return tokenized


def postprocess_qa_predictions(
    examples: Dataset | list[dict[str, Any]],
    features: Dataset | list[dict[str, Any]],
    raw_predictions: tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    """Map raw `(start_logits, end_logits)` back to character spans in the original context."""
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {ex["id"]: idx for idx, ex in enumerate(examples)}
    features_per_example: dict[int, list[int]] = collections.defaultdict(list)
    for feat_idx, feat in enumerate(features):
        features_per_example[example_id_to_index[feat["example_id"]]].append(feat_idx)

    predictions: dict[str, str] = {}
    nbest_predictions: dict[str, list[dict[str, Any]]] = {}

    for ex_idx, example in enumerate(examples):
        context = example["context"]
        feature_indices = features_per_example[ex_idx]
        prelim_predictions: list[dict[str, Any]] = []

        for feat_idx in feature_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits = all_end_logits[feat_idx]
            offset_mapping = features[feat_idx]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or (end_index - start_index + 1) > max_answer_length:
                        continue

                    start_char = int(offset_mapping[start_index][0])
                    end_char = int(offset_mapping[end_index][1])
                    prelim_predictions.append(
                        {
                            "score": float(start_logits[start_index] + end_logits[end_index]),
                            "text": context[start_char:end_char],
                            "start_char": start_char,
                            "end_char": end_char,
                        }
                    )

        prelim_sorted = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)
        seen_texts: set[str] = set()
        unique_nbest: list[dict[str, Any]] = []
        for pred in prelim_sorted:
            if pred["text"] not in seen_texts and pred["text"].strip():
                seen_texts.add(pred["text"])
                unique_nbest.append(pred)
            if len(unique_nbest) >= n_best_size:
                break

        if not unique_nbest:
            unique_nbest = [{"text": "", "score": 0.0, "start_char": 0, "end_char": 0}]

        scores = np.array([entry["score"] for entry in unique_nbest], dtype=np.float64)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        formatted_nbest: list[dict[str, Any]] = []
        for entry, prob in zip(unique_nbest, probs):
            formatted_nbest.append(
                {
                    "answer": entry["text"],
                    "probability": float(prob),
                    "logit_score": float(entry["score"]),
                    "start_char": entry["start_char"],
                    "end_char": entry["end_char"],
                }
            )

        predictions[example["id"]] = formatted_nbest[0]["answer"]
        nbest_predictions[example["id"]] = formatted_nbest

    return predictions, nbest_predictions


# ---------------------------------------------------------------------------
# Standalone Inference Wrapper (Notebook Custom Tests & Gradio App)
# ---------------------------------------------------------------------------


class ExtractiveQAInference:
    """Inference wrapper for a fine-tuned Hugging Face extractive QA model."""

    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_MODEL_DIR,
        fallback_model: str = DEFAULT_HUB_FALLBACK_MODEL,
        device: str | None = None,
    ) -> None:
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            resolved_path = model_path
            self.is_local_finetuned = True
        else:
            resolved_path = fallback_model
            self.is_local_finetuned = False

        self.model_path = resolved_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(resolved_path).to(self.device)
        self.model.eval()

    def predict(
        self,
        context: str,
        question: str,
        top_k: int = 3,
        max_length: int = 384,
        doc_stride: int = 128,
        max_answer_length: int = 30,
    ) -> dict[str, Any]:
        """Extract the top-k candidate answer spans and exact character offsets."""
        context = (context or "").strip()
        question = (question or "").strip()
        if not context or not question:
            return {
                "answer": "",
                "score": 0.0,
                "start_char": 0,
                "end_char": 0,
                "top_k": [],
                "model_path": self.model_path,
                "is_local_finetuned": self.is_local_finetuned,
            }

        example = {"id": ["single_query"], "context": [context], "question": [question]}
        features_dict = prepare_validation_features(
            example,
            self.tokenizer,
            max_length=max_length,
            doc_stride=doc_stride,
        )

        num_windows = len(features_dict["input_ids"])
        input_ids = torch.tensor(features_dict["input_ids"], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            features_dict["attention_mask"], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits.detach().cpu().numpy()
            end_logits = outputs.end_logits.detach().cpu().numpy()

        feature_records = [
            {
                "example_id": features_dict["example_id"][i],
                "offset_mapping": features_dict["offset_mapping"][i],
            }
            for i in range(num_windows)
        ]
        example_records = [{"id": "single_query", "context": context, "question": question}]

        _, nbest_map = postprocess_qa_predictions(
            examples=example_records,
            features=feature_records,
            raw_predictions=(start_logits, end_logits),
            n_best_size=max(20, top_k),
            max_answer_length=max_answer_length,
        )
        candidates = nbest_map["single_query"][:top_k]
        best = candidates[0]

        return {
            "answer": best["answer"],
            "score": best["probability"],
            "start_char": best["start_char"],
            "end_char": best["end_char"],
            "top_k": candidates,
            "model_path": self.model_path,
            "is_local_finetuned": self.is_local_finetuned,
        }


# ---------------------------------------------------------------------------
# spaCy Linguistic Analysis & Entity Annotation Helpers
# ---------------------------------------------------------------------------

_SPACY_NLP_CACHE: dict[str, Any] = {}


def _get_spacy_nlp(model_name: str = "en_core_web_sm") -> Any:
    """Lazily load and cache a spaCy pipeline with a clear hint if missing."""
    if model_name in _SPACY_NLP_CACHE:
        return _SPACY_NLP_CACHE[model_name]

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is not installed. Run `pip install spacy` and "
            f"`python -m spacy download {model_name}`."
        ) from exc

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run `python -m spacy download {model_name}`."
        ) from exc

    _SPACY_NLP_CACHE[model_name] = nlp
    return nlp


def analyze_spacy_ner_and_pos(
    texts: list[str],
    batch_size: int = 256,
    model_name: str = "en_core_web_sm",
) -> dict[str, Any]:
    """Run spaCy NER and POS tagging over a list of texts.

    Returns:
        Dictionary with:
        - `primary_entities`: list of primary entity labels (`doc.ents[0].label_` or `"NON-ENTITY"`)
        - `ner_counts`: Counter-like dict of entity label frequencies
        - `pos_counts`: Counter-like dict of universal POS tag frequencies
    """
    nlp = _get_spacy_nlp(model_name=model_name)
    primary_entities: list[str] = []
    ner_counts: collections.Counter[str] = collections.Counter()
    pos_counts: collections.Counter[str] = collections.Counter()

    # Disable lemmatizer/parser so batch tagging over thousands of spans is fast.
    disabled = [pipe for pipe in ("parser", "lemmatizer") if pipe in nlp.pipe_names]
    cleaned_texts = [t if isinstance(t, str) else "" for t in texts]

    for doc in nlp.pipe(cleaned_texts, batch_size=batch_size, disable=disabled):
        if doc.ents:
            primary_entities.append(doc.ents[0].label_)
            for ent in doc.ents:
                ner_counts[ent.label_] += 1
        else:
            primary_entities.append("NON-ENTITY")

        for token in doc:
            if not token.is_space and not token.is_punct:
                pos_counts[token.pos_] += 1

    return {
        "primary_entities": primary_entities,
        "ner_counts": dict(ner_counts.most_common()),
        "pos_counts": dict(pos_counts.most_common()),
    }


def annotate_answer_entity(
    text: str,
    model_name: str = "en_core_web_sm",
) -> str | None:
    """Return the primary spaCy named entity label for a single answer string, or None."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    nlp = _get_spacy_nlp(model_name=model_name)
    disabled = [pipe for pipe in ("parser", "lemmatizer") if pipe in nlp.pipe_names]
    doc = nlp(cleaned, disable=disabled)
    if doc.ents:
        return doc.ents[0].label_
    return None

