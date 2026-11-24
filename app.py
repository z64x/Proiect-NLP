"""Interactive Gradio demo for Project 15: Extractive Question Answering System."""

from __future__ import annotations

import pandas as pd
import gradio as gr

from qa_pipeline import (
    DEFAULT_LOCAL_MODEL_DIR,
    ExtractiveQAInference,
    TfidfSentenceBaseline,
    annotate_answer_entity,
)

# Initialize models once at startup
qa_model = ExtractiveQAInference(model_path=DEFAULT_LOCAL_MODEL_DIR)
baseline_model = TfidfSentenceBaseline(stop_words="english")

MODEL_STATUS_BANNER = (
    f"**Active Transformer Model:** Fine-Tuned Local DistilBERT (`{qa_model.model_path}`)"
    if qa_model.is_local_finetuned
    else f"**Active Transformer Model:** Pre-trained Hub Fallback (`{qa_model.model_path}` — run `Question_Answering_System.ipynb` to train local checkpoint)"
)

EXAMPLES = [
    [
        "The Transformer architecture was introduced in 2017 by Vaswani et al. in the landmark paper 'Attention Is All You Need'. Unlike recurrent neural networks, Transformers rely entirely on self-attention mechanisms, enabling massive parallelization during training.",
        "When was the Transformer architecture introduced?",
        3,
    ],
    [
        "DistilBERT is a small, fast, cheap, and light Transformer model trained by distilling BERT-base. It retains 97% of BERT's language understanding capabilities while having 40% fewer parameters and running 60% faster.",
        "How much of BERT's language understanding does DistilBERT retain?",
        3,
    ],
    [
        "Bucharest is the capital and largest city of Romania, located in the southeast of the country along the banks of the Dambovita River. It was first mentioned in documents in 1459 and became the capital of Romania in 1862.",
        "Which river flows through Bucharest?",
        3,
    ],
    [
        "The Apollo 11 mission successfully landed the first humans on the Moon on July 20, 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin stepped onto the lunar surface while Michael Collins remained in lunar orbit.",
        "Who remained in lunar orbit during Apollo 11?",
        3,
    ],
]


def _build_highlighted_passage(context: str, start_char: int, end_char: int) -> list[tuple[str, str | None]]:
    """Split context into non-answer segments and the highlighted answer span."""
    if not context:
        return []
    if end_char <= start_char or start_char < 0 or end_char > len(context):
        return [(context, None)]

    segments: list[tuple[str, str | None]] = []
    if start_char > 0:
        segments.append((context[:start_char], None))
    segments.append((context[start_char:end_char], "Answer"))
    if end_char < len(context):
        segments.append((context[end_char:], None))
    return segments


def answer_question(
    passage: str,
    question: str,
    top_k: int = 3,
) -> tuple[str, list[tuple[str, str | None]], pd.DataFrame, str]:
    """Run extractive QA with DistilBERT and compare against the TF-IDF baseline."""
    passage = (passage or "").strip()
    question = (question or "").strip()

    if not passage or not question:
        empty_df = pd.DataFrame(columns=["Rank", "Candidate Answer", "Confidence (%)", "Char Span"])
        return (
            "Please provide both a context passage and a question.",
            [(passage, None)] if passage else [],
            empty_df,
            "N/A",
        )

    bert_result = qa_model.predict(passage, question, top_k=int(top_k))
    baseline_result = baseline_model.predict(passage, question)

    confidence_pct = bert_result["score"] * 100.0
    entity_label = annotate_answer_entity(bert_result["answer"]) or "NON-ENTITY"
    answer_summary = (
        f"**{bert_result['answer']}**\n\n"
        f"- **Confidence:** {confidence_pct:.2f}%\n"
        f"- **spaCy Entity Type:** `{entity_label}`\n"
        f"- **Character Span:** `[{bert_result['start_char']}, {bert_result['end_char']}]`"
    )

    highlighted = _build_highlighted_passage(
        passage,
        bert_result["start_char"],
        bert_result["end_char"],
    )

    candidate_rows = [
        {
            "Rank": rank,
            "Candidate Answer": cand["answer"],
            "Confidence (%)": round(cand["probability"] * 100.0, 2),
            "Char Span": f"[{cand['start_char']}, {cand['end_char']}]",
        }
        for rank, cand in enumerate(bert_result["top_k"], start=1)
    ]
    candidates_df = pd.DataFrame(candidate_rows)

    baseline_text = (
        f"{baseline_result['answer']}\n\n"
        f"(Cosine similarity: {baseline_result['score']:.4f})"
    )

    return answer_summary, highlighted, candidates_df, baseline_text


with gr.Blocks(title="Project 15: Extractive Question Answering System") as demo:
    gr.Markdown(
        f"""
        # Extractive Question Answering System (Project 15)
        **Course:** Natural Language Processing and Text Mining
        
        {MODEL_STATUS_BANNER}
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            passage_input = gr.Textbox(
                label="Context Passage",
                lines=8,
                placeholder="Paste a paragraph or article excerpt here...",
            )
            question_input = gr.Textbox(
                label="Question",
                lines=2,
                placeholder="Ask a factual question answerable from the passage...",
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Top-K Candidate Spans",
            )
            submit_btn = gr.Button("Extract Answer", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Markdown(label="DistilBERT Predicted Answer")
            highlighted_output = gr.HighlightedText(
                label="Highlighted Answer Span in Passage",
                color_map={"Answer": "#22c55e"},
                combine_adjacent=False,
                show_legend=True,
            )
            candidates_table = gr.Dataframe(
                label="Top-K Candidate Answer Spans",
                interactive=False,
            )
            baseline_output = gr.Textbox(
                label="TF-IDF Sentence Baseline (NLTK + Scikit-learn)",
                lines=4,
                interactive=False,
            )

    submit_btn.click(
        fn=answer_question,
        inputs=[passage_input, question_input, top_k_slider],
        outputs=[answer_output, highlighted_output, candidates_table, baseline_output],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[passage_input, question_input, top_k_slider],
        label="Try Example Passages & Questions",
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
