import datasets
from lm_eval.api.task import Task
from lm_eval.api.registry import register_task
from lm_eval.api.metric import mean, make_bleu_metric

# Define a custom BLEU metric function if not using the default registry one
# (Standard harness usually has "bleu", but it's safer to import explicitly if unsure)


@register_task("sami_nmt")
class SamiNMTTask(Task):
    VERSION = 0
    DATASET_PATH = "j0ma/sami-mt-data"
    DATASET_NAME = "default"

    def __init__(self, config=None):
        super().__init__(config=config)
        # Default config if none provided via command line
        self.src_lang = "fi"
        self.tgt_lang = "se"

        # KEY CHANGE: Support different prompt styles for MADLAD vs Chat models
        # Passed via --task_args prompt_style=madlad
        self.prompt_style = (
            config.get("prompt_style", "default") if config else "default"
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return []

    def validation_docs(self):
        return []

    def test_docs(self):
        # map() is handled automatically by the harness if using DATASET_PATH
        # but since we need to rename columns to a standard format, we do it here
        return self.dataset["test"]

    def doc_to_text(self, doc):
        src_text = doc[self.src_lang]

        # MADLAD-400 Format (Critical for performance)
        if self.prompt_style == "madlad":
            # <2se> is the token for Northern Sami in MADLAD
            return f"<2{self.tgt_lang}> {src_text}"

        # Aya 101 / T5 Format (Instruction style)
        elif self.prompt_style == "aya":
            return f"Translate to Northern Sami: {src_text}"

        # Default / Decoder-Only (NorMistral/Llama)
        # Uses standard Few-Shot / Completion format
        else:
            return f"Finnish: {src_text}\nNorthern Sami:"

    def doc_to_target(self, doc):
        # For Encoder-Decoder, this is just the target string.
        # For Decoder-only, this is the completion.
        # The harness handles the difference automatically based on model type.
        return (
            f" {doc[self.tgt_lang]}"
            if self.prompt_style == "default"
            else doc[self.tgt_lang]
        )

    def process_results(self, doc, results):
        # The 'results' argument contains the generated strings
        prediction = results[0]
        reference = doc[self.tgt_lang]

        return {
            "bleu": (reference, prediction),
            "chrf": (reference, prediction),
        }

    def aggregation(self):
        from lm_eval.metrics import bleu, chrf

        return {
            "bleu": bleu,  # Uses built-in harness metrics
            "chrf": chrf,
        }
