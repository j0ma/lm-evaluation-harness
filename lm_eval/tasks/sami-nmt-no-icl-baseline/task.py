"""
Abstract: We consider a low-resource translation task from Finnish into Northern Sámi. Collecting all available parallel data between the languages, we obtain around 30,000 sentence pairs. However, there exists a significantly larger monolingual Northern Sámi corpus, as well as a rule-based machine translation (RBMT) system between the languages. To make the best use of the monolingual data in a neural machine translation (NMT) system, we use the backtranslation approach to create synthetic parallel data from it using both NMT and RBMT systems. Evaluating the results on an in-domain test set and a small out-of-domain set, we find that the RBMT backtranslation outperforms NMT backtranslation clearly for the out-of-domain test set, but also slightly for the in-domain data, for which the NMT backtranslation model provided clearly better BLEU scores than the RBMT. In addition, combining both backtranslated data sets improves the RBMT approach only for the in-domain test set. This suggests that the RBMT system provides general-domain knowledge that cannot be found from the relative small parallel training data.
"""

import datasets
from langcodes import Language

_CITATION = """
Mikko Aulamo, Sami Virpioja, Yves Scherrer, and Jörg Tiedemann. 2021. Boosting Neural Machine Translation from Finnish to Northern Sámi with Rule-Based Backtranslation. In Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa), pages 351–356, Reykjavik, Iceland (Online). Linköping University Electronic Press, Sweden.
"""

def code_to_language_name(lang_code):
    return Language.make(language=Language.get(lang_code)["language"]).display_name()

class SamiNMTTask(ConfigurableTask):
    VERSION = 0
    DATASET_PATH = "j0ma/sami-mt-data"
    DATASET_NAME = "default"

    def __init__(self, config):
        super().__init__(config=config)
        # Default config if none provided via command line
        # self.src_lang = "sme"
        # self.tgt_lang = "fin"


        # KEY CHANGE: Support different prompt styles for MADLAD vs Chat models
        # Passed via --task_args prompt_style=madlad
        self.prompt_style = config.get("prompt_style", "default")
        self.corpus = config["corpus"]

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
        src_lang_field = f"text_{self.src_lang}"
        src_text = doc[src_lang_field]

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
        tgt_lang_field = f"text_{self.tgt_lang}"
        return (
            f" {doc[tgt_lang_field]}"
            if self.prompt_style == "default"
            else doc[tgt_lang_field]
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

        return {
            "bleu": get_aggregation("bleu"),
            "chrf": get_aggregation("chrf"),
            # "comet": get_aggregation("comet"),
        }
