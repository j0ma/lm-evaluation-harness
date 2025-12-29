"""
Sami NMT (Neural Machine Translation) Task for LM Evaluation Harness.
Supports multiple prompt styles for different model families (MADLAD, Aya, decoder-only models).

Abstract: We consider a low-resource translation task from Finnish into Northern Sámi. Collecting all available parallel data between the languages, we obtain around 30,000 sentence pairs. However, there exists a significantly larger monolingual Northern Sámi corpus, as well as a rule-based machine translation (RBMT) system between the languages. To make the best use of the monolingual data in a neural machine translation (NMT) system, we use the backtranslation approach to create synthetic parallel data from it using both NMT and RBMT systems. Evaluating the results on an in-domain test set and a small out-of-domain set, we find that the RBMT backtranslation outperforms NMT backtranslation clearly for the out-of-domain test set, but also slightly for the in-domain data, for which the NMT backtranslation model provided clearly better BLEU scores than the RBMT. In addition, combining both backtranslated data sets improves the RBMT approach only for the in-domain test set. This suggests that the RBMT system provides general-domain knowledge that cannot be found from the relative small parallel training data.
"""

from typing import Any, Dict, Optional

import datasets
from langcodes import Language

from lm_eval.api.task import ConfigurableTask, get_aggregation

_CITATION = """
Mikko Aulamo, Sami Virpioja, Yves Scherrer, and Jörg Tiedemann. 2021. Boosting Neural Machine Translation from Finnish to Northern Sámi with Rule-Based Backtranslation. In Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa), pages 351–356, Reykjavik, Iceland (Online). Linköping University Electronic Press, Sweden.
"""


def code_to_language_name(lang_code: str):
    return Language.make(language=Language.get(lang_code)["language"]).display_name()


def code_to_shortcode(lang_code: str):
    """Converts a language code to a BCP-47 two-letter shortcode"""
    default = Language.make(language=Language.get(lang_code)["language"]).to_tag()
    cache = {
        "sme": "se",  # Northern Sami
        "fin": "fi",  # Finnish
    }

    return cache.get(lang_code, default)


class SamiNMTTaskFewShot(ConfigurableTask):
    VERSION = 0
    DATASET_NAME = "j0ma/sami-mt-data-withtrain"

    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}

        # Required config fields
        assert (
            "source_language_code" in config
        ), "SamiNMTTask must have a 'source_language_code' defined"
        assert (
            "target_language_code" in config
        ), "SamiNMTTask must have a 'target_language_code' defined"

        # Extract and pop language codes
        self.source_language_code = config.pop("source_language_code")
        self.source_language = code_to_language_name(self.source_language_code)
        self.target_language_code = config.pop("target_language_code")
        self.target_language = code_to_language_name(self.target_language_code)

        # Prompt style for different model families (default: "default" for decoder-only)
        self.prompt_style = config.pop("prompt_style", "default")

        # Optional: corpus specification (uit, yle, etc.)
        self.corpus = config.pop("corpus", None)

        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "dataset_name": self.DATASET_NAME,
            }
        )

    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Download and prepare the dataset."""
        downloaded_dataset = datasets.load_dataset(
            path=self.DATASET_NAME, name="default", cache_dir=None
        )

        self.dataset = downloaded_dataset

    def has_training_docs(self):
        return "uit_train" in self.dataset

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs:
            return self.dataset["uit_train"]
        else:
            return []

    def validation_docs(self):
        return []

    def test_docs(self):
        # Get the test split for the appropriate corpus
        test_docs = self.dataset[f"{self.corpus}_test"]
        print("[SamiNMTTask] Number of test documents:", len(test_docs))
        return test_docs

    def doc_to_text(self, doc):
        """Format the source sentence for the model input.

        Different prompt styles for different model families:
        - "madlad": Uses <2{lang_code}> token for MADLAD-style encoder-decoder models
        - "default": Uses completion-style prompts for decoder-only models (NorMistral, Llama)
        """
        src_field = f"text_{self.source_language_code}"
        src_text = doc[src_field]

        if self.prompt_style == "madlad":
            # MADLAD-400 format: <2se> for Northern Sami, <2fi> for Finnish, etc.
            target_lang_shortcode = code_to_shortcode(self.target_language_code)
            return f"<2{target_lang_shortcode}> {src_text}"
        else:
            out = (
                f"Translate from {self.source_language} to {self.target_language}\n"
                f"{self.source_language} sentence: {src_text}\n"
                f"{self.target_language} sentence: "
            )
            return out

    def doc_to_target(self, doc):
        """Return the target sentence.

        For decoder-only models, we add a leading space because the prompt
        ends with a space and we want the model to generate the actual text.
        For encoder-decoder models (MADLAD, Aya), the target is raw.
        """
        tgt_field = f"text_{self.target_language_code}"
        tgt_text = doc[tgt_field]

        # Decoder-only models need a leading space for proper tokenization
        if self.prompt_style == "default":
            return f" {tgt_text}"
        else:
            # Encoder-decoder models (MADLAD, Aya)
            return tgt_text

    def should_decontaminate(self):
        return False

    def process_results(self, doc, results):
        """Evaluate the generated translation against the reference."""
        hypothesis_sentence = results[0]
        src_field = f"text_{self.source_language_code}"
        tgt_field = f"text_{self.target_language_code}"
        source_sentence = doc[src_field]
        reference_sentence = doc[tgt_field]

        return {
            "bleu": (reference_sentence, hypothesis_sentence),
            "chrf": (reference_sentence, hypothesis_sentence),
        }

    def aggregation(self):
        """Return aggregation functions for metrics."""
        return {
            "bleu": get_aggregation("bleu"),
            "chrf": get_aggregation("chrf"),
        }
