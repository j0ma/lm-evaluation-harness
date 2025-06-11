"""
TODO: FLORES abstract
"""

import importlib.util
import re
from collections.abc import Callable
from functools import partial
from typing import Any, Dict, Optional

import datasets
import numpy as np
from langcodes import Language

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.task import get_metric, get_aggregation


_CITATION = """
TODO: FLORES citation
"""


def code_to_language_name(lang_code):
    return Language.make(language=Language.get(lang_code)["language"]).display_name()


class FLORESTask(ConfigurableTask):
    VERSION = 0
    DATASET_NAME = "facebook/flores"

    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}
        assert "source_language_code" in config, (
            "FLORESTask must have a 'source_language_code' defined"
        )
        assert "target_language_code" in config, (
            "FLORESTask must have a 'target_language_code' defined"
        )
        self.source_language_code = config.pop("source_language_code")
        self.source_language = code_to_language_name(self.source_language_code)
        self.target_language_code = config.pop("target_language_code")
        self.target_language = code_to_language_name(self.target_language_code)

        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "dataset_name": self.DATASET_NAME,
            }
        )


    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        downloaded_dataset = datasets.load_dataset(
            path=self.DATASET_NAME,
            name=f"{self.source_language_code}-{self.target_language_code}",
            cache_dir=None
        )
        downloaded_dataset = downloaded_dataset.rename_column(
            f"sentence_{self.source_language_code}", "source_sentence"
        )
        downloaded_dataset = downloaded_dataset.rename_column(
            f"sentence_{self.target_language_code}", "target_sentence"
        )

        self.dataset = downloaded_dataset

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["dev"]

    def test_docs(self):
        return self.dataset["devtest"]

    def doc_to_text(self, doc):
        out = '''
{src_lang_long} sentence: {src_sent}

{tgt_lang_long} sentence: '''.format(
            src_lang_long=self.source_language,
            src_sent=doc['source_sentence'],
            tgt_lang_long=self.target_language
        )

        return out

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        return doc["target_sentence"]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        hypothesis_sentence = results[0]
        source_sentence = doc['source_sentence']
        reference_sentence = doc['target_sentence']

        return {
            "bleu": (reference_sentence, hypothesis_sentence),
            "chrf": (reference_sentence, hypothesis_sentence),
            "comet": (source_sentence, reference_sentence, hypothesis_sentence)
        }


    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """

        return {
            "bleu": get_aggregation("bleu"),
            "chrf": get_aggregation("chrf"),
            "comet": get_aggregation("comet")
        }

