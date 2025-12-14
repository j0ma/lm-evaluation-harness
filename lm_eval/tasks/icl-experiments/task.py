"""
In-Context Learning Machine Translation Task for lm-evaluation-harness
Cross-lingual ICL experiments for MT evaluation with FLORES-200/NTREX-128.

Usage:
lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-chat-hf \
  --tasks icl_mt_baseline,icl_mt_sanity,exp1-icl-which-language-is-best,icl_mt_similarity,icl_mt_similarity_random,icl_mt_ordering,icl_mt_ordering_similarity_asc,icl_mt_ordering_similarity_desc,icl_mt_ordering_length_asc,icl_mt_ordering_length_desc \
  --num_fewshot 5 --batch_size 1 --output_path ./results
"""

import random
from typing import Any, Dict, Optional
from functools import cached_property
import datasets
from langcodes import Language

from lm_eval.api.task import ConfigurableTask, get_aggregation
from lm_eval.api.registry import register_task


def code_to_language_name(lang_code):
    return Language.make(language=Language.get(lang_code)["language"]).display_name()


def sentence_similarity_score(sent1: str, sent2: str) -> float:
    words1 = set(sent1.lower().split())
    words2 = set(sent2.lower().split())
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union
    len_sim = 1.0 - abs(len(sent1) - len(sent2)) / max(len(sent1), len(sent2), 1)
    return 0.7 * jaccard + 0.3 * len_sim


class ICLTranslationTask(ConfigurableTask):
    """Base class for In-Context Learning Translation experiments."""

    VERSION = 1
    DATASET_NAME = "facebook/flores"

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}

        assert "source_language_code" in config, (
            "ICLTranslationTask must have a 'source_language_code' defined"
        )
        assert "target_language_code" in config, (
            "ICLTranslationTask must have a 'target_language_code' defined"
        )

        self.source_language_code = config.pop("source_language_code")
        self.source_language = code_to_language_name(self.source_language_code)
        self.target_language_code = config.pop("target_language_code")
        self.target_language = code_to_language_name(self.target_language_code)

        self.experiment_type = config.pop("experiment_type", "base")
        self.icl_source_language_code = config.pop(
            "icl_source_language_code", self.source_language_code
        )
        self.icl_target_language_code = config.pop(
            "icl_target_language_code", self.target_language_code
        )
        self.ordering_strategy = config.pop("ordering_strategy", "random")
        self.icl_language_pairs = config.pop(
            "icl_language_pairs",
            [(self.source_language_code, self.target_language_code)],
        )

        self.icl_pool = None
        self._rng = random.Random(42)

        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "dataset_name": self.DATASET_NAME,
                **config,
            }
        )

    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        downloaded_dataset = datasets.load_dataset(
            path=self.DATASET_NAME,
            name=f"{self.source_language_code}-{self.target_language_code}",
            cache_dir=None,
        )
        downloaded_dataset = downloaded_dataset.rename_column(
            f"sentence_{self.source_language_code}", "source_sentence"
        )
        downloaded_dataset = downloaded_dataset.rename_column(
            f"sentence_{self.target_language_code}", "target_sentence"
        )
        self.dataset = downloaded_dataset
        self._prepare_icl_pool()

    def _prepare_icl_pool(self):
        if self.icl_pool is not None:
            return
        dev_docs = list(self.dataset["dev"])
        self.icl_pool = []
        for doc in dev_docs:
            sid = doc["id"]
            src_lang = doc.get("source_language_code", self.source_language_code)
            tgt_lang = doc.get("target_language_code", self.target_language_code)
            self.icl_pool.append(
                {
                    "id": sid,
                    "source_sentence": doc["source_sentence"],
                    "target_sentence": doc["target_sentence"],
                    "source_language_code": src_lang,
                    "target_language_code": tgt_lang,
                }
            )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        self._prepare_icl_pool()
        return self.dataset["dev"]

    def test_docs(self):
        self._prepare_icl_pool()
        return self.dataset["devtest"]

    def doc_to_text(self, doc):
        num_fewshot = getattr(self, "num_fewshot", 0)
        if num_fewshot == 0:
            return f"""
{self.source_language} sentence: {doc["source_sentence"]}

{self.target_language} sentence: """
        icl_examples = self._select_icl_examples(doc, num_fewshot)
        prompt_parts = []
        for example in icl_examples:
            prompt_parts.append(f"""{code_to_language_name(example["source_language_code"])} sentence: {example["source_sentence"]}
{code_to_language_name(example["target_language_code"])} sentence: {example["target_sentence"]}
""")
        prompt_parts.append(f"""{self.source_language} sentence: {doc["source_sentence"]}
{self.target_language} sentence: """)
        return "".join(prompt_parts)

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        return doc["target_sentence"]

    def _select_icl_examples(self, doc, num_examples):
        if not self.icl_pool or num_examples <= 0:
            return []
        current_id = doc["id"]
        candidates = [item for item in self.icl_pool if item["id"] != current_id]
        if self.experiment_type == "language":
            return self._experiment_specific_selection(doc, self.icl_pool, num_examples)
        else:
            if len(candidates) < num_examples:
                return candidates
            return self._experiment_specific_selection(doc, candidates, num_examples)

    def process_results(self, doc, results):
        hypothesis_sentence = results[0] if results else ""
        hypothesis_sentence = self._postprocess_prediction(hypothesis_sentence)
        source_sentence = doc["source_sentence"]
        reference_sentence = doc["target_sentence"]
        return {
            "bleu": (reference_sentence, hypothesis_sentence),
            "chrf": (reference_sentence, hypothesis_sentence),
            "comet": (source_sentence, reference_sentence, hypothesis_sentence),
        }

    def _postprocess_prediction(self, prediction):
        prediction = prediction.strip()
        prefixes_to_remove = [
            "Translation:",
            "translation:",
            "TRANSLATION:",
            "Output:",
            "output:",
            "OUTPUT:",
            "Answer:",
            "answer:",
            "ANSWER:",
        ]
        for prefix in prefixes_to_remove:
            if prediction.startswith(prefix):
                prediction = prediction[len(prefix) :].strip()
                break
        return prediction

    def aggregation(self):
        return {
            "bleu": get_aggregation("bleu"),
            "chrf": get_aggregation("chrf"),
            "comet": get_aggregation("comet"),
        }


@register_task("icl_mt_baseline")
class ICLBaselineTask(ICLTranslationTask):
    """
    Baseline

    - No ICL examples used.

    Research Question:
    - What is model translation performance without any ICL?

    Experimental Conditions:
    - Direct translation with no contextual examples.

    Independent Variable:
    - None.

    Dependent Variable:
    - BLEU, chrF, COMET.

    Confounding Variables:
    - Sentence difficulty.

    Expected Behavior:
    - Reflects the raw model quality.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["experiment_type"] = "baseline"
        super().__init__(config)

    def doc_to_text(self, doc):
        return f"""
{self.source_language} sentence: {doc["source_sentence"]}

{self.target_language} sentence: """

    def _select_icl_examples(self, doc, num_examples):
        return []


@register_task("icl_mt_sanity")
class ICLSanityCheckTask(ICLTranslationTask):
    """
    Sanity Check

    Baseline:
    - Model performance with no ICL.

    Research Question:
    - Does including the exact test sentence as ICL allow the model to "cheat"?

    Experimental Conditions:
    - Each test sentence is included in the context as an ICL example.

    Independent Variable:
    - Presence/absence of the exact test sentence as ICL.

    Dependent Variable:
    - BLEU, chrF, COMET.

    Confounding Variables:
    - Which other ICL examples are included.
    - ICL example order.

    Expected Behavior:
    - Score should be very high for copied pairs.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["experiment_type"] = "sanity"
        super().__init__(config)

    def _experiment_specific_selection(self, doc, candidates, num_examples):
        if num_examples <= 0:
            return []
        cheat_example = {
            "id": doc["id"],
            "source_sentence": doc["source_sentence"],
            "target_sentence": doc["target_sentence"],
            "source_language_code": self.source_language_code,
            "target_language_code": self.target_language_code,
        }
        if num_examples == 1:
            return [cheat_example]
        self._rng.seed(42 + hash(str(doc["id"])))
        additional = self._rng.sample(
            [item for item in candidates if item["id"] != doc["id"]],
            min(num_examples - 1, len(candidates)),
        )
        return [cheat_example] + additional


@register_task("exp1-icl-which-language-is-best")
class ICLEnExWhichLanguageIsBestTask(ICLTranslationTask):
    """
    Which Language Is Best for ICL?

    Baseline:
    - No ICL examples used.

    Research Question:
    - Does an ICL example in a different language pair help translation for the same sentence?

    Experimental Conditions:
    - For each test sentence, use its parallel translations in all other language pairs as ICL.

    Independent Variable:
    - ICL language pair.

    Dependent Variable:
    - BLEU, chrF, COMET.

    Confounding Variables:
    - Which other ICL examples are present.
    - ICL example order.

    Expected Behavior:
    - Score gains indicate helpful languages; high-resource pairs may help more.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["experiment_type"] = "language"
        super().__init__(config)

    def _prepare_icl_pool(self):
        if hasattr(self, "icl_indexed_pool"):
            return
        dev_docs = list(self.dataset["dev"])
        self.icl_indexed_pool = {}
        self.all_lang_pairs = set()
        for doc in dev_docs:
            sid = doc["id"]
            src_lang = doc.get("source_language_code", self.source_language_code)
            tgt_lang = doc.get("target_language_code", self.target_language_code)
            self.icl_indexed_pool[(sid, src_lang, tgt_lang)] = {
                "id": sid,
                "source_sentence": doc["source_sentence"],
                "target_sentence": doc["target_sentence"],
                "source_language_code": src_lang,
                "target_language_code": tgt_lang,
            }
            self.all_lang_pairs.add((src_lang, tgt_lang))
        self.all_lang_pairs = sorted(self.all_lang_pairs)

    def _experiment_specific_selection(self, doc, icl_pool_unused, num_examples):
        self._prepare_icl_pool()
        sid = doc["id"]
        test_pair = (self.source_language_code, self.target_language_code)
        icl_examples = []
        for src_lang, tgt_lang in self.all_lang_pairs:
            if (src_lang, tgt_lang) == test_pair:
                continue
            key = (sid, src_lang, tgt_lang)
            example = self.icl_indexed_pool.get(key)
            if example:
                icl_examples.append(example)
        if num_examples > 0:
            icl_examples = icl_examples[:num_examples]
        if len(icl_examples) < num_examples:
            self._rng.seed(42 + hash(str(doc["id"])))
            available_ids = set(k[0] for k in self.icl_indexed_pool.keys()) - {sid}
            other_candidates = [
                x for x in self.icl_indexed_pool.values() if x["id"] in available_ids
            ]
            icl_examples += self._rng.sample(
                other_candidates, num_examples - len(icl_examples)
            )
        return icl_examples


@register_task("icl_mt_similarity")
class ICLSimilarityTask(ICLTranslationTask):
    """
    Similarity-Based Selection

    Baseline:
    - Random ICL selection.

    Research Question:
    - Does using similar sentences as ICL examples improve translation?

    Experimental Conditions:
    - Select top-k similar sentences (same language pair) for ICL.

    Independent Variable:
    - Similarity of ICL context sentence to test sentence.

    Dependent Variable:
    - BLEU, chrF, COMET.

    Confounding Variables:
    - Which other ICL examples are present.
    - ICL example order.

    Expected Behavior:
    - Higher scores for more relevant examples.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["experiment_type"] = "similarity"
        super().__init__(config)

    def _experiment_specific_selection(self, doc, candidates, num_examples):
        test_source = doc["source_sentence"]
        candidates_scored = []
        for candidate in candidates:
            similarity = sentence_similarity_score(
                test_source, candidate["source_sentence"]
            )
            candidates_scored.append((similarity, candidate))
        candidates_scored.sort(key=lambda x: x[0], reverse=True)
        selected = [candidate for _, candidate in candidates_scored[:num_examples]]
        return selected


@register_task("icl_mt_similarity_random")
class ICLSimilarityRandomTask(ICLTranslationTask):
    """
    Similarity (Random Selection Control)

    Baseline:
    - Random ICL examples.

    Research Question:
    - Is random ICL less effective than similarity-based?

    Experimental Conditions:
    - ICL examples randomly sampled (same lang pair, different IDs).

    Independent Variable:
    - Selection: similarity-based vs random.

    Dependent Variable:
    - BLEU, chrF, COMET.

    Confounding Variables:
    - Which other ICL examples are present.
    - ICL example order.

    Expected Behavior:
    - Similarity should outperform random.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["experiment_type"] = "similarity_random"
        super().__init__(config)

    def _experiment_specific_selection(self, doc, candidates, num_examples):
        self._rng.seed(42 + hash(str(doc["id"])))
        return self._rng.sample(candidates, min(num_examples, len(candidates)))


@register_task("icl_mt_ordering")
class ICLOrderingTask(ICLTranslationTask):
    """
    ICL Example Ordering

    Baseline:
    - Fixed ordering.

    Research Question:
    - Does the order of ICL examples affect translation?

    Experimental Conditions:
    - Vary order: random, similarity ascending/descending, length ascending/descending.

    Independent Variable:
    - ICL example order.

    Dependent Variable:
    - BLEU, chrF, COMET.

    Confounding Variables:
    - Which ICL examples are present.

    Expected Behavior:
    - Order may impact scores due to recency/context.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["experiment_type"] = "ordering"
        super().__init__(config)

    def _experiment_specific_selection(self, doc, candidates, num_examples):
        self._rng.seed(42)
        selected = self._rng.sample(candidates, min(num_examples, len(candidates)))
        if self.ordering_strategy == "random":
            self._rng.seed(42 + hash(str(doc["id"])))
            self._rng.shuffle(selected)
        elif self.ordering_strategy == "similarity_ascending":
            test_source = doc["source_sentence"]
            selected.sort(
                key=lambda x: sentence_similarity_score(
                    test_source, x["source_sentence"]
                )
            )
        elif self.ordering_strategy == "similarity_descending":
            test_source = doc["source_sentence"]
            selected.sort(
                key=lambda x: sentence_similarity_score(
                    test_source, x["source_sentence"]
                ),
                reverse=True,
            )
        elif self.ordering_strategy == "length_ascending":
            selected.sort(key=lambda x: len(x["source_sentence"]))
        elif self.ordering_strategy == "length_descending":
            selected.sort(key=lambda x: len(x["source_sentence"]), reverse=True)
        return selected


@register_task("icl_mt_ordering_similarity_asc")
class ICLOrderingSimilarityAscTask(ICLOrderingTask):
    """Ordering: Least similar examples first"""

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["ordering_strategy"] = "similarity_ascending"
        super().__init__(config)


@register_task("icl_mt_ordering_similarity_desc")
class ICLOrderingSimilarityDescTask(ICLOrderingTask):
    """Ordering: Most similar examples first"""

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["ordering_strategy"] = "similarity_descending"
        super().__init__(config)


@register_task("icl_mt_ordering_length_asc")
class ICLOrderingLengthAscTask(ICLOrderingTask):
    """Ordering: Shortest examples first"""

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["ordering_strategy"] = "length_ascending"
        super().__init__(config)


@register_task("icl_mt_ordering_length_desc")
class ICLOrderingLengthDescTask(ICLOrderingTask):
    """Ordering: Longest examples first"""

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = {}
        config["ordering_strategy"] = "length_descending"
        super().__init__(config)
