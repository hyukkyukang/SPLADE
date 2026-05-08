import unittest

import torch
from omegaconf import OmegaConf

from src.data.dataclass import MetaItem
from src.data.pd_module.utils import build_rerank_inputs


class _DummyDataset:
    def __init__(self) -> None:
        self._query_texts = {"q1": "query alpha beta"}
        self._doc_texts = {
            "d1": "doc gamma delta",
            "d2": "doc epsilon zeta",
        }

    def resolve_query_text(self, meta_item: MetaItem) -> str:
        return self._query_texts[meta_item.qid]

    def resolve_doc_texts(
        self,
        doc_ids: list[str],
        inline_texts: list[str] | None,
    ) -> list[str]:
        _ = inline_texts
        return [self._doc_texts[doc_id] for doc_id in doc_ids]


class _DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.mask_token_id = 4
        self.cls_token_id = 101
        self.sep_token_id = 102
        self._vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 101,
            "[SEP]": 102,
            "[MASK]": 4,
            "query": 10,
            "alpha": 11,
            "beta": 12,
            "doc": 13,
            "gamma": 14,
            "delta": 15,
            "epsilon": 16,
            "zeta": 17,
        }

    def _encode(self, text: str) -> list[int]:
        token_ids: list[int] = [self.cls_token_id]
        token_ids.extend(self._vocab.get(piece, 1) for piece in text.split())
        token_ids.append(self.sep_token_id)
        return token_ids

    def __call__(
        self,
        text,
        *,
        add_special_tokens: bool = True,
        padding=False,
        truncation: bool = True,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        encoded_rows: list[list[int]] = []
        for row_text in texts:
            row_ids = self._encode(row_text) if add_special_tokens else [
                self._vocab.get(piece, 1) for piece in row_text.split()
            ]
            if truncation and max_length is not None:
                row_ids = row_ids[:max_length]
            encoded_rows.append(row_ids)

        if return_tensors == "pt":
            if padding == "max_length":
                assert max_length is not None
                resolved_length = int(max_length)
            elif padding is True:
                resolved_length = max(len(row) for row in encoded_rows)
            else:
                raise AssertionError("This dummy tokenizer only supports padded PT outputs.")
            padded_ids = []
            padded_masks = []
            for row in encoded_rows:
                row = row[:resolved_length]
                pad_len = resolved_length - len(row)
                padded_ids.append(row + ([self.pad_token_id] * pad_len))
                padded_masks.append(([1] * len(row)) + ([0] * pad_len))
            output = {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            }
            return output

        output = {"input_ids": encoded_rows}
        if return_attention_mask:
            output["attention_mask"] = [[1] * len(row) for row in encoded_rows]
        return output


class _DummyTermSupervisor:
    def top_k_query_target_ids(
        self,
        text: str,
        *,
        k: int,
        ignore_index: int,
    ) -> torch.Tensor:
        _ = text
        values = [21, 22][:k]
        while len(values) < k:
            values.append(ignore_index)
        return torch.tensor(values, dtype=torch.long)

    def top_k_doc_target_ids(
        self,
        text: str,
        *,
        k: int,
        ignore_index: int,
    ) -> torch.Tensor:
        _ = text
        values = [31, 32][:k]
        while len(values) < k:
            values.append(ignore_index)
        return torch.tensor(values, dtype=torch.long)


class OrderedMaskSlotDataUtilsTest(unittest.TestCase):
    def test_build_rerank_inputs_appends_mask_slots_and_assigns_targets(self) -> None:
        dataset = _DummyDataset()
        tokenizer = _DummyTokenizer()
        model_cfg = OmegaConf.create(
            {
                "family": "ordered_mask_slot_splade",
                "num_mask_slots": 2,
                "mask_token_id": 4,
            }
        )
        meta_item = MetaItem(
            qid="q1",
            pos_ids=["d1"],
            neg_ids=["d2"],
            pos_scores=None,
            neg_scores=None,
        )

        inputs = build_rerank_inputs(
            dataset=dataset,
            tokenizer=tokenizer,
            meta_item=meta_item,
            model_cfg=model_cfg,
            max_query_length=8,
            max_doc_length=8,
            max_padding=True,
            term_supervision=_DummyTermSupervisor(),
            term_supervision_ignore_index=-100,
        )

        query_active_len = int(inputs.query_attention_mask.sum().item())
        doc_active_len = int(inputs.doc_attention_mask[0].sum().item())
        self.assertEqual(
            inputs.query_input_ids[query_active_len - 2 : query_active_len].tolist(),
            [4, 4],
        )
        self.assertEqual(
            inputs.query_pooling_mask[query_active_len - 2 : query_active_len].tolist(),
            [1, 1],
        )
        self.assertEqual(
            inputs.doc_input_ids[0, doc_active_len - 2 : doc_active_len].tolist(),
            [4, 4],
        )
        self.assertEqual(
            inputs.doc_pooling_mask[0, doc_active_len - 2 : doc_active_len].tolist(),
            [1, 1],
        )
        self.assertTrue(torch.equal(inputs.query_slot_target_ids, torch.tensor([31, 32])))
        self.assertTrue(
            torch.equal(
                inputs.doc_slot_target_ids,
                torch.tensor([[21, 22], [-100, -100]], dtype=torch.long),
            )
        )
        self.assertTrue(torch.equal(inputs.pos_mask, torch.tensor([True, False])))
        self.assertTrue(torch.equal(inputs.doc_mask, torch.tensor([True, True])))


if __name__ == "__main__":
    unittest.main()
