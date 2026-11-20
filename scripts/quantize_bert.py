import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from chatmodel.models.bert import BertModel, artifact_digest
from chatmodel.models.quantized_bert import QuantizedBertModel
from chatmodel.paths import BERT_DIRECTORY, QUANTIZED_BERT_MANIFEST


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def main():
    destination = QUANTIZED_BERT_MANIFEST.parent
    if destination.exists():
        raise FileExistsError(f"Model directory already exists: {destination}")
    source = BertModel.load(BERT_DIRECTORY)
    with tempfile.TemporaryDirectory(prefix="bert-int8-", dir=destination.parent) as scratch:
        temporary = Path(scratch)
        stage = temporary / "artifact"
        stage.mkdir()
        fp32 = temporary / "fp32.onnx"
        quantized = stage / "model.onnx"
        examples = ["thank you for helping", "you are a stupid idiot", "hello"]
        inputs = source.tokenizer(examples, padding=True, return_tensors="pt")
        names = ["input_ids", "attention_mask", "token_type_ids"]
        print("Exporting fine-tuned BERT to ONNX", flush=True)
        with torch.no_grad():
            torch.onnx.export(
                Logits(source.model).eval(), tuple(inputs[name] for name in names), fp32,
                input_names=names, output_names=["logits"], opset_version=17, dynamo=False,
                dynamic_axes={**{name: {0: "batch", 1: "sequence"} for name in names},
                              "logits": {0: "batch"}},
            )
        print("Quantizing constant MatMul weights and embedding tables", flush=True)
        quantize_dynamic(
            fp32, quantized, weight_type=QuantType.QInt8, per_channel=True,
            op_types_to_quantize=["MatMul", "Gather"],
        )
        graph = onnx.load(quantized)
        integer_matmuls = sum(node.op_type == "MatMulInteger" for node in graph.graph.node)
        if not integer_matmuls:
            raise ValueError("ONNX graph contains no INT8 matrix multiplication.")
        del graph
        source.tokenizer.save_pretrained(stage)
        manifest = {
            "format": "onnx-dynamic-int8",
            "metadata": source.metadata,
            "threshold": source.threshold,
            "cpu_threads": 4,
            "source_artifact_sha256": artifact_digest(BERT_DIRECTORY),
            "integer_matmuls": integer_matmuls,
            "onnxruntime_version": ort.__version__,
        }
        (stage / QUANTIZED_BERT_MANIFEST.name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        model = QuantizedBertModel(stage)
        difference = np.max(np.abs(model.batch_scores(examples) - source.batch_scores(examples)))
        if difference > 0.05:
            raise ValueError(f"INT8 smoke test probability discrepancy: {difference:.4f}")
        stage.rename(destination)
        print(f"INT8 model ready: {integer_matmuls} integer matmuls, smoke difference {difference:.4f}", flush=True)


if __name__ == "__main__":
    main()
