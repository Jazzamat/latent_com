from collections import Counter

import torch

from config import ModelConfig
from data import CharTokenizer, TextDataset
from model import TinyLM
from utils import set_seed


TOM_CASES = [
    {
        "name": "sally_anne",
        "prompt": (
            "Q: Sally puts a marble in the basket. Anne moves it to the box while Sally is away. "
            "Where will Sally look first?\nA:"
        ),
        "answer": "Sally will look in the basket first.",
        "belief": "basket",
        "reality": "box",
    },
    {
        "name": "ben_mia",
        "prompt": (
            "Q: Ben leaves his toy in the drawer. Mia moves it to the shelf when Ben cannot see. "
            "Where does Ben think the toy is?\nA:"
        ),
        "answer": "Ben thinks the toy is in the drawer.",
        "belief": "drawer",
        "reality": "shelf",
    },
    {
        "name": "nora_leo",
        "prompt": (
            "Q: Nora hides a cookie in the jar. Leo moves it to the tin while Nora is outside. "
            "Where will Nora search first?\nA:"
        ),
        "answer": "Nora will search in the jar first.",
        "belief": "jar",
        "reality": "tin",
    },
    {
        "name": "omar_rina",
        "prompt": (
            "Q: Omar puts his keys on the table. Rina moves them to the bag while Omar is gone. "
            "Where does Omar believe the keys are?\nA:"
        ),
        "answer": "Omar believes the keys are on the table.",
        "belief": "table",
        "reality": "bag",
    },
    {
        "name": "ava_kai",
        "prompt": (
            "Q: Ava places the book on the desk. Kai moves it to the cabinet while Ava is in another room. "
            "Where will Ava look first?\nA:"
        ),
        "answer": "Ava will look on the desk first.",
        "belief": "desk",
        "reality": "cabinet",
    },
]

TOM_TRAIN_STEPS = 500


def _tag_response(response: str, belief_location: str, reality_location: str) -> str:
    normalized = response.lower()
    belief_hit = belief_location.lower() in normalized
    reality_hit = reality_location.lower() in normalized

    if belief_hit and not reality_hit:
        return "belief-consistent"
    if reality_hit and not belief_hit:
        return "reality-consistent"
    return "unclear"


def _train_tom_demo_model(device: str) -> tuple[TinyLM, CharTokenizer]:
    set_seed(321)

    corpus_rows = [f"{case['prompt']} {case['answer']}" for case in TOM_CASES]
    text = ("\n\n".join(corpus_rows) + "\n") * 120
    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        n_layers=2,
        n_heads=2,
        d_model=64,
        mlp_hidden=128,
        dropout=0.0,
    )
    dataset = TextDataset(encoded, block_size=model_config.block_size, train_split=0.9)
    model = TinyLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    model.train()
    for _step in range(TOM_TRAIN_STEPS):
        xb, yb = dataset.get_batch("train", batch_size=16, device=device)
        _logits, loss, _hidden = model(xb, yb)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    return model, tokenizer


def test_theory_of_mind_qualitative_report(blackbox_device: str) -> None:
    device = blackbox_device
    model, tokenizer = _train_tom_demo_model(device=device)

    results = []
    for case in TOM_CASES:
        prompt = case["prompt"]
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=96, temperature=1.0, top_k=1)
        decoded = tokenizer.decode(out[0].detach().cpu().tolist())

        completion = decoded[len(prompt):]
        response = completion.split("\n", 1)[0].strip()
        tag = _tag_response(response, case["belief"], case["reality"])

        results.append(
            {
                "name": case["name"],
                "prompt": prompt,
                "response": response,
                "tag": tag,
            }
        )

    print("\nTheory-of-mind qualitative report")
    print("--------------------------------")
    for row in results:
        print(f"[{row['name']}] {row['tag']}")
        print(f"prompt: {row['prompt']}")
        print(f"response: {row['response']}\n")

    counts = Counter(row["tag"] for row in results)
    print(f"Tag counts: {dict(counts)}")

    assert len(results) == 5
    assert all(row["response"] for row in results)
    assert all(
        row["tag"] in {"belief-consistent", "reality-consistent", "unclear"}
        for row in results
    )
    assert counts["belief-consistent"] >= 3
