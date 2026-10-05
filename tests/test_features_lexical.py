from pathlib import Path

import numpy as np
import pytest
from helpers import diamond_graph

from soren.config import load_config
from soren.data.features import (
    DEFAULT_LEXICON,
    LEXICAL_NAMES,
    TIER_L_NAMES,
    TIER_S_NAMES,
    Lexicon,
    feature_dim,
    featurize,
    lexical_flags,
)
from soren.data.schema import Node
from soren.data.synthetic import DANGEROUS_CALLS, generate_dataset

OP = "<operator>."


def flags(code: str, calls=(), ops=()) -> dict[str, float]:
    return lexical_flags(
        Node(id=1, line=1, kind="OTHER", code=code, calls=list(calls), ops=list(ops))
    )


# (flag, statements that must set it, statements that must not) using the text fallback.
TEXT_CASES = [
    (
        "array_subscript",
        ["buf[i] = 0;", "x = table[idx + 1];", "p->arr[3]++;"],
        ["x = y + 1;", "f(a, b);"],
    ),
    (
        "pointer_deref",
        ["*p = 0;", "x = *p;", "len = hdr->len;", "return *ptr;", "f(*p);", "**pp = q;"],
        ["x = a * b;", "char *p;", "n = len * 2;", "x = y;"],
    ),
    (
        "pointer_arith",
        ["*(p + 4) = 0;", "x = *p++;", "ptr += len;", "buf++;", "c = (p + n)->next;"],
        ["x = a + b;", "i++;", "n = len * 2;", "*p = 0;"],
    ),
    (
        "arith_op",
        [
            "x = a + b;",
            "n = len - 1;",
            "i++;",
            "total += n;",
            "x = a * b;",
            "y = n << 2;",
            "q = a / b;",
        ],
        ["x = y;", "len = hdr->len;", "char *p;", "f(a, b);", "*p = 0;"],
    ),
    (
        "comparison",
        [
            "if (i < n) {",
            "while (len >= size) {",
            "if (a == b) {",
            "if (p != NULL) {",
            "x = a > b;",
        ],
        ["x = y;", "len = hdr->len;", "y = n << 2;", "x = a >> 3;", "total += n;"],
    ),
    ("uses_sizeof", ["n = sizeof(buf);", "memset(p, 0, sizeof *p);"], ["size = n;", "x = y;"]),
    (
        "has_cast",
        [
            "x = (int)y;",
            "p = (char *)buf;",
            "n = (size_t)len;",
            "q = (struct foo *)p;",
            "v = (u32)x;",
        ],
        ["if (x) y = 1;", "f(a);", "n = sizeof(int) * 2;", "x = (a + b) * c;"],
    ),
    (
        "length_like_ident",
        [
            "len = 0;",
            "buf_size = n;",
            "x = pkt->data_len;",
            "i = start_idx;",
            "n = buflen;",
            "off2 = 1;",
        ],
        ["x = y;", "n = sizeof(buf);", "lens = 2;", "position = 1;"],
    ),
]


@pytest.mark.parametrize(("name", "positives", "negatives"), TEXT_CASES)
def test_text_fallback(name, positives, negatives):
    for code in positives:
        assert flags(code)[name] == 1.0, f"{name} should fire on {code!r}"
    for code in negatives:
        assert flags(code)[name] == 0.0, f"{name} should not fire on {code!r}"


OP_CASES = [
    ("array_subscript", "indirectIndexAccess"),
    ("pointer_deref", "indirection"),
    ("pointer_deref", "indirectFieldAccess"),
    ("arith_op", "addition"),
    ("arith_op", "shiftLeft"),
    ("comparison", "lessThan"),
    ("comparison", "notEquals"),
    ("uses_sizeof", "sizeOf"),
    ("has_cast", "cast"),
]


@pytest.mark.parametrize(("name", "op"), OP_CASES)
def test_joern_operators_set_flags(name, op):
    assert flags("opaque", ops=[OP + op])[name] == 1.0
    assert flags("opaque", ops=[OP + "assignment"])[name] == 0.0


def test_operators_take_precedence_over_text():
    # The text looks like a dereference and a subscript, but Joern saw only a multiplication.
    result = flags("x = a *b[0];", ops=[OP + "assignment", OP + "multiplication"])
    assert result["arith_op"] == 1.0
    assert result["pointer_deref"] == 0.0
    assert result["array_subscript"] == 0.0


def test_call_flags():
    dangerous = flags("memcpy(d, s, n);", calls=["memcpy"])
    assert dangerous["has_call"] == 1.0 and dangerous["dangerous_call"] == 1.0
    assert dangerous["alloc_call"] == 0.0 and dangerous["free_call"] == 0.0

    alloc = flags("p = kmalloc(n, GFP_KERNEL);", calls=["kmalloc"])
    assert alloc["alloc_call"] == 1.0 and alloc["dangerous_call"] == 0.0

    freed = flags("kfree(p);", calls=["kfree"])
    assert freed["free_call"] == 1.0 and freed["alloc_call"] == 0.0

    benign = flags("log_msg(x);", calls=["log_msg"])
    assert benign["has_call"] == 1.0
    assert benign["dangerous_call"] == benign["alloc_call"] == benign["free_call"] == 0.0

    assert flags("x = y;")["has_call"] == 0.0
    # Joern reports operators as calls; they are not API calls.
    assert flags("x = y + 1;", calls=[OP + "addition"])["has_call"] == 0.0


def test_call_flags_use_callee_names_not_text():
    assert flags("my_memcpy_wrapper(d, s);", calls=["my_memcpy_wrapper"])["dangerous_call"] == 0.0


def test_token_count_is_clipped():
    assert flags("")["token_count"] == 0.0
    assert flags("x = y;")["token_count"] == pytest.approx(4 / 40)
    assert flags(" + ".join(["a"] * 60))["token_count"] == 1.0


def test_custom_lexicon():
    lexicon = Lexicon(dangerous_calls=("my_copy",))
    node = Node(id=1, line=1, kind="CALL", code="my_copy(a);", calls=["my_copy"])
    assert lexical_flags(node, lexicon)["dangerous_call"] == 1.0
    assert lexical_flags(node)["dangerous_call"] == 0.0
    memcpy = Node(id=1, line=1, kind="CALL", code="memcpy(a);", calls=["memcpy"])
    assert lexical_flags(memcpy, lexicon)["dangerous_call"] == 0.0


def test_config_lexicon_matches_the_defaults():
    path = Path(__file__).parent.parent / "configs" / "data.yaml"
    assert load_config(Lexicon, path, section="lexicon") == DEFAULT_LEXICON


def test_tier_l_extends_tier_s():
    assert (*TIER_S_NAMES, *LEXICAL_NAMES) == TIER_L_NAMES
    assert feature_dim("L") == feature_dim("S") + 13 == 30
    graph = diamond_graph()
    lexical = featurize(graph, "L")
    assert lexical.shape == (graph.num_nodes, 30)
    assert lexical.dtype == np.float32
    np.testing.assert_array_equal(lexical[:, : feature_dim("S")], featurize(graph, "S"))


def test_tier_l_on_synthetic_graphs_exposes_the_signal():
    col = TIER_L_NAMES.index("dangerous_call")
    for graph in generate_dataset(60, seed=4):
        feats = featurize(graph, "L")
        assert not np.isnan(feats).any()
        assert feats.min() >= 0.0 and feats.max() <= 1.0
        for node in graph.nodes:
            expected = any(call in DANGEROUS_CALLS for call in node.calls)
            assert feats[node.id, col] == float(expected)


def test_tier_l_ignores_labels():
    np.testing.assert_array_equal(
        featurize(diamond_graph(vuln=2), "L"), featurize(diamond_graph(vuln=3), "L")
    )
