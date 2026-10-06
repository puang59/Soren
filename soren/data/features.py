"""Node featurizers.

Each tier maps a :class:`GraphRecord` to a ``[num_nodes, F]`` float32 matrix with every value in
``[0, 1]``. Features never use node ids or absolute line numbers, and nothing here may read the
ground-truth labels.

Tier S (structural) is the state described in the project description: node kind, in/out
degree and position in the control flow.

Tier E (embedding) appends a 32-dimensional embedding of each statement's text to Tier L.

Tier L (lexical, the default) appends cheap per-statement flags to Tier S: which kind of API
is called, and which operators and identifiers appear. Operator flags read the Joern operator
names stored on the node (``Node.ops``) when the node has any, and fall back to regular
expressions over the statement text otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from soren.data.schema import NODE_KINDS, GraphRecord, Node

DEGREE_CLIP = 4

TIER_S_NAMES: tuple[str, ...] = (
    *(f"kind_{kind.lower()}" for kind in NODE_KINDS),
    "in_degree",
    "out_degree",
    "depth",
    "in_loop",
    "loop_header",
    "rel_line",
)

_KIND_INDEX = {kind: i for i, kind in enumerate(NODE_KINDS)}


def _tier_s(graph: GraphRecord, lexicon: object = None) -> np.ndarray:
    n = graph.num_nodes
    out = np.zeros((n, len(TIER_S_NAMES)), dtype=np.float32)
    col = {name: i for i, name in enumerate(TIER_S_NAMES)}

    max_depth = max(1, max(node.depth for node in graph.nodes))
    num_lines = max(1, len(graph.source_lines), max(node.line for node in graph.nodes))
    loop_headers = {target for _, target in graph.back_edges}

    for node in graph.nodes:
        row = out[node.id]
        row[_KIND_INDEX[node.kind]] = 1.0
        row[col["in_degree"]] = min(len(graph.predecessors(node.id)), DEGREE_CLIP) / DEGREE_CLIP
        row[col["out_degree"]] = min(len(graph.successors(node.id)), DEGREE_CLIP) / DEGREE_CLIP
        # Unreachable nodes carry depth -1; treat them as depth 0.
        row[col["depth"]] = max(node.depth, 0) / max_depth
        row[col["in_loop"]] = float(node.in_loop)
        row[col["loop_header"]] = float(node.id in loop_headers)
        row[col["rel_line"]] = min(max(node.line, 0) / num_lines, 1.0)
    return out


# ----------------------------------------------------------------------------- Tier L

TOKEN_CLIP = 40

LEXICAL_NAMES: tuple[str, ...] = (
    "has_call",
    "dangerous_call",
    "alloc_call",
    "free_call",
    "array_subscript",
    "pointer_deref",
    "pointer_arith",
    "arith_op",
    "comparison",
    "uses_sizeof",
    "has_cast",
    "length_like_ident",
    "token_count",
)
TIER_L_NAMES: tuple[str, ...] = (*TIER_S_NAMES, *LEXICAL_NAMES)

DEFAULT_DANGEROUS_CALLS: tuple[str, ...] = (
    "memcpy",
    "memmove",
    "memset",
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "sprintf",
    "snprintf",
    "vsprintf",
    "vsnprintf",
    "gets",
    "fgets",
    "scanf",
    "sscanf",
    "read",
    "recv",
    "recvfrom",
    "fread",
    "copy_from_user",
    "copy_to_user",
    "alloca",
    "bcopy",
)
DEFAULT_ALLOC_CALLS: tuple[str, ...] = (
    "malloc",
    "calloc",
    "realloc",
    "kmalloc",
    "kzalloc",
    "kcalloc",
    "krealloc",
    "vmalloc",
    "vzalloc",
    "kvmalloc",
    "av_malloc",
    "g_malloc",
    "new",
)
DEFAULT_FREE_CALLS: tuple[str, ...] = (
    "free",
    "kfree",
    "vfree",
    "kvfree",
    "av_free",
    "g_free",
    "delete",
)


@dataclass(frozen=True)
class Lexicon:
    """API name lists behind the call flags; overridable from ``configs/data.yaml``."""

    dangerous_calls: tuple[str, ...] = DEFAULT_DANGEROUS_CALLS
    alloc_calls: tuple[str, ...] = DEFAULT_ALLOC_CALLS
    free_calls: tuple[str, ...] = DEFAULT_FREE_CALLS
    _sets: dict[str, frozenset[str]] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("dangerous_calls", "alloc_calls", "free_calls"):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            self._sets[name] = frozenset(values)

    def matches(self, name: str, calls: list[str]) -> bool:
        return not self._sets[name].isdisjoint(calls)


DEFAULT_LEXICON = Lexicon()

_OP = "<operator>."
_OPS_SUBSCRIPT = {_OP + "indirectIndexAccess"}
_OPS_DEREF = {_OP + "indirection", _OP + "indirectFieldAccess"}
_OPS_ARITH = {
    _OP + name
    for name in (
        "addition",
        "subtraction",
        "multiplication",
        "division",
        "modulo",
        "shiftLeft",
        "arithmeticShiftRight",
        "logicalShiftRight",
        "preIncrement",
        "postIncrement",
        "preDecrement",
        "postDecrement",
        "assignmentPlus",
        "assignmentMinus",
        "assignmentMultiplication",
        "assignmentDivision",
    )
}
_OPS_COMPARE = {
    _OP + name
    for name in (
        "lessThan",
        "greaterThan",
        "lessEqualsThan",
        "greaterEqualsThan",
        "equals",
        "notEquals",
    )
}
_OPS_SIZEOF = {_OP + "sizeOf"}
_OPS_CAST = {_OP + "cast"}

_RE_SUBSCRIPT = re.compile(r"[\w)\]]\s*\[")
_RE_DEREF = re.compile(r"->|(?:^|[=(,;{}!&|?:\[]|\breturn\b)\s*\*+\s*\(?\s*\w")
_RE_ARITH = re.compile(
    r"[\w)\]]\s*(?:\+(?!\+)|-(?![->])|/|%|<<|>>)\s*[\w(*&]"  # a + b, a - b, a << b ...
    r"|[\w)\]]\s+\*\s+[\w(]"  # a * b (spaced, to skip pointer declarations)
    r"|\+\+|--|[+\-*/%]="
)
_RE_COMPARE = re.compile(r"[<>=!]=|(?<![<>=!-])[<>](?![<>=])")
_RE_SIZEOF = re.compile(r"\bsizeof\b")
_RE_CAST = re.compile(
    r"(?<!sizeof)\(\s*(?:const\s+)?"
    r"(?:unsigned|signed|char|short|int|long|float|double|void|struct\s+\w+|\w+_t|[us](?:8|16|32|64))"
    r"\b[\w\s]*\**\s*\)\s*[\w(*&-]"
)
# Arithmetic on something that is then dereferenced, indexed through, or stepped as a pointer.
_RE_POINTER_ARITH = re.compile(
    r"\*\s*\(\s*\w+\s*[+-]"  # *(p + n)
    r"|\*\s*(?:\+\+|--)\s*\w|\*\s*\w+\s*(?:\+\+|--)"  # *++p, *p++
    r"|\(\s*\w+\s*[+-]\s*\w+\s*\)\s*(?:->|\[)"  # (p + n)->f
    r"|\b\w*(?:ptr|buf|pos|cur|end|data)\w*\s*(?:\+=|-=|\+\+|--)"  # ptr += n
)
_RE_IDENT = re.compile(r"[A-Za-z_]\w*")
_RE_TOKEN = re.compile(r"\w+|[^\w\s]")
_LENGTH_PARTS = {
    "len",
    "length",
    "size",
    "sz",
    "count",
    "cnt",
    "num",
    "idx",
    "index",
    "offset",
    "off",
    "pos",
}
_LENGTH_SUFFIXES = ("len", "size", "count", "idx", "offset")


def _is_length_like(identifier: str) -> bool:
    lowered = identifier.lower()
    if lowered == "sizeof":
        return False
    if any(part in _LENGTH_PARTS for part in re.split(r"_|\d+", lowered)):
        return True
    return lowered.endswith(_LENGTH_SUFFIXES)


def lexical_flags(node: Node, lexicon: Lexicon = DEFAULT_LEXICON) -> dict[str, float]:
    """The Tier L flags of one node, keyed by :data:`LEXICAL_NAMES`."""
    code = node.code
    ops = set(node.ops)
    calls = [call for call in node.calls if not call.startswith(_OP)]

    def has(op_names: set[str], pattern: re.Pattern[str]) -> bool:
        # Trust Joern's operators when the node has any; otherwise fall back to the text.
        return bool(ops & op_names) if ops else bool(pattern.search(code))

    return {
        "has_call": float(bool(calls)),
        "dangerous_call": float(lexicon.matches("dangerous_calls", calls)),
        "alloc_call": float(lexicon.matches("alloc_calls", calls)),
        "free_call": float(lexicon.matches("free_calls", calls)),
        "array_subscript": float(has(_OPS_SUBSCRIPT, _RE_SUBSCRIPT)),
        "pointer_deref": float(has(_OPS_DEREF, _RE_DEREF)),
        # Joern does not mark pointer arithmetic, so this one is always a text heuristic.
        "pointer_arith": float(bool(_RE_POINTER_ARITH.search(code))),
        "arith_op": float(has(_OPS_ARITH, _RE_ARITH)),
        "comparison": float(has(_OPS_COMPARE, _RE_COMPARE)),
        "uses_sizeof": float(has(_OPS_SIZEOF, _RE_SIZEOF)),
        "has_cast": float(has(_OPS_CAST, _RE_CAST)),
        "length_like_ident": float(any(_is_length_like(i) for i in _RE_IDENT.findall(code))),
        "token_count": min(len(_RE_TOKEN.findall(code)), TOKEN_CLIP) / TOKEN_CLIP,
    }


def _tier_l(graph: GraphRecord, lexicon: Lexicon) -> np.ndarray:
    structural = _tier_s(graph, lexicon)
    lexical = np.zeros((graph.num_nodes, len(LEXICAL_NAMES)), dtype=np.float32)
    for node in graph.nodes:
        flags = lexical_flags(node, lexicon)
        lexical[node.id] = [flags[name] for name in LEXICAL_NAMES]
    return np.concatenate([structural, lexical], axis=1)


# ----------------------------------------------------------------------------- Tier E

EMBED_KEY = "embed"
EMBED_DIM = 32
TIER_E_NAMES: tuple[str, ...] = (*TIER_L_NAMES, *(f"embed_{i}" for i in range(EMBED_DIM)))


def _tier_e(graph: GraphRecord, lexicon: Lexicon) -> np.ndarray:
    """Tier L plus the statement embeddings stored on the record.

    Embeddings come from a pretrained model and are computed offline by
    ``scripts/07_embed_statements.py``; a record without them cannot be featurized at Tier E.
    """
    stored = graph.features.get(EMBED_KEY)
    if stored is None:
        raise ValueError(
            f"{graph.sample_id}: no statement embeddings; run scripts/07_embed_statements.py"
        )
    embeddings = np.asarray(stored, dtype=np.float32)
    if embeddings.shape != (graph.num_nodes, EMBED_DIM):
        raise ValueError(
            f"{graph.sample_id}: embeddings have shape {embeddings.shape}, "
            f"expected {(graph.num_nodes, EMBED_DIM)}"
        )
    return np.concatenate([_tier_l(graph, lexicon), embeddings], axis=1)


# --------------------------------------------------------------------------- registry

_Featurizer = Callable[[GraphRecord, Lexicon], np.ndarray]
_TIERS: dict[str, tuple[tuple[str, ...], _Featurizer]] = {
    "S": (TIER_S_NAMES, _tier_s),
    "L": (TIER_L_NAMES, _tier_l),
    "E": (TIER_E_NAMES, _tier_e),
}


def _lookup(tier: str) -> tuple[tuple[str, ...], _Featurizer]:
    try:
        return _TIERS[tier]
    except KeyError:
        raise ValueError(f"unknown feature tier {tier!r}; available: {sorted(_TIERS)}") from None


def feature_names(tier: str) -> tuple[str, ...]:
    return _lookup(tier)[0]


def feature_dim(tier: str) -> int:
    return len(feature_names(tier))


def featurize(graph: GraphRecord, tier: str = "L", lexicon: Lexicon | None = None) -> np.ndarray:
    """Return the ``[num_nodes, feature_dim(tier)]`` feature matrix for ``graph``."""
    return _lookup(tier)[1](graph, lexicon or DEFAULT_LEXICON)
