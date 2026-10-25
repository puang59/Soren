import io

import pytest

from soren.agents.ppo import PPOConfig
from soren.agents.train import train
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.mathlog import MathLog, log
from soren.walkthrough import main


@pytest.fixture
def captured_log():
    stream = io.StringIO()
    old = (log.enabled, log._stream)
    log.enabled, log._stream = True, stream
    yield stream
    log.enabled, log._stream = old


def test_formula_layout():
    stream = io.StringIO()
    out = MathLog(enabled=True, stream=stream)
    out.formula("ratio", "r = exp(a − b)", "exp(0.1 − 0.0)", "1.105")
    out.formula("p", "", "1 / 2", "0.5")
    lines = stream.getvalue().splitlines()
    assert lines[0].split() == ["ratio", "r", "=", "exp(a", "−", "b)"]
    assert lines[1].strip() == "= exp(0.1 − 0.0)" and lines[2].strip() == "= 1.105"
    assert lines[3].split() == ["p", "=", "1", "/", "2", "=", "0.5"]
    MathLog(enabled=False, stream=stream).formula("x", "y")  # disabled: writes nothing
    assert len(stream.getvalue().splitlines()) == 4


def test_walkthrough_runs_every_part(captured_log, tmp_path):
    main(["--synthetic", "--checkpoint", str(tmp_path / "none.npz"), "--results", "none"])
    text = captured_log.getvalue()
    for heading in ("1. The function", "2. Its control flow graph", "3. The features",
                    "4. The observation vector", "5. One decision", "6. The whole episode",
                    "7. The return", "8. How PPO learns", "9. The evaluation"):  # fmt: skip
        assert heading in text, heading
    assert "✓ the same as what the record stores" in text
    assert "softmax" in text and "TD error" in text and "Adam step" in text


def test_training_explains_an_update(captured_log, tmp_path):
    graphs = generate_dataset(12, seed=0, cfg=SyntheticConfig(min_nodes=8, max_nodes=14))
    cfg = PPOConfig(
        total_timesteps=256, n_envs=2, n_steps=64, batch_size=64, n_epochs=2, net_arch=[16],
        eval_freq=256,
    )  # fmt: skip
    train(graphs[:8], graphs[8:], tmp_path / "run", cfg, explain_every=1)
    text = captured_log.getvalue()
    assert text.count("PPO update") == 2
    for piece in ("normalised reward", "TD error", "probability ratio", "clipped surrogate",
                  "global gradient norm", "second moment", "explained variance"):  # fmt: skip
        assert piece in text, piece
