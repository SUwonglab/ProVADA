"""Generic generator row alignment, metadata and output validation."""
import pandas as pd
import pytest

from provada.components.generator import Generator


class FrameGenerator(Generator):
    def __init__(self, output):
        self.output = output
        self.calls = []

    def _generate(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return self.output


@pytest.mark.parametrize("index", [[7, 7], [30, 4], ["second", "first"]])
def test_generic_generator_aligns_rows_by_position_and_preserves_metadata(index):
    frame = pd.DataFrame({"prompt": ["A__", "C__"], "parent_particle_id": [8, 2],
                          "starting_SCORE": [.8, .2]}, index=index)
    original = frame.copy(deep=True)
    output = pd.DataFrame({"sequence": ["AAA", "CCC"], "model_score": [.1, .9]}, index=[90, 10])
    generator = FrameGenerator(output)
    result = generator.generate(frame, temperature=.7)
    assert result.index.tolist() == [0, 1]
    assert result.sequence.tolist() == ["AAA", "CCC"]
    assert result.parent_particle_id.tolist() == [8, 2]
    assert result.starting_SCORE.tolist() == [.8, .2]
    assert result.prompt.tolist() == ["A__", "C__"]
    assert generator.calls == [(["A__", "C__"], {"temperature": .7})]
    pd.testing.assert_frame_equal(frame, original)


def test_generated_sequences_and_scores_replace_stale_input_columns():
    frame = pd.DataFrame({"prompt": ["A__", "C__"], "sequence": ["OLD", "OLD"],
                          "model_score": [-10., -20.], "parent_particle_id": [8, 2]})
    original = frame.copy(deep=True)
    output = [{"sequence": "AAA", "model_score": .1, "prompt": "A__"},
              {"sequence": "CCC", "model_score": .9, "prompt": "C__"}]
    result = FrameGenerator(output).generate(frame)
    assert result.sequence.tolist() == ["AAA", "CCC"]
    assert result.model_score.tolist() == [.1, .9]
    assert result.prompt.tolist() == ["A__", "C__"]
    assert result.parent_particle_id.tolist() == [8, 2]
    assert result.columns.is_unique
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_generic_generator_rejects_wrong_output_row_count(count):
    output = [{"sequence": "AAA"}] * count
    with pytest.raises(ValueError, match=f"returned {count} rows for 2 prompts"):
        FrameGenerator(output).generate(["A__", "C__"])


def test_generic_generator_rejects_output_missing_sequence_even_with_old_input_sequence():
    frame = pd.DataFrame({"prompt": ["A__"], "sequence": ["OLD"]})
    with pytest.raises(ValueError, match="output must contain a 'sequence'"):
        FrameGenerator([{"model_score": .5}]).generate(frame)


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_empty_generator_input_skips_backend_and_has_sequence_column(as_dataframe):
    generator = FrameGenerator([])
    prompts = pd.DataFrame(columns=["prompt", "parent_particle_id"]) if as_dataframe else []
    result = generator.generate(prompts)
    assert result.empty
    assert {"prompt", "sequence"} <= set(result.columns)
    if as_dataframe:
        assert "parent_particle_id" in result
    assert generator.calls == []


def test_generic_generator_rejects_missing_prompt_column_before_backend():
    generator = FrameGenerator([])
    with pytest.raises(ValueError, match="must contain a 'prompt'"):
        generator.generate(pd.DataFrame({"sequence": ["AAA"]}))
    assert generator.calls == []


def test_generic_generator_rejects_ambiguous_input_columns():
    generator = FrameGenerator([])
    frame = pd.DataFrame([["A__", "C__"]], columns=["prompt", "prompt"])
    with pytest.raises(ValueError, match="unique column names"):
        generator.generate(frame)
    assert generator.calls == []


def test_generic_generator_rejects_ambiguous_output_columns():
    output = pd.DataFrame([["AAA", "CCC"]], columns=["sequence", "sequence"])
    with pytest.raises(ValueError, match="output must have unique column names"):
        FrameGenerator(output).generate(["A__"])
