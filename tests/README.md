# Testing Guide

## Weights & Biases (wandb) Control

Tests can be run with wandb disabled, which is essential for CI/CD environments where authentication or network access may not be available.

### Disable wandb during tests

```bash
pytest --no-wandb
```

When this flag is used:
- The `WANDB_MODE` environment variable is set to `disabled`
- All wandb logging is suppressed
- Tests run without requiring wandb authentication
- No data is sent to wandb servers

## Marking Tests That Require GPU

To mark a test as requiring a GPU, use the `@pytest.mark.requires_gpu` decorator:

```python
import pytest

@pytest.mark.requires_gpu
def test_model_inference():
    # This test will be skipped when running with --skip-gpu
    model = SomeGPUModel()
    result = model.predict(data)
    assert result is not None


@pytest.mark.requires_gpu
def test_gpu_training():
    # Another GPU-requiring test
    trainer = Trainer(device="cuda")
    trainer.train()
```

You can also mark entire test classes:

```python
import pytest

@pytest.mark.requires_gpu
class TestGPUOperations:
    def test_forward_pass(self):
        # All tests in this class require GPU
        pass

    def test_backward_pass(self):
        pass
```

## Running Tests

### Run all tests (default behavior)
```bash
pytest
```

### Skip GPU tests
Run only tests that do NOT require a GPU:
```bash
pytest --skip-gpu
```

### Run only GPU tests
Run only tests that require a GPU:
```bash
pytest --gpu-only
```

### Specify GPU device
You can control which GPU device(s) to use:
```bash
# Use GPU 0
pytest --device 0

# Use GPU 1
pytest --device 1

# Use multiple GPUs
pytest --device 0,1
```

### Disable wandb
Disable Weights & Biases logging:
```bash
pytest --no-wandb
```

### Examples of Combined Options
```bash
# Run only GPU tests on device 1 without wandb
pytest --gpu-only --device 1 --no-wandb

# Run non-GPU tests without wandb (ideal for CI)
pytest --skip-gpu --no-wandb

# Run all tests on device 0 without wandb
pytest --device 0 --no-wandb -v
```