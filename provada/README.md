# ProVADA Package

This directory contains the core ProVADA package implementation.

## Package Structure

### `components/`
Core computational components that form the building blocks of the ProVADA system:
- Evaluators for scoring and validation
- Processors for data transformation
- Pipeline components for workflow orchestration

### `models/`
Machine learning models and wrappers:
- Model implementations and interfaces
- Pre-trained model integrations
- Model-specific utilities and configurations

### `sampler/`
Sampling algorithms and strategies:
- Monte Carlo sampling methods
- Adaptive sampling techniques
- Sampling configuration and management

### `sequences/`
Sequence processing and manipulation:
- Protein sequence handling
- Sequence analysis tools
- Sequence generation and modification

### `utils/`
Utility functions and helpers:
- Logging and configuration
- File I/O operations
- Common helper functions

## Key Files

- `base_variant.py` - Base classes and interfaces for variant handling
- `paths.py` - Path configuration and management
- `__init__.py` - Package initialization