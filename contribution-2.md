# Contribution #955: [MPS] torchao low-bit-precision optim does not expose `backend` argument to `torch.compile`

**Contribution Number:** 2  
**Student:** Potri Abhisri Barama  
**Fork:** https://github.com/Abhisri436/ao  
**Issue:** https://github.com/pytorch/ao/issues/955  
**My Branch:** https://github.com/Abhisri436/ao/tree/fix-issue-955-compile-backend  
**Pull Request:** https://github.com/pytorch/ao/pull/4686  
**Status:** Phase IV Complete — PR submitted, awaiting review

---

## Problem Summary

torchao’s low-bit Adam optimizers internally rely on `torch.compile`, but they do not currently expose the `backend` argument to users. This matters because some platforms, especially MPS, may not support the default backend well and may need an alternative backend such as `aot_eager`. I chose this issue because it connects to my interest in PyTorch and ML systems while still being scoped enough for a focused open-source contribution.

---

## Why I Chose This Issue

I chose this issue because it connects to my interest in PyTorch, machine learning systems, and optimizer behavior. After working on PyTorch Ignite for my first open-source contribution, I wanted my second contribution to stay within the PyTorch ecosystem while focusing on a practical issue that affects how ML tooling works across different hardware backends.

This issue is meaningful because torchao’s low-bit optimizers rely on `torch.compile`, but the optimizer path currently does not expose the `backend` argument. This limits flexibility for users on platforms such as MPS, where the default backend may not be supported. I also liked that the issue is not just a documentation change, but still seems manageable because the main compile call appears to be centralized in `torchao/optim/adam.py`, and the fix can be tested by verifying that a custom backend argument is forwarded correctly.

---

## Understanding the Issue

### Problem Description

torchao’s low-bit Adam optimizers internally call `torch.compile`, but they do not expose the `backend` argument to users. This means users cannot choose a different compile backend such as `aot_eager`.

This matters for platforms such as MPS, where the default `torch.compile` backend may not be supported. The issue is not that the optimizer logic itself is broken, but that the optimizer API is missing a way to pass a backend choice into the internal compile call.

### Expected Behavior

Users should be able to create a low-bit optimizer with an optional compile backend, for example:

```python
Adam8bit(model.parameters(), compile_backend="aot_eager")
```

When this option is provided, the optimizer should pass it into the internal `torch.compile` call as `backend="aot_eager"`.

If the option is not provided, the current default behavior should stay unchanged.

### Current Behavior

Passing `compile_backend="aot_eager"` currently raises a `TypeError` because the optimizer constructors do not accept this argument.

The internal compile call currently looks like:

```python
torch.compile(single_param_adam, fullgraph=True, dynamic=False)
```

There is no `backend` argument being passed.

### Affected Components

The main affected file is:

- `torchao/optim/adam.py`

The relevant code path is `_AdamBase.step()`, which is shared by the low-bit Adam optimizer classes such as:

- `Adam8bit`
- `AdamW8bit`
- `Adam4bit`
- `AdamW4bit`
- `AdamFp8`
- `AdamWFp8`

The tests were added in:

- `test/test_low_bit_optim.py`

Documentation was updated in:

- `torchao/optim/README.md`

---

## Reproduction Process

### Environment Setup

I set up the project locally on my Intel Mac. My first attempt used a standard Python virtual environment, but that installed `torch 2.2.2`, which was too old for the current `torchao` main branch. This caused import errors for newer PyTorch internals such as `torch._dynamo.utils.warn_once` and `torch.distributed._composable.fsdp`.

To fix this, I installed Miniforge and created a Conda environment using conda-forge:

```bash
conda create -n torchao-955-test -c conda-forge python=3.11 "pytorch>=2.4" numpy pytest parameterized expecttest packaging ruff -y
conda activate torchao-955-test
USE_CPP=0 python -m pip install -e . --no-build-isolation
```

This setup installed Python 3.11.15 and PyTorch 2.12.1, which resolved the import errors.

I verified the setup by importing the torchao optimizers and running the low-bit optimizer smoke tests:

```bash
python -m pytest test/test_low_bit_optim.py -k smoke -q
```

Result:

```text
12 passed, 34 deselected, 17 warnings in 135.95s
```

The warnings were PyTorch deprecation/cache-related warnings and did not block the test run.

### Steps to Reproduce

1. Activate the working Conda environment:

```bash
conda activate torchao-955-test
```

2. From the local `ao` repository, try constructing `Adam8bit` and `AdamW8bit` with a `compile_backend` argument:

```python
from torchao.optim import Adam8bit, AdamW8bit
import torch.nn as nn

for cls in (Adam8bit, AdamW8bit):
    cls(nn.Linear(4, 4).parameters(), compile_backend="aot_eager")
```

3. Observe that both constructors raise a `TypeError` because `compile_backend` is not currently accepted.

4. Patch `torch.compile` during a minimal optimizer step to inspect what keyword arguments it receives.

5. Observe that `torch.compile` receives `fullgraph=True` and `dynamic=False`, but no `backend` argument.

### Reproduction Evidence

- **Working branch:** https://github.com/Abhisri436/ao/tree/fix-issue-955-compile-backend
- **Source location:** `torchao/optim/adam.py`, inside `_AdamBase.step()`

Observed constructor error:

```text
Adam8bit.__init__() got an unexpected keyword argument 'compile_backend'
AdamW8bit.__init__() got an unexpected keyword argument 'compile_backend'
```

Observed internal compile kwargs:

```text
torch.compile called with kwargs: {'fullgraph': True, 'dynamic': False}
backend in kwargs: False
```

### My Findings

I reproduced the issue in two ways. First, constructing `Adam8bit` or `AdamW8bit` with `compile_backend="aot_eager"` raises a `TypeError` because the optimizers do not currently expose that parameter. Second, I patched `torch.compile` and ran a minimal optimizer step on CPU. The patched call showed that `torch.compile` receives `fullgraph=True` and `dynamic=False`, but no `backend` argument.

This confirms that users are currently locked into the default `torch.compile` backend. The issue can be reproduced without MPS hardware because the missing API surface is visible on CPU.

---

## Solution Approach

### Analysis

The root cause is that the low-bit Adam optimizer path hard-codes the internal `torch.compile` call without exposing a user-facing backend option. The relevant call is in `torchao/optim/adam.py` inside `_AdamBase.step()`:

```python
torch.compile(single_param_adam, fullgraph=True, dynamic=False)
```

All of the public low-bit Adam optimizers use this shared `_AdamBase.step()` path, but their constructors have explicit signatures and do not accept `compile_backend`. Because there is no `**kwargs` passthrough, adding the parameter only to `_AdamBase` would not be enough. The public optimizer constructors also needed to accept and forward the argument.

### Investigative Depth

During investigation, I found that `_AdamBase` already stores optimizer configuration values such as `block_size`, `bf16_stochastic_round`, and `is_adamw` in `__init__`, then uses them later in `step()`. This was an analogous pattern for adding `compile_backend`: the value could be accepted by the constructor, stored on the optimizer, and used later when calling `torch.compile`.

I also considered edge cases for preserving backward compatibility. If `compile_backend` is not provided, the optimizer should not pass a `backend` argument to `torch.compile`, so existing users continue getting the same default behavior. If a backend is provided, the value should be forwarded exactly to `torch.compile`.

### Implemented Solution

I added an optional `compile_backend` parameter to the low-bit Adam optimizers. The value is stored on `_AdamBase` and passed into `torch.compile` only when it is provided.

When `compile_backend` is not set, the optimizer preserves the current behavior exactly by calling `torch.compile` without a `backend` argument.

### Implementation Plan

Using UMPIRE framework:

**Understand:** torchao’s low-bit Adam optimizers call `torch.compile` internally, but users cannot choose the backend. This prevents users on platforms such as MPS from passing a backend like `aot_eager`.

**Match:** The optimizer already stores configuration values such as `block_size`, `bf16_stochastic_round`, and `is_adamw` in `_AdamBase.__init__`, then uses them later in `step()`. I followed this same pattern for `compile_backend`.

**Plan:**

1. Modify `torchao/optim/adam.py`.
2. Add `compile_backend=None` to `_AdamBase.__init__`.
3. Store it as `self.compile_backend`.
4. Add `compile_backend=None` to the public optimizer constructors:
   - `Adam8bit`
   - `AdamW8bit`
   - `Adam4bit`
   - `AdamW4bit`
   - `AdamFp8`
   - `AdamWFp8`
5. Forward `compile_backend=compile_backend` to `super().__init__`.
6. Update `_AdamBase.step()` so it passes `backend=self.compile_backend` to `torch.compile` only when `compile_backend` is not `None`.
7. Add tests in `test/test_low_bit_optim.py`.
8. Update `torchao/optim/README.md` to document the new argument.

**Implement:** Work happened on this branch:

https://github.com/Abhisri436/ao/tree/fix-issue-955-compile-backend

**Review:** I kept the diff focused and followed the project’s contribution expectations: added tests for code changes, kept formatting/linting clean with Ruff, and updated documentation because this is a public API change.

**Evaluate:** I verified the fix by adding tests that check:

- `compile_backend="aot_eager"` is accepted by the optimizer constructors.
- The backend argument is forwarded to `torch.compile`.
- The default behavior is preserved when `compile_backend` is not set.
- Existing low-bit optimizer tests still pass.

MPS hardware is not required for these tests because the missing behavior is an API-forwarding issue that can be verified on CPU.

---

## Testing Strategy

### Unit Tests

For Phase III, I added targeted tests in `test/test_low_bit_optim.py` for the new `compile_backend` behavior.

The tests verify that:

- [x] Low-bit Adam optimizer constructors accept `compile_backend="aot_eager"`.
- [x] `compile_backend` is forwarded to `torch.compile` as the `backend` argument.
- [x] When `compile_backend` is not provided, no `backend` argument is passed and the current default behavior is preserved.

### Integration / Regression Tests

I also ran the existing low-bit optimizer tests to confirm that the change did not break current behavior.

### Manual Testing

I ran the following validation commands locally:

```bash
ruff check --isolated --select F821,F823,W191
ruff check torchao/optim/adam.py test/test_low_bit_optim.py
ruff format --check torchao/optim/adam.py test/test_low_bit_optim.py
python -m pytest test/test_low_bit_optim.py -q
```

Results:

```text
ruff check --isolated --select F821,F823,W191 → All checks passed
ruff check torchao/optim/adam.py test/test_low_bit_optim.py → All checks passed
ruff format --check torchao/optim/adam.py test/test_low_bit_optim.py → 2 files already formatted
python -m pytest test/test_low_bit_optim.py -q → 44 passed, 10 skipped, 17 warnings
```

I also ran the targeted compile backend and smoke tests:

```bash
python -m pytest test/test_low_bit_optim.py -k "compile_backend or smoke" -q
```

Result:

```text
20 passed, 34 deselected, 17 warnings
```

The warnings were existing PyTorch deprecation/cache warnings and did not block the test runs.

### Final Pre-Submission Checks

Before opening the PR, I rebased my branch on the latest `upstream/main` and reran the relevant checks.

Final validation results:

```text
ruff check --isolated --select F821,F823,W191 → All checks passed
ruff check torchao/optim/adam.py test/test_low_bit_optim.py → All checks passed
ruff format --check torchao/optim/adam.py test/test_low_bit_optim.py → 2 files already formatted
python -m pytest test/test_low_bit_optim.py -k "compile_backend or smoke" -q → 20 passed, 34 deselected
python -m pytest test/test_low_bit_optim.py -q → 44 passed, 10 skipped
```

---

## Implementation Notes

### Week 6 Progress

For Week 6, I selected PyTorch AO issue #955 for my second open-source contribution. I forked the repository, created the working branch `fix-issue-955-compile-backend`, and posted a comment on the issue stating my intent to work on it.

I also completed the local setup and reproduction process. The main setup challenge was that my first Python virtual environment installed `torch 2.2.2`, which was too old for the current `torchao` main branch on my Intel Mac. I resolved this by using Miniforge and creating a Conda environment with a newer PyTorch version from conda-forge.

### Phase III Progress

During Phase III, I implemented the fix for PyTorch AO issue #955 in small, testable increments.

I added an optional `compile_backend` argument to the low-bit Adam optimizer path in `torchao/optim/adam.py`. The value is stored on `_AdamBase` and forwarded to `torch.compile` as the `backend` argument only when provided. When `compile_backend` is not set, the optimizer preserves the existing default behavior by calling `torch.compile` without a `backend` argument.

I also added targeted tests in `test/test_low_bit_optim.py` to verify constructor support, backend forwarding, and preserved default behavior. Finally, I updated `torchao/optim/README.md` with a short usage note documenting the new argument.

### Phase IV Progress

During Phase IV, I completed final pre-submission checks before opening the pull request. I rebased my branch on the latest `upstream/main`, and because the rebase changed my commit hashes, I force-pushed safely using `--force-with-lease`.

I opened PR #4686 against `pytorch/ao:main` from my fork branch `fix-issue-955-compile-backend`. GitHub automatically requested reviews from the relevant code owners. The Meta CLA check was initially pending, but I signed the CLA and the check was approved.

I also posted the required Phase IV completion message in Slack.

Current status: PR submitted and awaiting maintainer review.

### Code Changes

- **Files modified:**
  - `torchao/optim/adam.py`
  - `test/test_low_bit_optim.py`
  - `torchao/optim/README.md`

- **Key commits:**
  - `667519935` — optim: expose compile backend in low-bit Adam optimizers
  - `b65280372` — test: cover compile backend for low-bit Adam optimizers
  - `2e924c887` — docs: document compile backend for low-bit optimizers

- **Working branch:** https://github.com/Abhisri436/ao/tree/fix-issue-955-compile-backend
- **Pull request:** https://github.com/pytorch/ao/pull/4686

### Approach Decisions

I kept the change additive and backward-compatible. The new `compile_backend` argument defaults to `None`, and the implementation only passes `backend` to `torch.compile` when the user explicitly provides a value. This preserves the existing default behavior for current users.

I also implemented the work in three small increments: source change, tests, and documentation. Each increment was committed separately with a focused commit message.

### Self-Review

Before moving to Phase IV, I reviewed the branch against `upstream/main`. The final diff only includes three relevant files:

- `torchao/optim/adam.py`
- `test/test_low_bit_optim.py`
- `torchao/optim/README.md`

The commit history is clean, the changes are focused, and the implementation is additive and backward-compatible.

---

## Pull Request

**PR Link:** https://github.com/pytorch/ao/pull/4686

**Working Branch:** https://github.com/Abhisri436/ao/tree/fix-issue-955-compile-backend

**PR Description:**

I contributed support for an optional `compile_backend` argument in torchao’s low-bit Adam optimizers. This allows users to pass a backend such as `"aot_eager"` to the internal `torch.compile` call while preserving the existing default behavior when no backend is provided.

**Maintainer Feedback:**

- No formal code review feedback has been received yet.
- GitHub automatically requested reviews from the relevant code owners after the PR was opened.
- The Meta CLA check was initially pending, but I signed the CLA and the check was approved.
- Current status is awaiting maintainer review.

**Status:** PR submitted and awaiting review

---

## Learnings & Reflections

### Technical Skills Gained

I learned more about how torchao’s low-bit optimizers use `torch.compile` internally and how optimizer constructor arguments flow into shared base-class behavior. I also learned how environment setup can depend heavily on hardware architecture, since my Intel Mac could not install a new enough PyTorch version through a standard pip-based virtual environment.

During Phase III, I also gained practice making a small API change in a larger ML systems codebase. I learned how to preserve backward compatibility by only passing optional arguments when they are explicitly provided, and how to test API-forwarding behavior by patching `torch.compile` instead of requiring a real MPS device.

During Phase IV, I learned how to prepare an open-source pull request for review by rebasing on the latest upstream branch, force-pushing safely with `--force-with-lease`, writing a clear PR description, handling the CLA check, and checking that code owner reviews were requested.

### Challenges Overcome

The biggest challenge in Phase II was environment setup. My first setup failed because the available PyTorch version for my Intel Mac was too old for the current torchao main branch. I resolved this by installing Miniforge, creating a Conda environment, and verifying that the newer PyTorch version could import the required torchao optimizer components.

During Phase III, the main challenge was keeping the change small while still covering all public optimizer constructors. The actual compile call lives in `_AdamBase.step()`, but the public optimizer classes have explicit constructor signatures, so I needed to add and forward `compile_backend` through all relevant low-bit Adam and AdamW classes. I solved this by following the existing pattern used for optimizer configuration values like `block_size`, `bf16_stochastic_round`, and `is_adamw`.

During Phase IV, the main challenge was making sure the branch was ready for public review. The upstream branch had advanced, so I rebased my branch on the latest `upstream/main` and used `--force-with-lease` to safely update my fork after the rebase changed my commit hashes.

### What I'd Do Differently Next Time

Next time, I would check the project’s PyTorch version expectations and my local hardware compatibility earlier, especially for projects in the PyTorch ecosystem that may depend on newer internal APIs.

I would also continue using small, focused commits because it made the implementation easier to review and helped me verify each piece before moving on.

---

## Resources Used

- Issue #955: https://github.com/pytorch/ao/issues/955
- Pull Request #4686: https://github.com/pytorch/ao/pull/4686
- PyTorch AO repository: https://github.com/pytorch/ao
- PyTorch AO contributing guide: https://github.com/pytorch/ao/blob/main/CONTRIBUTING.md
- PyTorch AO optimizer source: `torchao/optim/adam.py`
- PyTorch AO optimizer tests: `test/test_low_bit_optim.py`
