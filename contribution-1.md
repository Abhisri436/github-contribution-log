# Contribution #1757: More EpochMetric's compute_fn output types

**Contribution Number:** 1  
**Student:** Potri Abhisri Barama  
**Issue:** [More EpochMetric's compute_fn output types #1757](https://github.com/pytorch/ignite/issues/1757)  
**My Branch:** [https://github.com/Abhisri436/ignite/tree/fix-issue-1757](https://github.com/Abhisri436/ignite/tree/fix-issue-1757)

* **Key commits:**

  * [`77759941`](https://github.com/Abhisri436/ignite/commit/77759941) — Add EpochMetric output type tests
  * [`00e92338`](https://github.com/Abhisri436/ignite/commit/00e92338) — Support flexible EpochMetric compute outputs
  * [`fcaef4cf`](https://github.com/Abhisri436/ignite/commit/fcaef4cf) — Add distributed EpochMetric output tests

**Status:** Phase IV Complete — PR submitted, awaiting review

---

## Why I Chose This Issue

I chose this issue because it connects to my interest in machine learning metrics and model evaluation. `EpochMetric` is used when a metric needs to be computed over an entire epoch, and improving the types of outputs that `compute_fn` can return would make it more flexible for users who are building custom metrics. Since I have worked with Python, PyTorch-style ML workflows, and evaluation metrics in my previous projects, this issue feels connected to my background while still giving me room to learn more about how metrics are implemented inside a real open-source library.

I also like that the issue has a clear goal: support more output types such as tensors, tuples/lists, or mappings of tensors, and improve the error message when an unsupported type is returned. Through this contribution, I hope to get more comfortable reading a larger codebase, writing tests for edge cases, and understanding how an ML library designs clean APIs for users. This feels like a good balance between being realistic for a first contribution and still being technical enough to help me grow.

---

## Understanding the Issue

### Problem Description

EpochMetric is currently documented and typed as if compute_fn should only return a scalar value. However, issue #1757 asks for EpochMetric to support more flexible output types from compute_fn, such as tensors, tuples/lists of tensors, or mappings/dictionaries of tensors. This is useful because some custom epoch-level metrics may need to return multiple values instead of a single scalar.

### Expected Behavior

EpochMetric should clearly support valid compute_fn output types such as scalar values, tensors, tuple/list outputs, and mapping/dictionary outputs when appropriate. The supported behavior should be reflected in the type hints, documentation, tests, and distributed handling. Unsupported output types should either be rejected with a clear error message or handled according to the maintainers’ preferred API design.

### Current Behavior

In my local single-process reproduction, EpochMetric returned whatever compute_fn returned. It accepted scalar floats, scalar tensors, vector tensors, tuple/list outputs, dictionary outputs, and even an unsupported string without raising an error.

This means richer output types are not necessarily failing in single-process mode, but they are also not officially typed, documented, validated, or tested. The distributed path still appears scalar-focused because compute() initializes the result as 0.0 and casts the broadcasted result to float.

### Affected Components

The main affected component is:

- ignite/metrics/epoch_metric.py

The main test file involved is:

- tests/ignite/metrics/test_epoch_metric.py

Related files/patterns may include:

- ignite/metrics/metric.py
- ignite/metrics/precision_recall_curve.py
- ignite/metrics/roc_auc.py
- ignite/metrics/accumulation.py

---

## Reproduction Process

### Environment Setup

I set up the PyTorch Ignite repository locally in VS Code using my fork of the project. I cloned my fork, added the original pytorch/ignite repository as the upstream remote, and created a working branch named fix-issue-1757.

During setup, I initially ran into dependency issues. The full development requirements failed because llvmlite could not build while I was using Python 3.10.13. I also found that conda was not installed on my machine, so I used pyenv instead. I installed Python 3.11.11 with pyenv, created a local .venv inside the Ignite repo, installed Ignite in editable mode, and installed the minimal dependencies needed to run the EpochMetric tests.

Additional setup fixes included installing numpy<2, pytest-xdist, and pytest-timeout so the relevant tests could run cleanly.

Branch Link:
https://github.com/Abhisri436/ignite/tree/fix-issue-1757 

### Steps to Reproduce

1. Clone the Ignite fork and open it locally.
git clone https://github.com/Abhisri436/ignite.git
cd ignite
git remote add upstream https://github.com/pytorch/ignite.git

2. Create and switch to the working branch.
git checkout -b fix-issue-1757

3. Activate the local Python environment.
source .venv/bin/activate
python --version
which python

4. Confirm the branch and working tree.
git status
git branch --show-current

5. Run the existing EpochMetric tests.
pytest tests/ignite/metrics/test_epoch_metric.py -vv

6. Run a local reproduction script that tests different compute_fn return types.
python repro_issue_1757.py
python repro_issue_1757.py

The reproduction tested these return types:

- scalar float
- scalar tensor
- vector tensor
- tuple of tensors
- list of tensors
- dictionary of tensors
- unsupported string output

### Reproduction Evidence

- **Commit showing reproduction:** N/A for now. The reproduction script and notes were kept local and excluded from the Ignite PR so they do not affect the source code.
- **Branch Link:** https://github.com/Abhisri436/ignite/tree/fix-issue-1757 
- **Screenshots/logs:** Local terminal output from running repro_issue_1757.py twice and pytest tests/ignite/metrics/test_epoch_metric.py -vv.
- **My findings:** In single-process mode, EpochMetric currently returns whatever compute_fn returns. Scalar floats, tensors, tuple/list outputs, dictionary outputs, and even a string all returned without error. This suggests the issue is not that richer outputs always fail locally, but that the implementation is still scalar-focused in its type hints, documentation, tests, validation behavior, and distributed broadcast logic.

The existing test command passed:

7 passed, 13 skipped

The skipped tests were expected because they relate to CUDA, MPS, XLA, Horovod, or NCCL environments that are not available on my CPU-only Mac setup.

---

## Solution Approach

### Analysis

The root cause appears to be that EpochMetric was originally designed around scalar metric outputs. In ignite/metrics/epoch_metric.py, the docstring says that compute_fn returns a scalar, the type hint for compute_fn is Callable[[torch.Tensor, torch.Tensor], float], _result is typed as float | None, and compute() is typed as returning float.

The distributed path also appears scalar-focused. Inside compute(), self._result is initialized as 0.0, only rank 0 runs compute_fn, and then the result is broadcast and cast to float. This design does not clearly support tuple/list or mapping outputs in distributed mode.

The current tests in tests/ignite/metrics/test_epoch_metric.py mainly check scalar outputs such as 0.0 or .item(). There are no dedicated tests for tensor outputs, tuple/list outputs, mapping outputs, or unsupported output types.

### Proposed Solution

I plan to update EpochMetric so its supported compute_fn output types are explicit, tested, and documented. At a high level, the fix should broaden the type hints and documentation, add validation for supported output types, and update the distributed broadcast logic so non-scalar supported outputs can be handled consistently.

The proposed supported output types are:

- scalar numbers
- torch.Tensor
- tuple/list of tensors
- mapping/dictionary outputs with string keys and tensor values

For unsupported output types, such as a plain string, the proposed behavior is to raise a clear TypeError. I will keep this as proposed behavior and confirm during implementation/review, since it changes the current silent pass-through behavior.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** EpochMetric should support more flexible outputs from compute_fn, not just scalar values. My reproduction showed that single-process mode already passes through many output types, but the class is still typed, documented, and tested as scalar-only. The distributed path also appears to assume scalar results.

**Match:** 
Similar patterns in the codebase include:
- ignite/metrics/accumulation.py, which has a clear TypeError pattern for unsupported output types.
- ignite/metrics/precision_recall_curve.py and ignite/metrics/roc_auc.py, which use idist.broadcast(..., safe_mode=True) for tensor-like distributed outputs.
- ignite/metrics/metric.py, which already includes logic related to Mapping results.
- tests/ignite/metrics/test_metric.py and tests/ignite/metrics/test_accumulation.py, which include examples of validation and TypeError tests.

**Plan:** 
1. Modify ignite/metrics/epoch_metric.py to broaden the expected return type of compute_fn beyond float.
2. Update the EpochMetric docstring so it no longer says compute_fn only returns a scalar.
3. Add a helper or internal validation step for supported compute_fn output types.
4. Update compute() so it no longer initializes _result as only a scalar 0.0 when non-scalar outputs are supported.
5. Update the distributed broadcast logic so supported tensor, tuple/list, and mapping outputs can be broadcast correctly if feasible.
6. Add tests in tests/ignite/metrics/test_epoch_metric.py for:
- scalar float output
- scalar tensor output
- vector tensor output
- tuple/list of tensors
- dictionary/mapping of tensors
- unsupported output type such as string
7. Run the relevant tests and formatting/lint checks before submitting a PR.

**Implement:** Implementation will happen in Phase III on this branch:

https://github.com/Abhisri436/ignite/tree/fix-issue-1757

**Review:** Before opening a PR, I will review the project’s contribution guidelines and check that:

- the new tests pass
- the relevant existing tests still pass
- the code follows Ignite’s style
- formatting/linting checks pass, such as ruff or pre-commit if available
- the working tree is clean
- the PR clearly references issue #1757
- the PR description explains the behavior change and test coverage

**Evaluate:** 
I will verify the fix by running:

pytest tests/ignite/metrics/test_epoch_metric.py -vv

If the full local environment supports it, I will also run formatting/lint checks such as:

pre-commit run -a

or the equivalent ruff commands used by the project.

The fix will be considered successful if:

- existing scalar behavior still works
- tensor outputs are supported and tested
- tuple/list tensor outputs are supported and tested
- dictionary/mapping tensor outputs are supported and tested
- unsupported output types produce a clear error if that behavior is accepted
- existing EpochMetric tests continue to pass

---

## Testing Strategy

### Unit Tests

* [x] Test case 1: Added a test for scalar `torch.Tensor` output from `EpochMetric.compute_fn`.
* [x] Test case 2: Added a test for vector `torch.Tensor` output from `EpochMetric.compute_fn`.
* [x] Test case 3: Added tests for tuple and list outputs containing tensors.
* [x] Test case 4: Added a test for mapping/dictionary outputs with string keys and tensor values.
* [x] Test case 5: Added a test that unsupported output types, such as `str`, raise a clear `TypeError`.
* [x] Test case 6: Added a test that nested containers containing unsupported values raise `TypeError`.
* [x] Test case 7: Added a test that mappings with non-string keys raise `TypeError`.

### Integration Tests

* [x] Ran the existing `EpochMetric` distributed integration test in `tests/ignite/metrics/test_epoch_metric.py`.
* [x] Verified the new distributed behavior under a real `WORLD_SIZE=2` configuration using the `gloo` backend on CPU.
* [x] Verified that non-scalar tensor outputs are broadcast successfully and are identical across ranks.
* [x] Verified that tuple/list/mapping outputs raise `NotImplementedError` consistently on all ranks instead of entering mismatched collective calls.
* [x] Verified that unsupported output types such as `str` raise `TypeError` consistently on all ranks.
* [ ] Multi-rank behavior was not tested with GPU/NCCL or with more than two ranks.

### Manual Testing

I validated the implementation locally by running the targeted `EpochMetric` test file, real multi-rank distributed tests, and style checks.

Commands run:

```bash
pytest tests/ignite/metrics/test_epoch_metric.py -vv

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES="" pytest --dist=each --tx "2*popen//python=python" \
  tests/ignite/metrics/test_epoch_metric.py -m distributed \
  -k "test_distrib_output and gloo_cpu" -q

ruff check ignite/metrics/epoch_metric.py tests/ignite/metrics/test_epoch_metric.py
ruff format --check ignite/metrics/epoch_metric.py tests/ignite/metrics/test_epoch_metric.py
```

Results:

```text
Normal targeted tests: 17 passed, 57 skipped
WORLD_SIZE=2 distributed tests: 10 passed
Ruff check: All checks passed!
Ruff format check: 2 files already formatted
```

I also reviewed the final diff using:

```bash
git status
git log --oneline upstream/master..HEAD
git diff --stat upstream/master...HEAD
git diff --name-only upstream/master...HEAD
git diff --check upstream/master...HEAD
```

The final diff only modifies files relevant to issue #1757:

```text
ignite/metrics/epoch_metric.py
tests/ignite/metrics/test_epoch_metric.py
```

---

## Implementation Notes

### Phase III Progress

For Phase III, I implemented support for more flexible `EpochMetric.compute_fn` output types. Previously, `EpochMetric` was typed and documented as returning a scalar, and unsupported outputs such as strings were not clearly rejected in the local single-process path. My implementation expands the supported output types and adds validation so the behavior is clearer and more reliable.

### What I Built

I updated `EpochMetric` so `compute_fn` can support:

* `int` and `float` scalar outputs
* `torch.Tensor` outputs, including scalar tensors and vector tensors
* tuple/list outputs containing supported values
* mapping/dictionary outputs with string keys and supported values

I also added recursive output validation through a helper method so unsupported output types raise a clear `TypeError`.

For distributed mode, I used a conservative approach. Scalar and tensor results can still be broadcast. Tuple/list/mapping outputs are supported in non-distributed mode, but in `world_size > 1`, they raise a clear `NotImplementedError` instead of attempting unverified recursive distributed container broadcasting. To avoid possible distributed hangs, I added a status-code decision before broadcasting so all ranks follow the same path.

I later added permanent multi-rank tests and verified the distributed behavior with `WORLD_SIZE=2` using the `gloo` backend on CPU. The tests confirm that non-scalar tensor outputs broadcast consistently across ranks, while container outputs and unsupported types fail symmetrically without leaving ranks in mismatched collective calls.

### Challenges Faced

One challenge was that some flexible outputs, such as tensors, tuples, lists, and dictionaries, already appeared to work in the local single-process path because the result was passed through without validation. However, the code, type hints, documentation, and distributed path were still scalar-focused. This made the issue less obvious than a simple failing test.

Another challenge was the distributed implementation. The local test environment did not fully exercise `world_size > 1`, so I had to be careful not to introduce untested distributed behavior. I first considered supporting recursive container broadcasting, but decided that was riskier. I solved this by adding a conservative status-code guard so all ranks make the same decision before broadcasting or raising an error.

A third challenge was discovering an overlapping open PR for the same issue after I had already selected issue #1757 and started my implementation work. Instead of treating this as unrelated, I reviewed the overlap and used it to think more carefully about how my contribution could still be useful. My tests check actual output values, not only output types or shapes, which makes them useful for validating the expected behavior.

Tools that helped:

* `pytest` for running the targeted test file
* `ruff` for linting and formatting checks
* `git diff`, `git status`, and `git log` for self-review
* Claude Code for local implementation assistance and diff review
* ChatGPT for planning, review, and open-source contribution strategy

### Code Changes

* **Files modified:**

  * `ignite/metrics/epoch_metric.py`
  * `tests/ignite/metrics/test_epoch_metric.py`

* **Key commits:**

  * [`77759941`](https://github.com/Abhisri436/ignite/commit/77759941) — Add EpochMetric output type tests
  * [`00e92338`](https://github.com/Abhisri436/ignite/commit/00e92338) — Support flexible EpochMetric compute outputs
  * [`fcaef4cf`](https://github.com/Abhisri436/ignite/commit/fcaef4cf) — Add distributed EpochMetric output tests
 
* **Approach decisions:**

  * I added tests before changing the implementation so the expected behavior was clearly defined.
  * I validated scalar tensor, vector tensor, tuple/list, mapping, and unsupported output behavior.
  * I initially used a conservative distributed approach because full multi-rank behavior had not yet been tested locally. I later verified the implementation with `WORLD_SIZE=2` using the `gloo` backend and added permanent distributed regression tests.
  * I avoided unrelated files and kept the final diff focused on the metric implementation and its tests.

* **Branch link:**

  * https://github.com/Abhisri436/ignite/tree/fix-issue-1757

---

## Pull Request

**PR Link:** https://github.com/pytorch/ignite/pull/3802

**Working Branch:** https://github.com/Abhisri436/ignite/tree/fix-issue-1757

**PR Description:**

This PR addresses issue #1757 by extending `EpochMetric.compute_fn` output support beyond scalar values. It adds validation for supported output types including scalars, tensors, sequences, and mappings, and unsupported output types now raise a clear `TypeError`.

The implementation also updates the `EpochMetric` docstring and adds tests covering scalar tensor, vector tensor, tuple/list tensor outputs, mapping outputs, and unsupported output behavior. For distributed mode, this PR uses a conservative approach: scalar and tensor outputs can be broadcast, while tuple/list/mapping outputs raise a clear `NotImplementedError` in `world_size > 1` instead of attempting unverified recursive distributed container broadcasting. 

The PR also includes multi-rank distributed tests that verify non-scalar tensor outputs are consistent across ranks and that unsupported container/type outputs fail symmetrically.

**Maintainer Feedback:**

* I opened PR #3802 for issue #1757 after completing my local implementation and final checks.
* I had already selected issue #1757 and started my implementation before discovering overlapping PR #3789.
* I mentioned the overlapping PR in my PR description and noted that I am happy to close, adjust, or consolidate my PR if maintainers prefer.
* Current PR status: no direct review feedback received yet.

**Status:** Awaiting review

---

## Learnings & Reflections

### Technical Skills Gained

Through this phase, I learned more about how metrics are implemented in a real machine learning library. I practiced reading existing source code, identifying where type assumptions were built into the implementation, and adding tests that check actual computed values.

I also learned more about distributed code safety. Even a small change to a distributed broadcast path can cause problems if different ranks execute different collective calls. This helped me understand why distributed behavior needs careful reasoning and CI coverage.

### Challenges Overcome

The hardest part was deciding how much distributed support to implement. I wanted to solve the issue, but I also did not want to add risky behavior that I could not test locally. I handled this by keeping the implementation conservative and documenting the limitation clearly.

I also had to handle the open-source coordination challenge of an overlapping PR. Since I had already started the issue before discovering the overlap, I continued documenting my own work while also commenting on the existing PR to avoid unnecessary duplicate effort.

### What I'd Do Differently Next Time

Next time, I would look more closely at the repository's distributed testing setup at the beginning so I could plan around what can and cannot be tested locally.

---

## Resources Used

* PyTorch Ignite issue #1757: https://github.com/pytorch/ignite/issues/1757
* My working branch: https://github.com/Abhisri436/ignite/tree/fix-issue-1757
* Existing `EpochMetric` tests in `tests/ignite/metrics/test_epoch_metric.py`
* Existing distributed broadcast patterns in Ignite metrics
* PyTorch Ignite contribution files and test suite
* My PR #3802: https://github.com/pytorch/ignite/pull/3802
* `pytest`
* `ruff`

---
