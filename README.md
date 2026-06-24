# Contribution #1757: More EpochMetric's compute_fn output types

**Contribution Number:** 1  
**Student:** Potri Abhisri Barama  
**Issue:** [More EpochMetric's compute_fn output types #1757](https://github.com/pytorch/ignite/issues/1757)  
**Status:** Phase I Complete

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

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week [X] Progress

[What you built this week, challenges faced, decisions made]

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
