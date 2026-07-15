# Contribution #955: [MPS] torchao low-bit-precision optim does not expose `backend` argument to `torch.compile`

**Contribution Number:** 2   
**Student:** Potri Abhisri Barama  
**Fork:** https://github.com/Abhisri436/ao  
**Issue:** https://github.com/pytorch/ao/issues/955  
**Status:** Phase I Complete — Phase II In Progress

---

## Why I Chose This Issue

I chose this issue because it connects to my interest in PyTorch, machine learning systems, and optimizer behavior. After working on PyTorch Ignite for my first open-source contribution, I wanted my second contribution to stay within the PyTorch ecosystem while focusing on a practical issue that affects how ML tooling works across different hardware backends.

This issue is meaningful because torchao’s low-bit optimizers rely on `torch.compile`, but the optimizer path currently does not expose the `backend` argument. This limits flexibility for users on platforms such as MPS, where the default backend may not be supported. I also liked that the issue is not just a documentation change, but still seems manageable because the main compile call appears to be centralized in `torchao/optim/adam.py`, and the fix can likely be tested by verifying that a custom backend argument is forwarded correctly.

---

## Understanding the Issue

### Problem Description

[In your own words, what's broken or missing?]

### Expected Behavior

[What should happen?]

### Current Behavior

[What actually happens?]

### Affected Components

[Which parts of the codebase are involved?]

---

## Reproduction Process

### Environment Setup

[Notes on setting up your local development environment - challenges you faced, how you solved them]

### Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Observed result]

### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork]
- **Screenshots/logs:** [If applicable]
- **My findings:** [What you discovered during reproduction]

---

## Solution Approach

### Analysis

[Your analysis of the root cause - what's causing the issue?]

### Proposed Solution

[High-level description of your fix approach]

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** [Restate the problem]

**Match:** [What similar patterns/solutions exist in the codebase?]

**Plan:** [Step-by-step implementation plan]
1. [Modify file X to do Y]
2. [Add function Z]
3. [Update tests]

**Implement:** [Link to your branch/commits as you work]

**Review:** [Self-review checklist - does it follow the project's contribution guidelines?]

**Evaluate:** [How will you verify it works?]

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
