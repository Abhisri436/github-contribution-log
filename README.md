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
