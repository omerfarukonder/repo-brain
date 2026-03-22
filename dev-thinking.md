# RepoBrain — Developer Thinking & Task Evaluation Model

## Purpose

This document encodes how experienced software engineers **interpret, decompose, and estimate tasks**. It is designed to be used as a knowledge base for an AI agent (e.g., Claude Code) so that it can:

* Understand tasks like a real developer
* Estimate effort realistically
* Identify risks and unknowns
* Avoid hallucinated or overconfident outputs
* Continuously improve via feedback loops

---

# 1. Core Principle

> Developers do NOT estimate effort directly.
> They estimate **structure, uncertainty, and familiarity**.

---

# 2. Task Understanding Pipeline

When a developer receives a task, they internally transform it:

## Step 1 — Scope Extraction

Identify affected systems:

* Frontend
* Backend
* Database
* APIs
* External services

👉 Output:

```
components = [frontend, backend, api]
```

---

## Step 2 — Intent Clarification

Convert vague request into concrete goal.

Example:

Input:

```
"Improve product listing"
```

Internal interpretation:

```
"Add filtering + sorting + performance improvements"
```

---

## Step 3 — Atomic Decomposition

Break task into smallest meaningful units:

Example:

```
- Create UI component
- Add API endpoint
- Modify DB query
- Add validation
- Write tests
```

👉 Rule:

> If a task cannot be estimated, it is not decomposed enough.

---

## Step 4 — Dependency Mapping

```
UI → API → DB
```

👉 Identify:

* Sequential dependencies
* Parallelizable work

---

## Step 5 — Unknown Detection

Classify uncertainty:

| Level  | Example               |
| ------ | --------------------- |
| Low    | Clear UI change       |
| Medium | Partial backend logic |
| High   | "Improve performance" |

---

## Step 6 — Risk Assessment

Evaluate:

* Critical systems touched?
* Performance-sensitive?
* Data integrity risks?

---

# 3. Complexity Scoring Model

## Formula

```
Complexity =
    (Surface Area × Integration Depth)
  + Unknowns
  + Risk
  + Cognitive Load
```

---

## Dimensions

### 1. Surface Area (1–5)

* 1 → Single file
* 3 → Single layer (frontend/backend)
* 5 → Multiple systems

---

### 2. Integration Depth (1–5)

* 1 → Isolated
* 3 → Connected flow
* 5 → Core system

---

### 3. Unknowns (1–5)

* 1 → Fully specified
* 3 → Some ambiguity
* 5 → Highly vague

---

### 4. Risk (1–5)

* 1 → Safe change
* 3 → Moderate impact
* 5 → High production risk

---

### 5. Cognitive Load (1–5)

* 1 → Simple logic
* 3 → Multi-step logic
* 5 → Complex reasoning/domain knowledge

---

## Example

```
Surface Area: 3
Integration Depth: 3
Unknowns: 2
Risk: 2
Cognitive Load: 3

Complexity = 13/25 → Medium
```

---

# 4. ETA Estimation Model

## Core Formula

```
ETA =
    (Sum of Atomic Tasks)
  × Uncertainty Multiplier
  × Context Multiplier
```

---

## Step 1 — Atomic Time Estimation

Example:

```
UI → 2h
API → 3h
Testing → 2h
Debugging → 3h
```

---

## Step 2 — Uncertainty Multiplier

| Clarity | Multiplier |
| ------- | ---------- |
| Clear   | 1.2        |
| Medium  | 1.5        |
| Vague   | 2.0–3.0    |

---

## Step 3 — Context Multiplier

| Context         | Multiplier |
| --------------- | ---------- |
| Familiar system | 1.0        |
| New system      | 1.5        |
| Legacy/complex  | 2.0        |

---

## Step 4 — Final ETA

```
Final ETA ≈ Raw Estimate × 1.5–3
```

---

# 5. Behavioral Heuristics

## Pattern Matching

Developers map tasks to past experiences:

> "This looks like something I did before"

---

## Progressive Refinement

* Initial estimate → rough
* After exploration → refined
* During execution → adjusted

---

## Hidden Buffers

Developers implicitly account for:

* Bugs
* Integration issues
* Interruptions

---

## Emotional Factors

| Factor     | Effect                  |
| ---------- | ----------------------- |
| Confidence | Lower → higher estimate |
| Pressure   | Underestimation         |
| Fatigue    | Overestimation          |

---

# 6. Anti-Hallucination Rules (CRITICAL)

The agent MUST follow these:

## Rule 1 — Do Not Assume Missing Details

If task is vague:

```
Return: "Insufficient information to estimate accurately"
```

---

## Rule 2 — Explicitly List Unknowns

```
Unknowns:
- API structure unclear
- Data schema unknown
```

---

## Rule 3 — Provide Confidence Score

```
Confidence: 65%
```

---

## Rule 4 — Never Output Single-Point ETA

Always provide range:

```
ETA: 1.5 – 3 days
```

---

## Rule 5 — Explain Reasoning

```
Reason:
- Backend complexity medium
- Integration risk present
- Unknown edge cases
```

---

# 7. Output Format (Standardized)

Agent should respond like:

```
Task Breakdown:
- UI implementation → 2h
- API changes → 3h
- Testing → 2h

Complexity Score: 13/25 (Medium)

Unknowns:
- API structure unclear

Risks:
- Might affect existing flows

ETA: 1.5 – 2.5 days

Confidence: 70%
```

---

# 8. Feedback Loop Integration

After task completion:

## Collect

* Actual time
* Issues faced
* Missed unknowns

---

## Update

* Complexity weights
* Multipliers
* Pattern database

---

## Learn

```
If repeated underestimation → increase uncertainty multiplier
```

---

# 9. Advanced Extensions

Future improvements:

* Delay prediction model
* Bug probability scoring
* Developer-specific calibration
* Task similarity embeddings

---

# 10. Final Principle

> A good estimation system does not predict time.
> It models **uncertainty, structure, and experience**.

---

# End of Document
