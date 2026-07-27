# CTRE Simulator

Minimal experimental simulator for testing the **Causal Transition & Representation Evolution (CTRE)** hypothesis.

## Core Question

Can an adaptive system detect when parameter updates are no longer sufficient and autonomously revise its representation to improve transfer performance?

The benchmark compares:

- **State Adaptation**: optimizing parameters inside a fixed representation
- **Representation Adaptation**: modifying the coordinate system when structural failure is detected

## v0 Experiment

The first experiment uses a minimal scalar latent environment:

\[
y = a \cdot v \cdot (1+z)
\]

where:

- `v` is observable
- `z` is hidden from the agent
- prediction failures accumulate as representation pressure

The agent must determine whether errors can be solved by parameter updates or require a structural transformation.

## Agents

### Agent B — State Only

- fixed representation
- optimal parameter adaptation
- no structural mutation

### Agent E — REE

Representation Evolution Engine:

- detects persistent residual failure
- generates anonymous structural hypotheses
- evaluates candidates using complexity penalties and validation performance
- commits only beneficial transformations

## Design Principles

The benchmark enforces:

- no access to hidden variables
- no semantic labels
- no LLM-generated hypotheses
- compute parity between agents
- null hypothesis ("do nothing") included
- holdout validation

## Repository Status

v0 implementation.

The goal is not to prove the full CTRE framework, but to test whether representation revision provides measurable advantages over optimal state adaptation in a controlled environment.

## Structure

ctre_simulator/
├── env.py
├── state_agent.py
├── ree_agent.py
├── hypothesis.py
├── evaluator.py
└── run.py


## Evaluation

Primary metrics:

- transfer performance
- intervention performance
- representation regret
- schema stability
- adaptation efficiency

Success requires Agent E to outperform Agent B under identical interaction and compute budgets.
