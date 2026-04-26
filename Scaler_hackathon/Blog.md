#  When Systems Dont Talk: What a Broken AI Workflow Taught Me About Power Grids

Most people think power grids fail because something breaks.

Thats not true.

They fail because **things dont understand each other**.

---

##  It Started With a Website

I was building a project using AI.

Everything was working  until I switched models.

One model had built the structure.
The next one didnt understand it.

Suddenly:

* logic broke
* features stopped working
* the system became unstable

Not because the models were bad.

Because they werent coordinated.

---

##  Thats When It Clicked

A few weeks later, I read about a national power outage.

Different systems were running.
Each one was working.

But they werent aligned.

And the entire grid collapsed.

---

##  Same Pattern, Different System

This wasnt about AI anymore.

It was about **systems**.

> Systems dont fail because one part is wrong.
> They fail because parts dont understand each other.

---

##  What Actually Happens in a Power Grid

Behind a simple switch, theres a constant balancing act:

* supply vs demand
* production vs consumption

Even a small mismatch can cause:

* instability
* cascading failures
* full blackout

And this happens continuously.

---

##  So I Built a Simulation

Instead of reading about systems, I wanted to experience one.

I built a grid simulation where:

* demand changes over time
* supply must be adjusted in real-time
* decisions affect future stability

And instead of one controller

I split it into three.

---

##  Three Agents, One System

### Planner

Plans ahead.

### Dispatch

Balances in real time.

### Market

Optimises cost.

Each one makes logical decisions.

Individually, theyre fine.

Together?

Things get interesting.

---

##  Adding AI Into the System

To make this more than a simulation, I trained a model.

* Base model: TinyLlama
* Fine-tuned using LoRA
* Trained on simulation data

The model learns:

* state  action  reward

Simple idea.

Complex behaviour.

---

##  The Results

At first, it looked promising.

Performance improved by nearly **70%**.

Everything seemed stable.

---

##  Then It Fell Apart

Step by step, things started drifting.

* small imbalances appeared
* decisions became inconsistent
* instability grew

Eventually:

> the system collapsed

---

##  The Most Important Insight

This wasnt a failure.

It was the result.

> Short-term optimisation is not the same as long-term stability.

The model learned:

* how to act well now

But not:

* how those actions affect the future

---

##  What Research Says

Modern research shows that AI is already used across:

* load forecasting
* fault detection
* energy optimisation

But theres a problem.

Most systems optimise **in isolation**.

And that creates:

* conflicting decisions
* poor coordination
* unstable outcomes

 shows that energy management and fault detection are often treated separately, missing system-level interaction.

---

##  Why This Is Hard

Real-world grids deal with:

* uncertainty
* incomplete information
* real-time constraints
* multiple decision-makers

Even advanced models struggle:

* Deep Learning  accurate but slow
* Reinforcement Learning  adaptive but unstable
* Optimisation  stable but rigid

---

##  The Real Problem

Not intelligence.

Coordination.

---

##  Why AI Alone Isnt Enough

AI can:

* predict
* optimise
* automate

But it doesnt guarantee:

* alignment
* consistency
* system-wide stability

As research highlights:

> AI improves efficiency and reliability, but depends heavily on data quality, coordination, and system integration 

---

##  What This Project Actually Shows

This project isnt just about energy.

Its about:

* multi-agent systems
* decision-making under uncertainty
* coordination failure

And most importantly:

> how complex systems behave over time

---

##  Methodology (Short Version)

* simulation-based environment
* multi-agent architecture
* reward-driven learning
* LoRA fine-tuning
* evaluation via stability and reward

---

##  Why This Matters

This applies to:

* AI agents
* financial systems
* distributed computing
* real-world infrastructure

---

##  Final Thought

We spend a lot of time making systems smarter.

But maybe the real problem is this:

> they dont understand each other.

---

##  One Line Summary

**Intelligence improves parts.
Coordination determines survival.**
