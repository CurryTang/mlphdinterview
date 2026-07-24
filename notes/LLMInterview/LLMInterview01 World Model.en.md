# LLM Primer 1 · World Model

> Placeholder: This page covers basic concepts. Future expansions may include JEPA, predictive coding, Dreamer, video world models, agent memory, model-based RL, and foundation model planning.

## I. What is a World Model?

A world model can be understood as a model's internal, predictive representation of "how the environment changes." It does not necessarily have to be an explicit physical simulator, nor does it need to generate high-definition video. More generally, it answers:

```text
If the current state is s and I take action a,
what might I see next?
Which variables will change?
Which constraints will remain unchanged?
What outcomes are more likely to occur?
```

In reinforcement learning, a world model is typically expressed as environmental dynamics:

```text
p(s_{t+1}, r_t | s_t, a_t)
```

That is, given the current state and action, predict the next state and reward. In the broader context of LLMs/agents, a world model may not be a separate module, but rather a predictive capability formed by the combination of model parameters, contextual memory, tool feedback, and latent representations.

## II. Why Do We Need a World Model?

Models that only perform pattern matching can perform well on familiar distributions, but when faced with problems requiring planning, counterfactual reasoning, long-range dependencies, or physical/social constraints, single-step imitation is often insufficient.

The value of a world model lies in allowing the model to perform internal simulations before taking action:

```text
Current state
  -> Imagine several possible actions
  -> Predict the consequences of each action
  -> Compare risks and rewards
  -> Choose the better action
```

This corresponds to three capabilities:

- Prediction: Predicting future observations, states, rewards, or outcomes.
- Planning: Searching or evaluating multi-step action sequences before actual execution.
- Generalization: Leveraging environmental structure rather than just memorizing surface patterns from training samples.

## III. What a World Model Is Not

A world model is not simply a "knowledge base." A knowledge base stores facts, such as "Paris is the capital of France"; a world model is more concerned with state transitions, such as "If I push this cup to the edge of the table, it might fall off."

A world model is also not equivalent to a standard generative model. A generative model can produce plausible-looking future segments, but a world model emphasizes predicting variables useful for action: which states will change, which constraints must be maintained, and what consequences an action will bring.

A simple distinction:

| Concept | Focus | Example |
| --- | --- | --- |
| Knowledge | Static facts | What are the parameters of a certain API? |
| Memory | Past experiences | What were the user's preferences last time? |
| World Model | How states evolve | If I call this API, which system states will change? |
| Policy | What action to take | Should I search, write code, or run tests next? |

## IV. World Models in LLMs / Agents

The environment for an LLM agent is not limited to the physical world; it also includes web pages, codebases, shells, databases, tool calls, and multi-turn conversations. Here, the world model is often manifested as:

- Predicting why a test might fail before modifying code.
- Predicting what information an output might provide before calling a tool.
- Understanding which states have already been changed versus which are merely text records in multi-turn tasks.
- Knowing to observe the environment first rather than jumping to conclusions when faced with uncertain information.

The weaker an agent's world model, the more prone it is to these issues:

- Guessing answers based solely on the prompt without checking the environment.
- Failing to understand that tool calls change external states.
- Not running relevant tests after modifying code.
- Treating observations as actions it has already performed.
- Losing track of intermediate states in long tasks and repeating the same action.

## V. Why a Language World Model When the Real Environment Exists?

A Language World Model (LWM) is not simply about saving the cost of a real environment. More importantly, it provides another scaling axis for agent training: the real environment continues to exist, but the model can additionally learn a verbalized simulation capability of "how the environment responds."

The first usage is **decoupling**: treating the LWM as a turn-level environment simulator. Real environments often have hard constraints: they require sandboxes, GUI VMs, real search engines, real apps, or system permissions; some environments are irreversible, have commercial restrictions, or lack public implementations. An LWM can turn the observations/transitions of these environments into controllable simulations, allowing agents to be trained at a larger scale.

The point here is not "cheap replacement," but **controllability**. Real search/tool environments usually return standard results, but an LWM can deliberately simulate:

- Partial search results: Forcing the agent to perform multi-turn follow-ups and cross-validation.
- Tool failures: Command timeouts, API rate limits, or insufficient permissions.
- Environment perturbations: Web structure changes, disk space exhaustion, or concurrent file modifications.
- Rare edge cases: Scenarios rarely encountered in real systems but which break tasks when they do occur.

Such targeted perturbations can expose weaknesses in an agent that are not easily covered in real rollouts. In other words, an LWM does not replace the real environment; it supplements it with a training distribution that is controllable, scalable, and directionally sampleable.

The second usage is **unifying**: integrating world-modeling capabilities directly into the agent's foundation model. The agent doesn't just learn "what to do now," but also learns a future-oriented thinking pattern:

```text
If I execute this action,
what observation might the environment return?
Which states will change?
Which constraints still hold?
What should I be prepared to verify next?
```

This is parallel to reflection as a meta-level thinking process. Reflection is looking back at the past: "Where did I go wrong?" Simulation is predicting the future: "What will happen if I do this next?" A strong agent often requires both to coexist.

## VI. Relationship with Model-Based RL

Model-based RL explicitly learns a dynamics model and uses it for planning or imagination rollouts:

```text
real experience
  -> train dynamics model
  -> imagined rollout
  -> improve policy
```

Model-free RL learns a policy or value more directly, without necessarily predicting the environment explicitly. The trade-off is:

| Method | Pros | Cons |
| --- | --- | --- |
| Model-free | Direct implementation, does not rely on prediction accuracy | Low sample efficiency, weak planning capability |
| Model-based | Can use imagined trajectories to improve sample efficiency, suitable for planning | Model errors accumulate; incorrect predictions mislead the policy |

LLM agent training faces similar issues: if the model's internal prediction of the environment is wrong, its reasoning will drift; if it doesn't predict consequences at all and relies solely on trial and error, the cost is very high.

## VII. How to Answer in an Interview

A concise answer:

> A world model is a model's internal representation of environmental state transitions. It allows the model to not only know "what is the current state," but also to predict "what might happen next if I take a certain action." In agents and RL, this capability directly impacts planning, tool use, long-horizon decision-making, and sample efficiency.

Add a system-level perspective:

> For an LLM agent, the world model includes not just the physical world, but also codebases, terminals, web pages, tools, and the state of user tasks. A strong agent needs to connect observations, actions, state changes, and rewards; otherwise, it is limited to text imitation and will struggle to complete long-range tasks reliably.

## VIII. To Be Added

- JEPA / predictive representation
- Dreamer / model-based RL
- Video world models
- Agent memory vs. world model
- World model eval: prediction, planning, counterfactual, causal consistency
- Relationship between world models and agentic security / red teaming
