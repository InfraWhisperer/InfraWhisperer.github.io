---
layout: post
title: "Prefix-cache routing silently dies under disaggregation"
date: 2026-07-15
topic: Disaggregation
reading_time: "5 min"
description: "Split vLLM into prefill/decode under Dynamo and the KV-events subject moves from component.backend to component.prefill. A router still on the old subject gets zero events and fails open — no error, only silent round-robin."
tags: [dynamo, vllm, kv-cache, prefix-caching, disaggregation, nats]
---

My prefix-cache hit rate was zero. Not low — zero, across every request and every workload, for hours. The router in front of the fleet was wired right, and its prefix index had 14,000 nodes in it. It took me too long to stop debugging the scorer and go read what Dynamo actually publishes, and where.

The answer is a one-token subject mismatch that no log line will ever tell you about. If you route by prefix cache in front of disaggregated vLLM, you can hit this and never know — your routing quietly degrades to round-robin and everything still returns 200s.

## The setup

I run an external router in front of a Dynamo 1.2.1 deployment: prefix-aware placement, so a request lands on the worker that already holds its prompt prefix in KV and skips the recompute. To know *which* worker holds *which* prefix, the router builds a radix index from Dynamo's KV-cache event plane — the `BlockStored` / `BlockRemoved` events vLLM emits and Dynamo republishes on NATS. Standard push model: workers publish cache state, the router subscribes and indexes it, routing reads the index.

This worked in aggregated mode. Then I moved the same deployment to prefill/decode disaggregation — split the workers, one pool computes prefill and ships KV over NIXL, another pool decodes — and prefix affinity went dead.

## The symptom that looks like a scoring bug

The router's per-pod match-fraction metric was a clean zero on every worker, every request. Not noisy-low — exactly zero. My first instinct was the scorer: a sign flip, a bad normalization, a lookup keyed wrong. That was the wrong instinct, and I burned real time on it.

Two facts eventually pointed away from the scorer:

1. The index was **not empty** — 14,000 nodes. So something was populating it.
2. Every node in it was keyed to a **decode** worker. Zero prefill-worker keys.

That second fact is its own small trap. In disaggregation the request's routing primary is the decode worker (that's where the response streams from), so the router's own speculative fallback — the "I sent this prefix to worker X a moment ago, remember that" insert — was recording decode pods. But prefix affinity for *prefill* placement needs to know which *prefill* worker computed and cached the prefix. The index was full of the wrong role. And the authoritative signal that would have carried the right role — the engine's own `BlockStored` events — was nowhere in it.

So the real question wasn't "why is the scorer returning zero," it was "why am I receiving zero KV-cache events."

## Read the source, not the tea leaves

Dynamo builds the KV-event subject in one place. In `lib/llm/src/kv_router/indexer/subscriber.rs`:

```rust
let kv_event_subject = format!(
    "namespace.{}.component.{}.{}",
    component.namespace().name(),
    component.name(),
    KV_EVENT_SUBJECT
);
```

and `KV_EVENT_SUBJECT` is `"kv-events"` (`lib/kv-router/src/protocols.rs`). So the full subject is:

```
namespace.<namespace>.component.<component>.kv-events
```

The load-bearing token is `<component>` — the worker's Dynamo component name. And that name is not stable across serving topologies:

- Aggregated vLLM registers as component **`backend`** — that's the default (`lib/backend-common/src/args.rs`, `DYN_COMPONENT` defaults to `"backend"`; `components/src/dynamo/vllm/args.py` sets `dynamo_config.component = "backend"`).
- A disaggregated deployment overrides it per role: `DYN_COMPONENT=prefill` and `DYN_COMPONENT=decode`.

My subscription was still on `component.backend.kv-events` from the aggregated setup. Under disaggregation nothing publishes to `backend` anymore — the events are on `component.prefill.kv-events` and `component.decode.kv-events`. My subscriber matched nothing. Zero events, no error, index never learns a single prefill key.

## Confirm it on the wire

Reading the format is a hypothesis; the bus is the proof. A raw NATS subscribe on `>` while firing a burst of identical-prefix requests, and there it is:

```
  12  namespace.<ns>.component.prefill.kv-events      <== one per request
 150  namespace.<ns>.component.prefill.forward-pass-metrics
 135  namespace.<ns>.component.backend.forward-pass-metrics
  60  namespace.<ns>.kv_metrics
```

Twelve requests, twelve kv-events, all on `component.prefill.kv-events`. (The namespace token also carries a worker-hash suffix, so match it with a wildcard, not a literal.)

Two things worth pinning here, because they surprised me:

- **Only the prefill worker publishes prefix events (in a standard disagg config).** The decode worker is run with `--no-enable-prefix-caching` — it receives KV over NIXL, it doesn't own or recompute the prefix, so it emits nothing useful for prefix routing. All the affinity signal lives on the prefill side. If you're going to subscribe to one role, it's `prefill`.
- **There's a `backend` component still on the bus** — but only for `forward-pass-metrics`, not kv-events. Enough to make you think you're on the right namespace while you're on the wrong subject. Don't let the presence of *a* `backend.*` subject convince you your kv-events subscription is live.

## The fix

One line — point the subscription at the prefill component:

```
component.backend.kv-events   →   component.prefill.kv-events
```

Events start flowing on the next request. The index keys on prefill workers. And the behavior you actually wanted shows up immediately: fire the same prefix repeatedly and the requests concentrate on the one worker that cached it, instead of spraying across the pool. Match rate goes from a flat zero to tracking the real reuse in the workload.

## The lesson

Prefix-cache routing is only as good as the event-plane wiring underneath it, and disaggregation quietly re-topics that event plane per role. The failure mode is the nasty kind: it **fails open**. A wrong subject isn't a crash or a refused connection — it's a subscription that sits there healthy and receives nothing, so your smart router degrades to round-robin and every request still succeeds. Nothing in your logs says "you are no longer prefix-aware."

Concrete takeaways if you build or operate this:

- **If your cache-hit rate is a clean zero, suspect the subject before you touch the scorer.** Zero is a plumbing signature, not a scoring one. A real scorer bug is usually noisy-wrong, not perfectly-zero.
- **The component name is part of your contract.** `backend` for aggregated, `prefill`/`decode` for disaggregated. If your subscriber hardcodes a component, it breaks silently the day someone splits the workers.
- **Subscribe to `prefill` for prefix affinity.** In a standard disagg config the decode side has prefix caching off and publishes nothing you can route on.
- **Watch the bus, not only your metrics.** A 30-second raw NATS subscribe told me in one shot what hours of staring at a zero gauge did not.

Everything above is upstream Dynamo/vLLM behavior traced to its own source; observed on Dynamo 1.2.1, with the file references pinned to that release's commit below so they don't rot as line numbers drift.

---

*Source references, Dynamo 1.2.1, pinned to [ai-dynamo/dynamo@`919682d`](https://github.com/ai-dynamo/dynamo/commit/919682da679aa699d5bca9c872f4c1d9a530bbc0):*

- *KV-event subject format — [`kv_router/indexer/subscriber.rs#L48`](https://github.com/ai-dynamo/dynamo/blob/919682da679aa699d5bca9c872f4c1d9a530bbc0/lib/llm/src/kv_router/indexer/subscriber.rs#L48)*
- *`KV_EVENT_SUBJECT = "kv-events"` — [`kv-router/src/protocols.rs#L18`](https://github.com/ai-dynamo/dynamo/blob/919682da679aa699d5bca9c872f4c1d9a530bbc0/lib/kv-router/src/protocols.rs#L18)*
- *`DYN_COMPONENT` default `"backend"` — [`backend-common/src/args.rs#L36`](https://github.com/ai-dynamo/dynamo/blob/919682da679aa699d5bca9c872f4c1d9a530bbc0/lib/backend-common/src/args.rs#L36)*
- *component set to `"backend"` — [`vllm/args.py#L168`](https://github.com/ai-dynamo/dynamo/blob/919682da679aa699d5bca9c872f4c1d9a530bbc0/components/src/dynamo/vllm/args.py#L168)*
