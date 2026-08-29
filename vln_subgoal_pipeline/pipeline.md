# VLN Pipeline with Subgoal Handling

This document describes a Vision-and-Language Navigation (VLN) pipeline that decomposes a long-horizon instruction into subgoals, executes them one at a time on a robot, and loops until the task is complete.

## Stages

1. **Long-Horizon VLN Data** — Input dataset of long-horizon vision-and-language navigation instructions/trajectories.
2. **Fine-tune Cosmos 3 Reasoner** — The Cosmos 3 reasoning model is fine-tuned on the VLN data.
3. **Subgoal Decomposition** — The fine-tuned reasoner breaks a long-horizon instruction into a sequence of smaller subgoals.
4. **Subgoal Queue** — Decomposed subgoals are **enqueued** into a queue (FIFO store of pending subgoals).
5. **Get Next Subgoal** — Dequeues the next subgoal from the Subgoal Queue to be acted on.
6. **Locate Anything / Pointing** — Given the current subgoal, an open-vocabulary "locate anything" / pointing model identifies the target object or location in the scene.
7. **3D Grounding** — The 2D detection/pointing result is grounded into 3D space (i.e., mapped to real-world coordinates).
8. **Nav2** — The 3D-grounded target is passed to the Nav2 navigation stack, which plans a path to it.
9. **Robot** — The robot physically executes the navigation plan (moves toward the target).

## Decision Loop

10. **Subgoal Reached?**
    - **No** → loop back to **Locate Anything / Pointing** to re-detect/re-ground and keep navigating toward the current subgoal.
    - **Yes** → proceed to check the queue.
11. **More Subgoals in Queue?**
    - **Yes** → loop back to **Get Next Subgoal** to dequeue and start the next subgoal.
    - **No** → proceed to **Task Complete**.
12. **Task Complete** — Terminal state once all subgoals have been executed successfully.

## Pipeline Summary (Flow)

```
Long-Horizon VLN Data
        │
        ▼
Fine-tune Cosmos 3 Reasoner
        │
        ▼
Subgoal Decomposition ──enqueue──▶ Subgoal Queue
                                        │
                        ┌───────────────┘
                        ▼
              Get Next Subgoal ◀────────────────┐
                        │                        │ Yes
                        ▼                        │
           Locate Anything / Pointing ◀───┐  More Subgoals in Queue?
                        │                 │            ▲
                        ▼                 │ No         │ Yes
                  3D Grounding            │            │
                        │                 │            │
                        ▼                 │            │
                      Nav2                │            │
                        │                 │            │
                        ▼                 │            │
                      Robot               │            │
                        │                 │            │
                        ▼                 │            │
              Subgoal Reached? ───────────┘            │
                        │ Yes                           │
                        └───────────────────────────────┘
                                        │ No (from More Subgoals?)
                                        ▼
                                 Task Complete
```

## Key Design Points

- **Training phase** (left side): a long-horizon VLN dataset is used to fine-tune the Cosmos 3 Reasoner, which then performs subgoal decomposition and populates the Subgoal Queue.
- **Execution phase** (right side): the pipeline dequeues one subgoal at a time and drives perception (Locate Anything / Pointing → 3D Grounding), navigation (Nav2), and actuation (Robot).
- **Two feedback loops**:
  - An inner loop (**Subgoal Reached? → No**) keeps re-perceiving and navigating until the current subgoal is achieved.
  - An outer loop (**More Subgoals in Queue? → Yes**) advances to the next subgoal once the current one is reached, repeating until the queue is empty.
- The pipeline terminates at **Task Complete** only when the subgoal queue is exhausted.