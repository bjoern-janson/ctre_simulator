import numpy as np

from env import ScalarLatentEnvironment
from state_agent import StateAgent
from ree_agent import REEAgent
from evaluator import SchemaEvaluator


STEPS = 5000
SEEDS = range(1, 11)

SLEEP_WINDOW = 100


def collect_batch(agent, env, steps):
    """
    Runs interaction loop.

    Returns:
        observations
        targets
        errors
    """

    observations = []
    targets = []
    errors = []

    state = env.reset()

    for _ in range(steps):

        observation, metadata = env.step(
            state
        )

        prediction = agent.predict(
            observation
        )

        error = (
            prediction
            -
            observation["target"]
        )

        if hasattr(agent, "update"):
            agent.update(
                observation,
                observation["target"]
            )

        observations.append(
            observation
        )

        targets.append(
            observation["target"]
        )

        errors.append(
            error
        )


    return (
        observations,
        np.array(targets),
        np.array(errors)
    )



def evaluate_transfer(agent):
    """
    Tests whether learned representation
    transfers to unseen latent distribution.
    """

    env = ScalarLatentEnvironment(
        z_range=(-0.8, 0.8),
        seed=999
    )

    errors = []

    state = env.reset()

    for _ in range(1000):

        observation, _ = env.step(
            state
        )

        prediction = agent.predict(
            observation
        )

        errors.append(
            abs(
                prediction
                -
                observation["target"]
            )
        )


    return float(
        np.mean(errors)
    )



def run_state_agent(seed):

    env = ScalarLatentEnvironment(
        z_range=(0.1, 0.3),
        seed=seed
    )

    agent = StateAgent()

    collect_batch(
        agent,
        env,
        STEPS
    )

    transfer_error = evaluate_transfer(
        agent
    )

    return {
        "agent": "state",
        "seed": seed,
        "transfer_error": transfer_error,
        "mutations": 0,
        "representation_size":
            agent.representation_size()
    }



def run_ree_agent(seed):

    env = ScalarLatentEnvironment(
        z_range=(0.1, 0.3),
        seed=seed
    )

    agent = REEAgent()

    evaluator = SchemaEvaluator()


    observations = []
    targets = []
    errors = []

    state = env.reset()


    for step in range(STEPS):

        observation, _ = env.step(
            state
        )

        prediction = agent.predict(
            observation
        )

        error = (
            prediction
            -
            observation["target"]
        )


        agent.update(
            observation,
            observation["target"]
        )


        observations.append(
            observation
        )

        targets.append(
            observation["target"]
        )

        errors.append(
            error
        )


        #
        # Offline sleep cycle
        #
        if (
            agent.should_sleep()
            and len(errors) > SLEEP_WINDOW
        ):
            print("SLEEP TRIGGERED", step)

            recent_obs = (
                observations[-SLEEP_WINDOW:]
            )

            recent_errors = np.array(
                errors[-SLEEP_WINDOW:]
            )


            candidates = (
                agent.sleep_cycle(
                    recent_obs,
                    recent_errors
                )
            )


            feature_matrix = np.array(
                [
                    [
                        o["velocity"],
                        o["context"]
                    ]
                    for o in recent_obs
                ]
            )


            scores = []

            for candidate in candidates:

                score = evaluator.evaluate(
                    candidate,
                    feature_matrix,
                    np.array(
                        [
                            o["target"]
                            for o in recent_obs
                        ]
                    ),
                    feature_matrix,
                    np.array(
                        [
                            o["target"]
                            for o in recent_obs
                        ]
                    )
                )

                scores.append(score)


            best = candidates[
                int(
                    np.argmax(scores)
                )
            ]


            if best.operator != "null":

                agent.commit_hypothesis(
                    best
                )


    transfer_error = evaluate_transfer(
        agent
    )

    return {
        "agent": "ree",
        "seed": seed,
        "transfer_error": transfer_error,
        "mutations":
            agent.mutation_count,
        "representation_size":
            agent.representation_size()
    }



def main():

    results = []


    for seed in SEEDS:

        print(
            f"running seed {seed}"
        )

        results.append(
            run_state_agent(seed)
        )

        results.append(
            run_ree_agent(seed)
        )


    print("\nresults\n")

    for result in results:
        print(result)



    state_scores = [
        r["transfer_error"]
        for r in results
        if r["agent"] == "state"
    ]

    ree_scores = [
        r["transfer_error"]
        for r in results
        if r["agent"] == "ree"
    ]


    print("\nsummary")
    print(
        "state mean:",
        np.mean(state_scores)
    )

    print(
        "ree mean:",
        np.mean(ree_scores)
    )


if __name__ == "__main__":
    main()
