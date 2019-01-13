import random


def walk_to_goal(start):
    """
    Walks randomly on the grid, until the goal is reached

    :param start: The state to start in
    :return: The number of actions needed
    """
    num_steps = 0
    current_state = start

    # Loop until we reach the goal
    while current_state != 0:
        # Pick the action randomly, except when we are already in the upper state
        action = random.choice([-1, 1]) if current_state < 10 else -1

        # Apply the state change
        current_state += action
        num_steps += 1

    return num_steps


if __name__ == '__main__':
    num_of_steps = [0] * 11     # Stores how many steps where needed for every start-state
    counts = [0] * 11           # Stores how often each start state was evaluated, to calculate the average later on

    for _ in range(500000):
        # Pick a random state
        s = random.randint(0, 10)

        num_of_steps[s] += walk_to_goal(s)
        counts[s] += 1

    for i in range(11):
        print("F({}) = {}".format(i, num_of_steps[i]/counts[i]))
