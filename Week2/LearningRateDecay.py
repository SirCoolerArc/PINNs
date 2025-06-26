starting_learning_rate = 1.
learning_rate_decay = 0.1
for step in range(20):
    learning_rate = starting_learning_rate * (1. / (1 + learning_rate_decay * step))

    print(f"The step no. is {step} & rate is {learning_rate}")