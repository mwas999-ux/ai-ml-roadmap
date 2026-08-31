import numpy as np

# Pretend the "true" answer we're trying to find is 10
# Our model starts with a random bad guess
guess = 0.0
learning_rate = 0.1
target = 10

for step in range(20):
    error = guess - target  # how far off we are
    loss = error**2  # squared error, same as before
    gradient = 2 * error  # calculus: slope of the loss at this point
    guess = guess - learning_rate * gradient  # step downhill

    print(f"Step {step+1}: guess={guess:.3f}, loss={loss:.3f}")
