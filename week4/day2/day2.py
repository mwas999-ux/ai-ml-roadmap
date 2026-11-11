x = 2
w1 = 3
w2 = 4
target = 10

# FORWARD PASS: data flows left to right
h = w1 * x  # first "chef" (hidden value)
y = w2 * h  # second "chef" (final prediction)
loss = (y - target) ** 2
print("h:", h, "| prediction y:", y, "| loss:", loss)

# BACKWARD PASS: blame flows right to left
dloss_dy = 2 * (y - target)  # how much the loss changes if y changes
dloss_dw2 = dloss_dy * h  # w2's share of the blame
dloss_dh = dloss_dy * w2  # blame passed back to the first chef
dloss_dw1 = dloss_dh * x  # w1's share of the blame

print("Blame on w2:", dloss_dw2)
print("Blame on w1:", dloss_dw1)
learning_rate = 0.001

w1_new = w1 - learning_rate * dloss_dw1
w2_new = w2 - learning_rate * dloss_dw2
print("w1:", w1, "->", w1_new)
print("w2:", w2, "->", w2_new)

h_new = w1_new * x
y_new = w2_new * h_new
print("New prediction:", y_new, "| New loss:", (y_new - target) ** 2)
x = 2
w1 = 3.0
w2 = 4.0
target = 10
learning_rate = 0.001

for step in range(1, 21):
    # forward pass
    h = w1 * x
    y = w2 * h
    loss = (y - target) ** 2

    # backward pass
    dloss_dy = 2 * (y - target)
    dloss_dw2 = dloss_dy * h
    dloss_dw1 = dloss_dy * w2 * x

    # update both weights
    w1 = w1 - learning_rate * dloss_dw1
    w2 = w2 - learning_rate * dloss_dw2

    print(f"Step {step}: prediction={y:.3f}, loss={loss:.3f}")
