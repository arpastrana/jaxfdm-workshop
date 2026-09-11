from jax import grad

# Define a loss function
def loss_fn(x):
    loss = 0.0
    for xi in x:
        loss = loss + xi ** 2
    return loss

# Create an array of values
x = [1.0, 2.0, 3.0, 4.0]

# Evaluate the loss function
L = loss_fn(x)  # 30.0

# Evaluate the gradient of the loss function
dL = grad(loss_fn)(x)  # [2.0, 4.0, 6.0, 8.0]
