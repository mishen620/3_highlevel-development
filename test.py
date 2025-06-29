# Define global buffer
theta_buffer = [155.0, 155.8, 156.4, 156.9, 157.5]

def update_theta_buffer(new_theta, max_size=5):
    """
    Updates the global theta_buffer by removing the first element
    and appending the new theta value.

    Args:
        new_theta (float): The latest theta value to add.
        max_size (int): Maximum size of the buffer.
    """
    global theta_buffer
    if len(theta_buffer) >= max_size:
        theta_buffer.pop(0)
    theta_buffer.append(new_theta)

update_theta_buffer(158.1)
print(theta_buffer)
# Output: [155.8, 156.4, 156.9, 157.5, 158.1]
