import pandas as pd
import sqlite3
import numpy as np
import time

curr_angle=122.6019568
# Read the first sheet
ds_curl_flexion = pd.read_csv('generalized_curl_flexion.csv')
ds_curl_extension = pd.read_csv('generalized_curl_extension.csv')

last_states=[123.9724515, 123.7027902, 123.4314282, 123.1569917, 122.880542]


import numpy as np

def build_state_vector(theta_now, theta_prev):

    dt = 0.01  # Fixed time step
    theta_dot = (theta_now - theta_prev) / dt
    return np.array([[theta_now], [theta_dot]])


import numpy as np

def compute_next_state(A_matrix, current_state):

    return A_matrix @ current_state


def get_lower_A_matrix(given_theta, phase):
    # Step 1: Connect and load only theta + index
    conn = sqlite3.connect("A_matrices.db")
    df = pd.read_sql_query(
        "SELECT step_index, theta FROM full_theta_with_A WHERE dataset_name = ? ORDER BY theta ASC",
        conn, params=(phase,)
    )

    # Step 2: Find bounds
    lower_index = None
    for i in range(len(df) - 1):
        theta1 = df.iloc[i]['theta']
        theta2 = df.iloc[i + 1]['theta']
        if theta1 <= given_theta <= theta2:
            lower_index = int(df.iloc[i]['step_index'])
            break

    cursor = conn.cursor()

    # Step 3: Handle out-of-range → return 2x2 zero matrix
    if lower_index is None:
        conn.close()
        return {
            "matched_theta": given_theta,
            "step_index": -1,
            "A_matrix": np.zeros((2, 2))
        }

    # Step 4: Fetch corresponding 2x2 A matrix
    A_query = """
        SELECT a11, a12,
               a21, a22
        FROM full_theta_with_A
        WHERE dataset_name = ? AND step_index = ?
    """
    cursor.execute(A_query, (phase, lower_index))
    row = cursor.fetchone()
    conn.close()

    A_matrix = np.array(row).reshape(2, 2)

    return {
        "matched_theta": df[df['step_index'] == lower_index]['theta'].values[0],
        "step_index": lower_index,
        "A_matrix": A_matrix
    }

def direction(past_angles):

    diffs = [past_angles[i+1] - past_angles[i] for i in range(len(past_angles)-1)]

    avg_diff = sum(diffs) / len(diffs)
    print(avg_diff)
    if avg_diff > 0.1:
        return 1
    elif avg_diff < -0.1:
        return -1
    else:
        return 0

    return value

def assign_state(state,dir):
    if dir>0:
        result = get_lower_A_matrix(state, "curl_flexion")
        return result['A_matrix']

    elif dir<0:
        result = get_lower_A_matrix(state, "curl_flexion")
        return result['A_matrix']
    else:
        pass



if __name__ == "__main__":
    direction=direction(last_states)
    start_time = time.time()
    state_matrix_A=assign_state(curr_angle,direction)
    
    theta_now = curr_angle

    theta_prev = last_states[-1]
    state_vector_X = build_state_vector(theta_now, theta_prev)

    estimated_next_angle=compute_next_state(state_matrix_A, state_vector_X)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    print(estimated_next_angle)