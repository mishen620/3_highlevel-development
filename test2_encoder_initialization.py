initial_encoder_value=12.5

def Encoder_angle_initialization(curr_encoder_value):
    initial_encoder_value=curr_encoder_value



def curr_angle(curr_encoder_value):
    theta=20 #initialize these two values pakoo
    pulse=100 #initialize these two values pakoo
    calculated_current_angle=(curr_encoder_value-initial_encoder_value)*(theta/pulse)+155
    