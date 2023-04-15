import numpy as np


if __name__ == '__main__':
    # state num: 3
    # action num: 2
    state_num = 3
    action_num = 2
    
    gamma = 0.9
    trans = np.zeros(shape=(action_num, state_num, state_num))
    value = [0, 0, 0]
    reward = [1, 3, -1]
    trans[0] = np.array([[0.8, 0.1, 0.1], [0.2, 0.1, 0.7], [0.9, 0.1, 0.0]])
    trans[1] = np.array([[0.1, 0.1, 0.8], [0.7, 0.1, 0.2], [0.9, 0.0, 0.1]])

    # 2 iteration
    for i in range(2):
        for state_i in range(state_num):
            state_value = value[state_i]

            best_action_i = None
            max_v = -1e9
            for action_i in range(action_num):
                action_trans = trans[action_i]
                v = 0
                for state_j in range(state_num):
                    v += action_trans[state_i, state_j] * (reward[state_j] + gamma * state_value)
                
                if v > max_v:
                    best_action_i = action_i
                    max_v = v
                pass
            pass

            value[state_i] = max_v
        print("iteration {}:".format(i), value)
        pass