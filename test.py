# THỰC HÀNH 26
# Reinforcement Learning

import numpy as np

# Định nghĩa môi trường
maze_size = 4
start = (0, 0)
goal = (3, 3)

# Khởi tạo giá trị Q
Q_table = np.zeros((maze_size, maze_size, 4))

# Siêu tham số
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Hàm cập nhật giá trị Q
def update_Q_table(pos, action, reward, next_pos):
    Q_table[pos[0], pos[1], action] = Q_table[pos[0], pos[1], action] + \
        alpha * (reward + gamma * np.max(Q_table[next_pos[0], next_pos[1], :]) - Q_table[pos[0], pos[1], action])

# Hàm chọn hành động
def choose_action(pos):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 4)
    else:
        return np.argmax(Q_table[pos[0], pos[1], :])


# Huấn luyện Q-learning
for episode in range(10000):
    pos = start
    while pos != goal:
        action = choose_action(pos)
        next_pos = pos
        
        if action == 0 and pos[0] > 0:                      # lên
            next_pos = (pos[0] - 1, pos[1])
        elif action == 1 and pos[0] < maze_size - 1:        # xuống
            next_pos = (pos[0] + 1, pos[1])
        elif action == 2 and pos[1] > 0:                    # trái
            next_pos = (pos[0], pos[1] - 1)
        elif action == 3 and pos[1] < maze_size - 1:        # phải
            next_pos = (pos[0], pos[1] + 1)
        
        reward = 0
        if next_pos == goal:
            reward = 1
        elif next_pos == pos:
            reward = -1
        
        update_Q_table(pos, action, reward, next_pos)
        pos = next_pos

# Xuất ra giá trị Q
print("Bảng giá trị Q")
print(Q_table)

# Bảng giá trị Q
# [[[-0.468559   -0.468559   -0.59049     ]
#   [-0.40951     0.65318357  0.531441    0.6561    ]
#   [-0.3439      0.729       0.59049     0.59049   ]
#   [-0.48923147  0.45838396  0.6561     -0.46284171]]

#  [[ 0.531441    0.21189339 -0.7073087   0.3377304 ]
#   [ 0.59049     0.52237054  0.39832593  0.72895002]
#   [ 0.6561      0.81        0.65027585  0.56933089]
#   [ 0.59049     0.40629269  0.69406316 -0.58365799]]

#  [[ 0.45697462  0.         -0.23296657  0.        ]
#   [ 0.45869031  0.48482885  0.18986755  0.81      ]
#   [ 0.729       0.9         0.729       0.48234574]
#   [ 0.55644463  0.468559    0.3317031  -0.38848697]]

#  [[ 0.21904682  0.         -0.093693    0.        ]
#   [ 0.729      -0.16114555  0.06322521  0.51257951]
#   [ 0.81       -0.1         0.6561      1.        ]
#   [ 0.          0.          0.          0.        ]]]