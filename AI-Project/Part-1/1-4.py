#For a 3x4 GridWorld with win and loss terminal states and no blocking states, implement a MDP solver with Policy iteration.


import numpy as np


rows, cols = 3, 4
terminal_states = {(0,3): 1, (1,3): -1}  
actions = ['U', 'D', 'L', 'R']
action_vectors = {'U': (-1,0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}


def reward(state):
    if state in terminal_states:
        return terminal_states[state]
    else:
        return -0.04


def in_grid(state):
    r,c = state
    return 0 <= r < rows and 0 <= c < cols


def next_state(state, action):
    if state in terminal_states:
        return state  
    dr, dc = action_vectors[action]
    new_state = (state[0] + dr, state[1] + dc)
    if in_grid(new_state):
        return new_state
    else:
        return state


gamma = 0.9
theta = 1e-4  

# Tüm durumlar
states = [(r,c) for r in range(rows) for c in range(cols)]

# Başlangıç politikası: her durumda rastgele eylem
policy = {}
for s in states:
    if s in terminal_states:
        policy[s] = None
    else:
        policy[s] = np.random.choice(actions)

# Değer fonksiyonu başlangıçta 0
V = {s:0 for s in states}

def policy_evaluation(policy, V):
    while True:
        delta = 0
        for s in states:
            if s in terminal_states:
                continue
            v = V[s]
            a = policy[s]
            s_prime = next_state(s, a)
            V[s] = reward(s) + gamma * V[s_prime]
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(policy, V):
    policy_stable = True
    for s in states:
        if s in terminal_states:
            continue
        old_action = policy[s]
        action_values = {}
        for a in actions:
            s_prime = next_state(s, a)
            action_values[a] = reward(s) + gamma * V[s_prime]
        best_action = max(action_values, key=action_values.get)
        policy[s] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable


iteration = 0
while True:
    iteration += 1
    V = policy_evaluation(policy, V)
    policy, stable = policy_improvement(policy, V)
    if stable:
        break


print("Policy iteration converged in", iteration, "iterations.")
print("\nOptimal Value Function:")
for r in range(rows):
    for c in range(cols):
        print(f"{V[(r,c)]:6.2f}", end=" ")
    print()

print("\nOptimal Policy:")
for r in range(rows):
    for c in range(cols):
        if (r,c) in terminal_states:
            print("  T  ", end=" ")
        else:
            print(f"  {policy[(r,c)]}  ", end=" ")
    print()
