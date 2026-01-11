import numpy as np

# Configuración
gamma = 0.5
num_states = 5
V = np.zeros(num_states) # V inicial
iterations = 10

# Recompensas (0 en todos lados, 1 al salir del estado 5 según problemas anteriores)
# Nota: El problema dice "game does not halt", pero reward values "as in first few problems".
# Asumiremos R(s)=0 para s=1..4, y R(5)=1.
R = np.array([0, 0, 0, 0, 1])

for _ in range(iterations):
    V_new = np.zeros(num_states)
    for s in range(num_states): # s es indice 0 a 4 (representando 1 a 5)
        # Acciones: Stay (0), Left (1), Right (2)
        q_values = []
        
        # Action: Stay
        # Success 1/2 (stay), Fail depends on loc.
        # Boundary (0 or 4): Fail (1/2) -> neighbor.
        # Inner (1,2,3): Fail -> 1/4 left, 1/4 right.
        val_stay = 0
        if s == 0: # Leftmost
            val_stay = 0.5 * (R[s] + gamma*V[s]) + 0.5 * (R[s] + gamma*V[s+1])
        elif s == 4: # Rightmost
            val_stay = 0.5 * (R[s] + gamma*V[s]) + 0.5 * (R[s] + gamma*V[s-1])
        else: # Inner
            val_stay = 0.5 * (R[s] + gamma*V[s]) + 0.25 * (R[s] + gamma*V[s-1]) + 0.25 * (R[s] + gamma*V[s+1])
        q_values.append(val_stay)

        # Action: Move Left
        # Success 1/3 (move left). Fail 2/3.
        # Fail logic: Boundary Leftmost -> same as stay logic (1/2 stay, 1/2 neighbor).
        # Inner fail: "fails to move". Usualmente significa quedarse quieto (R + gV[s]).
        val_left = 0
        if s == 0: # Leftmost logic described in text
            # "ends up exactly the same as choosing to stay"
            val_left = val_stay
        else:
            # Inner or Rightmost moving left
            # Success (1/3) -> move to s-1
            term_success = (1/3) * (R[s] + gamma*V[s-1])
            # Fail (2/3) -> Stay? Texto dice "fails to move". Asumimos Stay.
            term_fail = (2/3) * (R[s] + gamma*V[s])
            val_left = term_success + term_fail
        q_values.append(val_left)

        # Action: Move Right
        val_right = 0
        if s == 4: # Rightmost logic described in text
             # "ends up exactly the same as choosing to stay"
            val_right = val_stay
        else:
            # Success (1/3) -> move to s+1
            term_success = (1/3) * (R[s] + gamma*V[s+1])
            # Fail (2/3) -> Stay
            term_fail = (2/3) * (R[s] + gamma*V[s])
            val_right = term_success + term_fail
        q_values.append(val_right)

        V_new[s] = max(q_values)
    V = V_new

print("V_100:", np.round(V, 4))