import heapq
import random
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===================== Variables globales =====================
graph = {
    'A': {'B': 5, 'C': 2, 'D': 6},
    'B': {'C': 2, 'E': 5},
    'C': {'D': 3, 'E': 9, 'F': 6},
    'D': {'F': 1},
    'E': {'G': 2},
    'F': {'E': 1, 'G': 6},
    'G': {}
}

fixed_pos = {
    'A': (0, 0),
    'B': (2, 2),
    'C': (2, 0),
    'D': (2, -2),
    'E': (3, 1.5),
    'F': (3, -1.5),
    'G': (4, 0)
}

G = nx.DiGraph()
for node, neighbors in graph.items():
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)
pos = fixed_pos

steps = []
step_index = -1
auto_simulation_running = False

# ===================== Algorithme Dijkstra =====================
def dijkstra_iter(graph, start):
    distances = {node: float('inf') for node in graph}
    previous = {node: None for node in graph}
    distances[start] = 0
    queue = [(0, start)]
    visited = set()
    local_steps = []

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        local_steps.append({'distances': distances.copy(), 'current_node': current_node, 'visited': visited.copy()})

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))
                local_steps.append({'distances': distances.copy(), 'current_node': current_node, 'visited': visited.copy()})

    return local_steps, distances, previous

def reconstruct_path(previous, start, end):
    if end not in previous:  # nœud absent
        return []
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = previous.get(node)
    path.reverse()
    if path[0] == start:
        return path
    return []

def compute_complexity():
    V = len(graph)
    E = sum(len(neighbors) for neighbors in graph.values())
    return f"Complexité : O(({V}+{E}) log {V})"

# ===================== Affichage =====================
def show_initial_graph():
    global ax, text, steps, step_index
    step_index = -1
    ax.clear()
    ax.set_title("Graphe de départ", fontsize=18)
    ax.set_aspect('auto')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=18, ax=ax)
    for u, v, weight in G.edges(data=True):
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): weight['weight']}, ax=ax, font_size=14)
    text.set("Graphe de départ affiché. Cliquez sur 'Suivant' pour démarrer la simulation Dijkstra.")
    canvas.draw_idle()

def show_step(step_index):
    global ax, text, steps
    step = steps[step_index]
    distances = step['distances']
    visited = step['visited']

    ax.clear()
    ax.set_title("Simulation Dijkstra", fontsize=18)
    ax.set_xlabel("Cliquez sur 'Suivant' ou 'Précédent'...", fontsize=14)
    ax.set_aspect('auto')

    # Dessiner les nœuds
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=1000, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(visited), node_color='yellow', node_size=1000, ax=ax)

    # Dessiner les arêtes
    nx.draw_networkx_edges(G, pos, edgelist=[], arrows=True, ax=ax)
    for u, v, weight in G.edges(data=True):
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): weight['weight']}, ax=ax, font_size=14)

    # Étiquettes distances
    labels = {node: f"{node} ({distances[node]})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='black', ax=ax)

    start_node = node_combobox.get()
    destination = dest_combobox.get()
    text_str = f"Étape {step_index + 1}/{len(steps)}\nDistances depuis {start_node} :\n"
    text_str += "\n".join([f"{n}: {distances[n]}" for n in G.nodes()])

    # Si dernière étape, afficher chemin le plus court vers destination
    if step_index == len(steps) - 1:
        _, final_distances, previous = dijkstra_iter(graph, start_node)
        shortest_path = reconstruct_path(previous, start_node, destination)
        if shortest_path:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)],
                width=4, edge_color='red', arrows=True, ax=ax
            )
            text_str += f"\n\nChemin le plus court vers {destination} : {' -> '.join(shortest_path)} (Distance: {final_distances[destination]})"
        else:
            text_str += f"\n\nPas de chemin vers {destination}."
        text_str += f"\n{compute_complexity()}"

    text.set(text_str)
    canvas.draw_idle()

# ===================== Navigation =====================
def next_step(event=None):
    global step_index, steps
    if step_index <= -1:
        start_node = node_combobox.get()
        steps, _, _ = dijkstra_iter(graph, start_node)
        step_index = 0
        show_step(step_index)
        return
    if step_index < len(steps) - 1:
        step_index += 1
        show_step(step_index)

def prev_step(event=None):
    global step_index
    if step_index <= 0:
        step_index = -1
        show_initial_graph()
    else:
        step_index -= 1
        show_step(step_index)

def reset_graph():
    global graph, fixed_pos, G, pos, steps, step_index
    graph = {
        'A': {'B': 5, 'C': 2, 'D': 6},
        'B': {'C': 2, 'E': 5},
        'C': {'D': 3, 'E': 9, 'F': 6},
        'D': {'F': 1},
        'E': {'G': 2},
        'F': {'E': 1, 'G': 6},
        'G': {}
    }
    fixed_pos = {
        'A': (0, 0),
        'B': (2, 2),
        'C': (2, 0),
        'D': (2, -2),
        'E': (3, 1.5),
        'F': (3, -1.5),
        'G': (4, 0)
    }
    G.clear()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    pos = fixed_pos
    steps = []
    step_index = -1
    node_combobox['values'] = list(graph.keys())
    node_combobox.set('A')
    dest_combobox['values'] = list(graph.keys())
    dest_combobox.set('G')
    show_initial_graph()

def generate_random_graph():
    global graph, fixed_pos, G, pos
    nodes = [chr(65+i) for i in range(random.randint(5,8))]
    graph = {n:{} for n in nodes}
    for n in nodes:
        edges = random.sample([x for x in nodes if x != n], random.randint(1,3))
        for e in edges:
            graph[n][e] = random.randint(1,10)
    fixed_pos = {n: (random.uniform(0,5), random.uniform(-3,3)) for n in nodes}
    G.clear()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    pos = fixed_pos
    node_combobox['values'] = list(graph.keys())
    node_combobox.set(nodes[0])
    dest_combobox['values'] = list(graph.keys())
    dest_combobox.set(nodes[-1])
    show_initial_graph()

# ===================== Simulation automatique =====================
def start_auto_simulation():
    global auto_simulation_running
    auto_simulation_running = True
    auto_simulate_step()

def stop_auto_simulation():
    global auto_simulation_running
    auto_simulation_running = False

def auto_simulate_step():
    global step_index, steps
    if step_index == -1:
        start_node = node_combobox.get()
        steps, _, _ = dijkstra_iter(graph, start_node)
    if auto_simulation_running and step_index < len(steps) - 1:
        step_index += 1
        show_step(step_index)
        root.after(1000, auto_simulate_step)

# ===================== Interface Tkinter =====================
root = Tk()
root.title("Simulation Dijkstra")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

text = StringVar()
text.set("Cliquez sur 'Suivant' pour démarrer la simulation")
label = Label(root, textvariable=text, font=("Arial",12))
label.pack(side=BOTTOM)

control_frame = Frame(root)
control_frame.pack(side=TOP, pady=5)

prev_button = Button(control_frame, text="Précédent", command=prev_step)
prev_button.pack(side=LEFT, padx=5)
next_button = Button(control_frame, text="Suivant", command=next_step)
next_button.pack(side=LEFT, padx=5)
auto_button = Button(control_frame, text="Lancer Auto", command=start_auto_simulation)
auto_button.pack(side=LEFT, padx=5)
stop_button = Button(control_frame, text="Arrêter Auto", command=stop_auto_simulation)
stop_button.pack(side=LEFT, padx=5)
reset_button = Button(control_frame, text="Réinitialiser", command=reset_graph)
reset_button.pack(side=LEFT, padx=5)
random_button = Button(control_frame, text="Graphe Aléatoire", command=generate_random_graph)
random_button.pack(side=LEFT, padx=5)

node_combobox = ttk.Combobox(control_frame, values=list(graph.keys()), width=5, font=("Arial",12))
node_combobox.pack(side=LEFT, padx=5)
node_combobox.set("A")

dest_combobox = ttk.Combobox(control_frame, values=list(graph.keys()), width=5, font=("Arial",12))
dest_combobox.pack(side=LEFT, padx=5)
dest_combobox.set("G")

start_button = Button(control_frame, text="Démarrer Dijkstra", command=show_initial_graph)
start_button.pack(side=LEFT, padx=5)

show_initial_graph()
root.mainloop()

