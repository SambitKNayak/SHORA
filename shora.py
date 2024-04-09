from io import StringIO
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import csv
import math
import random
import numpy as np

# Function to load synthetic VRPTW data from CSV
def load_data_from_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = []
        for row in reader:
            data.append({
                'id': int(row['id']),
                'x': float(row['x']),
                'y': float(row['y']),
                'demand': int(row['demand']),
                'start_time': int(row['start_time']),
                'end_time': int(row['end_time'])
            })
    return data

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

# Function for exploring neighborhood (swap two customers in a route)
def explore_neighborhood(route):
    new_route = route.copy()
    idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
    new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
    return new_route

# Function to load synthetic VRPTW data from UploadedFile
def load_data_from_csv(uploaded_file):
    data = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(data)
    return df.to_dict('records')

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

# Function to calculate savings for all possible pairs of customers
def calculate_savings(customers):
    savings = {}
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            saving = euclidean_distance(customers[i], customers[0]) + euclidean_distance(customers[j], customers[0]) - euclidean_distance(customers[i], customers[j])
            savings[(i, j)] = saving
    return savings

# Function to implement Clarke-Wright Savings Algorithm
def clarke_wright_savings(customers):
    savings = calculate_savings(customers)
    sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)
    routes = [[i] for i in range(len(customers))]
    for (i, j), saving in sorted_savings:
        route_i = next((route for route in routes if i in route), None)
        route_j = next((route for route in routes if j in route), None)
        if route_i and route_j and route_i != route_j:
            if route_i[0] == i:
                route_i.reverse()
            if route_j[-1] == j:
                route_j.reverse()
            route_i.extend(route_j)
            routes.remove(route_j)
    return routes

# Function to implement Nearest Neighbor Algorithm
def nearest_neighbor_algorithm(customers):
    vehicle_capacity = 25
    depot = customers[0]
    unvisited_customers = set(range(1, len(customers)))
    routes = [[]]
    while unvisited_customers:
        current_customer = routes[-1][-1] if routes[-1] else 0
        nearest_customer = min(unvisited_customers, key=lambda c: euclidean_distance(customers[current_customer], customers[c]))
        route_demand = sum(customers[customer_id]['demand'] for customer_id in routes[-1])
        if route_demand + customers[nearest_customer]['demand'] <= vehicle_capacity:
            routes[-1].append(nearest_customer)
            unvisited_customers.remove(nearest_customer)
        else:
            routes.append([])
    return routes

def calculate_total_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += calculate_distance(route[i], route[i + 1])
    return total_distance

def calculate_distance(coord1, coord2):
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    # If the dimensions are different, pad the smaller vector with zeros
    if len(coord1) != len(coord2):
        max_length = max(len(coord1), len(coord2))
        coord1 = np.pad(coord1, (0, max_length - len(coord1)), mode='constant')
        coord2 = np.pad(coord2, (0, max_length - len(coord2)), mode='constant')
    return np.linalg.norm(coord1 - coord2)
# Simulated Annealing algorithm
def simulated_annealing(customers, initial_route, initial_temperature, cooling_rate, num_iterations):
    current_route = initial_route
    current_cost = calculate_total_distance(current_route)
    best_route = current_route
    best_cost = current_cost
    temperature = initial_temperature

    for iteration in range(num_iterations):
        new_route = explore_neighborhood(current_route)
        new_cost = calculate_total_distance(new_route)
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_route = new_route
            current_cost = new_cost
        if new_cost < best_cost:
            best_route = new_route
            best_cost = new_cost
        temperature *= cooling_rate

    return best_route

# Hybrid Routing Optimization Algorithm
def hroa(customers, vehicles, population_size, num_generations, tournament_size, crossover_rate, mutation_rate, initial_temperature, cooling_rate, sa_num_iterations):
    # Step 1: Initial Solution Generation (Nearest Neighbor)
    initial_routes = nearest_neighbor_algorithm(customers)
    
    # Step 2: Improvement Phase with Clarke-Wright Savings Algorithm
    improved_routes = clarke_wright_savings(customers)
    
    # Step 3: Improvement Phase with Genetic Algorithm
    improved_routes += genetic_algorithm(customers, vehicles, population_size, num_generations, tournament_size, crossover_rate, mutation_rate)
    
    # Step 4: Improvement Phase with Simulated Annealing Algorithm
    improved_routes += simulated_annealing(customers, initial_routes, initial_temperature, cooling_rate, sa_num_iterations)
    
    # Step 5: Fine-Tuning and Local Optimization with Sweep Algorithm
    final_routes = sweep_algorithm(customers)
    
    # Combine initial_routes and improved_routes into final_routes
    final_routes += initial_routes + improved_routes
    
    return final_routes

# Function to generate initial population
def generate_initial_population(customers, num_vehicles, population_size):
    population = []
    for _ in range(population_size):
        route = [0]  # Start with depot as the first customer
        remaining_customers = list(range(1, len(customers)))
        while remaining_customers:
            next_customer = random.choice(remaining_customers)
            route.append(next_customer)
            remaining_customers.remove(next_customer)
        route.append(0)  # Return to depot
        population.append(route)
    return population

# Function for tournament selection
def tournament_selection(population, customers, tournament_size):
    selected_parents = []
    while len(selected_parents) < 2:
        tournament_pool = random.sample(population, tournament_size)
        best_route = max(tournament_pool, key=lambda route: evaluate_fitness(route, customers))
        selected_parents.append(best_route)
    return selected_parents

# Function for partially mapped crossover (PMX)
def pmx_crossover(parent1, parent2):
    crossover_point1 = random.randint(1, len(parent1) - 2)
    crossover_point2 = random.randint(crossover_point1 + 1, len(parent1) - 1)
    mapping = {}
    for i in range(crossover_point1, crossover_point2 + 1):
        mapping[parent1[i]] = parent2[i]
        mapping[parent2[i]] = parent1[i]
    offspring = [0] * len(parent1)
    for i in range(len(parent1)):
        if i < crossover_point1 or i > crossover_point2:
            offspring[i] = parent1[i]
        else:
            offspring[i] = mapping[parent1[i]]
    return offspring

# Function for mutation (swap mutation)
def swap_mutation(route):
    mutation_point1 = random.randint(1, len(route) - 2)
    mutation_point2 = random.randint(1, len(route) - 2)
    route[mutation_point1], route[mutation_point2] = route[mutation_point2], route[mutation_point1]
    return route

# Function to evaluate fitness of a route
def evaluate_fitness(route, customers):
    total_distance = 0
    for i in range(len(route) - 1):
        customer1 = customers[route[i]]
        customer2 = customers[route[i + 1]]
        total_distance += euclidean_distance(customer1, customer2)
    return 1 / total_distance  # Inverse of total distance as fitness

# Main Genetic Algorithm function
def genetic_algorithm(customers, num_vehicles, population_size, num_generations, tournament_size, crossover_rate, mutation_rate):
    population = generate_initial_population(customers, num_vehicles, population_size)
    for generation in range(num_generations):
        parents = tournament_selection(population, customers, tournament_size)
        if random.random() < crossover_rate:
            offspring = pmx_crossover(parents[0], parents[1])
        else:
            offspring = parents[random.randint(0, 1)]
        if random.random() < mutation_rate:
            offspring = swap_mutation(offspring)
        worst_individual_index = min(range(len(population)), key=lambda i: evaluate_fitness(population[i], customers))
        population[worst_individual_index] = offspring
    best_route = max(population, key=lambda route: evaluate_fitness(route, customers))
    return best_route

# Function to implement Sweep Algorithm
def sweep_algorithm(customers):
    vehicle_capacity = 25
    depot = customers[0]
    sorted_customers = sorted(customers[1:], key=lambda customer: math.atan2(customer['y'] - depot['y'], customer['x'] - depot['x']))
    routes = [[]]
    for customer in sorted_customers:
        if not routes[-1]:
            routes[-1].append(customer['id'])
        else:
            route_demand = sum(customers[customer_id - 1]['demand'] for customer_id in routes[-1])
            if route_demand + customer['demand'] <= vehicle_capacity:
                routes[-1].append(customer['id'])
            else:
                routes.append([customer['id']])
    return routes

# Function to plot routes using Plotly
def plot_routes(routes_data):
    fig = go.Figure(routes_data)
    fig.update_layout(
        title='Coordinates per Route (First 20 Routes)',
        xaxis=dict(title='X Coordinate'),
        yaxis=dict(title='Y Coordinate'),
        showlegend=True,
        hovermode='closest'
    )
    st.plotly_chart(fig)


# def save_routes_to_csv(routes, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Route', 'Customer IDs'])
#         for i, route in enumerate(routes, 1):
#             writer.writerow([f'Route {i}', ', '.join(map(str, route))])
#             st.success(f"Routes saved to {filename}")
                  

# Streamlit UI
def main():
    st.title('Sambit\'s Hybrid Optimization Routing Algorithm')

    # Upload synthetic dataset
    uploaded_file = st.file_uploader("Upload Synthetic Dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        vehicle_capacity = st.number_input('Vehicle Capacity:', min_value=1, value=25)
        population_size = st.number_input('Population Size:', min_value=1, value=50)
        num_generations = st.number_input('Number of Generations:', min_value=1, value=100)
        tournament_size = st.number_input('Tournament Size:', min_value=1, value=5)
        crossover_rate = st.slider('Crossover Rate:', min_value=0.0, max_value=1.0, value=0.8, step=0.05)
        mutation_rate = st.slider('Mutation Rate:', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        initial_temperature = st.number_input('Initial Temperature:', min_value=1, value=1000)
        cooling_rate = st.slider('Cooling Rate:', min_value=0.0, max_value=1.0, value=0.99, step=0.01)
        sa_num_iterations = st.number_input('Number of SA Iterations:', min_value=1, value=1000)

        number_of_routes = st.number_input('Number of Routes:', min_value=1, value=20)
        # Button to run the Hybrid Routing Optimization Algorithm
        if st.button('Run Hybrid Routing Optimization Algorithm'):
            customers = load_data_from_csv(uploaded_file)
            routes = hroa(customers, vehicle_capacity, population_size, num_generations, tournament_size, crossover_rate, mutation_rate, initial_temperature, cooling_rate, sa_num_iterations)

            # Plot the routes
            routes_data = []
            for i, route in enumerate(routes[:number_of_routes], 1):
                route_coordinates = [(customer['x'], customer['y']) for customer in customers if customer['id'] in route]
                route_x, route_y = zip(*route_coordinates)
                routes_data.append(go.Scatter(x=route_x, y=route_y, mode='lines+markers', name=f'Route {i}'))
            plot_routes(routes_data)
            
            # Define the filename for saving the routes
            filename = 'hybrid_Result.csv'
            # save_routes_to_csv(routes, filename)
            
             
            # Define the filename for saving the routes
            # filename = 'hybrid_Result.csv'
            # 
            # Save routes to a CSV file
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Route'] + ['Customer ' + str(i + 1) for i in range(len(routes[0]))])
                for i, route in enumerate(routes):
                    # Ensure route is a list
                    if not isinstance(route, list):
                        route = [route]  # Convert single integer to a list
                    # Convert each element of the route to string
                    route_str = [str(item) for item in route]
                    writer.writerow(['Route ' + str(i + 1)] + route_str)

            st.download_button(
                label="Download hybrid Result CSV",
                data=open("hybrid_Result.csv", 'rb').read(),
                file_name="hybrid_Result.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
