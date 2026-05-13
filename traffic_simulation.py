import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict, deque
import csv
import os

# === SECTION 1: CONFIGURATION & SCENARIOS ===

SIM_TIME = 3600  # Total simulation time in seconds (1 hour)
ANIMATION_DURATION = 600  # How many seconds of simulation to animate
ANIMATION_SPEED_MS = 50  # Interval per frame in milliseconds (20x speed)
PASSING_TIME = 2.0  # Seconds taken for a vehicle to clear the intersection on green

SCENARIOS = {
    "Rush Hour": {
        "arrivals": {"N": 1/3, "S": 1/3, "E": 1/3, "W": 1/3},
        "timings": {"NS_G": 30, "EW_G": 30, "Y": 5},
        "adaptive": False
    },
    "Off-Peak": {
        "arrivals": {"N": 1/12, "S": 1/12, "E": 1/12, "W": 1/12},
        "timings": {"NS_G": 30, "EW_G": 30, "Y": 5},
        "adaptive": False
    },
    "Unbalanced": {
        "arrivals": {"N": 1/3, "S": 1/3, "E": 1/10, "W": 1/10},
        "timings": {"NS_G": 30, "EW_G": 30, "Y": 5},
        "adaptive": False
    },
    "Emergency Priority": {
        "arrivals": {"N": 1/6, "S": 1/6, "E": 1/6, "W": 1/6},
        "timings": {"NS_G": 50, "EW_G": 15, "Y": 5},
        "adaptive": False
    },
    "Night Mode": {
        "arrivals": {"N": 1/20, "S": 1/20, "E": 1/20, "W": 1/20},
        "timings": {"NS_G": 45, "EW_G": 45, "Y": 5},
        "adaptive": False
    },
    "Adaptive Timing": {
        "arrivals": {"N": 1/6, "S": 1/6, "E": 1/6, "W": 1/6},
        "timings": {"NS_G": 30, "EW_G": 30, "Y": 5},
        "adaptive": True
    },
    "Congestion": {
        "arrivals": {"N": 1/1.5, "S": 1/1.5, "E": 1/1.5, "W": 1/1.5},
        "timings": {"NS_G": 30, "EW_G": 30, "Y": 5},
        "adaptive": False
    }
}

LANES = ["N", "S", "E", "W"]


# === SECTION 2: SIMPY PROCESSES (TrafficLight, VehicleGenerator, Vehicle) ===

class IntersectionModel:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # State: queues per lane
        self.queues = {lane: deque() for lane in LANES}
        # State: light colors ("G", "Y", "R")
        self.lights = {"N": "R", "S": "R", "E": "R", "W": "R"}
        
        # Metrics
        self.wait_times = {lane: [] for lane in LANES}
        self.queue_lengths_over_time = {lane: [] for lane in LANES} 
        self.vehicles_processed = {lane: 0 for lane in LANES}
        self.vehicles_arrived = {lane: 0 for lane in LANES}
        
        # Snapshot data for animation and plotting
        self.snapshots = []
        self.snapshot_60s = []
        
        # Start processes
        self.env.process(self.traffic_light_controller())
        for lane in LANES:
            self.env.process(self.vehicle_generator(lane))
            self.env.process(self.lane_processor(lane))
        self.env.process(self.snapshot_recorder())

    def traffic_light_controller(self):
        # NS starts first
        ns_active = True
        
        while True:
            # Set active directions
            active_lanes = ["N", "S"] if ns_active else ["E", "W"]
            inactive_lanes = ["E", "W"] if ns_active else ["N", "S"]
            
            # Phase: GREEN
            for lane in active_lanes: self.lights[lane] = "G"
            for lane in inactive_lanes: self.lights[lane] = "R"
            
            base_green = self.config["timings"]["NS_G" if ns_active else "EW_G"]
            green_duration = base_green
            
            yield self.env.timeout(green_duration)
            
            # Adaptive Timing Logic
            if self.config.get("adaptive"):
                # Check queue of active lanes to extend green
                max_queue = max([len(self.queues[l]) for l in active_lanes])
                if max_queue > 10:
                    extension = 10
                    # Limit total green to 60s
                    if green_duration + extension > 60:
                        extension = 60 - green_duration
                    yield self.env.timeout(extension)

            # Phase: YELLOW
            for lane in active_lanes: self.lights[lane] = "Y"
            yield self.env.timeout(self.config["timings"]["Y"])
            
            # Swap directions
            ns_active = not ns_active

    def vehicle_generator(self, lane):
        arrival_rate = self.config["arrivals"][lane]
        while True:
            # Poisson arrival process
            inter_arrival = random.expovariate(arrival_rate)
            yield self.env.timeout(inter_arrival)
            
            arrival_time = self.env.now
            self.vehicles_arrived[lane] += 1
            self.queues[lane].append(arrival_time)

    def lane_processor(self, lane):
        while True:
            if self.lights[lane] == "G" and len(self.queues[lane]) > 0:
                # Process one vehicle
                arrival_time = self.queues[lane].popleft()
                wait_time = self.env.now - arrival_time
                self.wait_times[lane].append(wait_time)
                
                # Takes PASSING_TIME to clear intersection
                yield self.env.timeout(PASSING_TIME)
                self.vehicles_processed[lane] += 1
            else:
                # Check frequently if light changed or queue updated
                # In discrete-event simulation, we could wait for events, but a small polling interval works
                yield self.env.timeout(0.1)

    def snapshot_recorder(self):
        while True:
            # Record every 1 second for animation
            self.snapshots.append({
                "time": self.env.now,
                "queues": {l: len(self.queues[l]) for l in LANES},
                "lights": {l: self.lights[l] for l in LANES},
            })
            
            # Record every 60 seconds for time-series charts
            if int(self.env.now) % 60 == 0:
                self.snapshot_60s.append({
                    "time": self.env.now,
                    "queues": {l: len(self.queues[l]) for l in LANES}
                })
                
            yield self.env.timeout(1.0)


# === SECTION 3: SIMULATION RUNNER ===

def run_simulation():
    print("Running simulations...")
    results = {}
    
    for scenario_name, config in SCENARIOS.items():
        print(f"  Simulating: {scenario_name}")
        env = simpy.Environment()
        model = IntersectionModel(env, config)
        env.run(until=SIM_TIME)
        results[scenario_name] = model
        
    return results


# === SECTION 4: METRICS & CSV OUTPUT ===

def generate_metrics(results):
    metrics = {}
    csv_data = [["Scenario", "Lane", "Avg Wait (s)", "Max Wait (s)", "Avg Queue", "Max Queue", "Throughput", "Efficiency %"]]
    
    for scenario, model in results.items():
        metrics[scenario] = {}
        total_processed = 0
        total_arrived = 0
        
        for lane in LANES:
            waits = model.wait_times[lane]
            qs = [s["queues"][lane] for s in model.snapshot_60s]
            
            avg_w = np.mean(waits) if waits else 0
            max_w = np.max(waits) if waits else 0
            avg_q = np.mean(qs) if qs else 0
            max_q = np.max(qs) if qs else 0
            
            throughput = model.vehicles_processed[lane]
            arrived = model.vehicles_arrived[lane]
            total_processed += throughput
            total_arrived += arrived
            
            eff = (throughput / arrived * 100) if arrived > 0 else 0
            
            metrics[scenario][lane] = {
                "avg_wait": avg_w, "max_wait": max_w,
                "avg_queue": avg_q, "max_queue": max_q,
                "throughput": throughput, "efficiency": eff
            }
            
            csv_data.append([
                scenario, lane, f"{avg_w:.2f}", f"{max_w:.2f}",
                f"{avg_q:.2f}", f"{max_q:.2f}", f"{throughput}", f"{eff:.1f}%"
            ])
            
        metrics[scenario]["TOTAL"] = {
            "throughput": total_processed,
            "efficiency": (total_processed / total_arrived * 100) if total_arrived > 0 else 0
        }
    
    # Write CSV
    with open("results_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    # Print Console Summary
    print("\n" + "="*105)
    print(f"{'SCENARIO':<20} | {'LANE':<5} | {'AVG WAIT':<10} | {'MAX WAIT':<10} | {'AVG Q':<8} | {'MAX Q':<8} | {'THRUPUT':<8} | {'EFFICIENCY'}")
    print("-" * 105)
    for row in csv_data[1:]:
        print(f"{row[0]:<20} | {row[1]:<5} | {row[2]:<10} | {row[3]:<10} | {row[4]:<8} | {row[5]:<8} | {row[6]:<8} | {row[7]}")
    print("=" * 105 + "\n")
    
    return metrics


# === SECTION 5: STATIC VISUALIZATIONS ===

def plot_static_results(results, metrics):
    scenarios = list(SCENARIOS.keys())
    x = np.arange(len(scenarios))
    width = 0.2
    
    # 1. Wait Times (results_wait_time.png)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, lane in enumerate(LANES):
        avg_waits = [metrics[s][lane]["avg_wait"] for s in scenarios]
        ax.bar(x + i*width - width*1.5, avg_waits, width, label=lane)
    
    ax.set_ylabel("Average Wait Time (s)")
    ax.set_title("Average Wait Time per Lane across Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    fig.savefig("results_wait_time.png")
    plt.close(fig)
    
    # 2. Queue lengths over time (results_queue.png)
    fig, axs = plt.subplots(len(scenarios), 1, figsize=(10, 15), sharex=True)
    if len(scenarios) == 1: axs = [axs]
    
    for i, (s_name, model) in enumerate(results.items()):
        times = [snap["time"] for snap in model.snapshot_60s]
        for lane in LANES:
            qs = [snap["queues"][lane] for snap in model.snapshot_60s]
            axs[i].plot(times, qs, label=lane)
        axs[i].set_title(s_name)
        axs[i].set_ylabel("Queue Len")
        if i == 0: axs[i].legend(loc="upper left", bbox_to_anchor=(1, 1))
        
    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("results_queue.png")
    plt.close(fig)
    
    # 3. Throughput and Efficiency (results_throughput.png)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    throughputs = [metrics[s]["TOTAL"]["throughput"] for s in scenarios]
    efficiencies = [metrics[s]["TOTAL"]["efficiency"] for s in scenarios]
    
    ax1.bar(x - 0.2, throughputs, 0.4, color='b', label='Throughput (Vehicles)')
    ax2.bar(x + 0.2, efficiencies, 0.4, color='orange', label='Efficiency (%)')
    
    ax1.set_ylabel("Total Throughput (Vehicles)", color='b')
    ax2.set_ylabel("Efficiency (%)", color='orange')
    ax1.set_title("Total Throughput and Efficiency across Scenarios")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    fig.savefig("results_throughput.png")
    plt.close(fig)
    print("Static visualisations saved as PNGs.")


# === SECTION 6: ANIMATION ===

def render_animation(results, scenario_name="Rush Hour"):
    model = results[scenario_name]
    snapshots = model.snapshots
    
    # Filter to requested animation duration
    anim_data = [s for s in snapshots if s["time"] <= ANIMATION_DURATION]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#222222")
    
    # Draw intersection static elements
    ax.plot([-100, -10], [10, 10], 'w-', lw=2)
    ax.plot([-100, -10], [-10, -10], 'w-', lw=2)
    ax.plot([10, 100], [10, 10], 'w-', lw=2)
    ax.plot([10, 100], [-10, -10], 'w-', lw=2)
    
    ax.plot([-10, -10], [10, 100], 'w-', lw=2)
    ax.plot([10, 10], [10, 100], 'w-', lw=2)
    ax.plot([-10, -10], [-100, -10], 'w-', lw=2)
    ax.plot([10, 10], [-100, -10], 'w-', lw=2)
    
    ax.text(0, 80, "North", color="w", ha="center")
    ax.text(0, -80, "South", color="w", ha="center")
    ax.text(80, 0, "East", color="w", va="center")
    ax.text(-80, 0, "West", color="w", va="center")
    
    # Dynamic elements (lights and queues)
    def create_light(): return plt.Circle((0, 0), radius=4, color="gray", zorder=3)
    
    lights = {
        "N": plt.Circle((-5, 15), radius=4, color="r"),
        "S": plt.Circle((5, -15), radius=4, color="r"),
        "E": plt.Circle((15, 5), radius=4, color="r"),
        "W": plt.Circle((-15, -5), radius=4, color="r")
    }
    
    for l in lights.values():
        ax.add_patch(l)
        
    bars = {
        "N": ax.bar(5, 0, width=8, bottom=25, color='cyan', align='center'),
        "S": ax.bar(-5, 0, width=8, bottom=-25, color='cyan', align='center'),
        "E": ax.barh(-5, 0, height=8, left=25, color='cyan', align='center'),
        "W": ax.barh(5, 0, height=8, left=-25, color='cyan', align='center')
    }
    
    time_text = ax.text(-90, 80, "", color="white", fontsize=12)
    
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Simulation: {scenario_name}", color="white")
    
    color_map = {"R": "red", "Y": "yellow", "G": "lime"}
    
    def update(frame_idx):
        if frame_idx >= len(anim_data):
            return
            
        data = anim_data[frame_idx]
        time_text.set_text(f"Time: {data['time']:.0f}s")
        
        # Update lights
        for lane in LANES:
            lights[lane].set_color(color_map[data['lights'][lane]])
            
        # Update queue bars
        q_scale = 1.5  # visual scaling
        # North bar grows up
        bars['N'][-1].set_height(data['queues']['N'] * q_scale)
        # South bar grows down
        bars['S'][-1].set_height(-data['queues']['S'] * q_scale)
        # East bar grows right
        bars['E'][-1].set_width(data['queues']['E'] * q_scale)
        # West bar grows left
        bars['W'][-1].set_width(-data['queues']['W'] * q_scale)
        
        return list(lights.values()) + [time_text] + [bars[l][-1] for l in LANES]
            
    print(f"Starting animation for {scenario_name} (first {ANIMATION_DURATION}s)...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(anim_data), 
        interval=ANIMATION_SPEED_MS, blit=False, repeat=False
    )
    plt.show()


# === SECTION 7: MAIN ===

if __name__ == '__main__':
    # 1. Run SimPy simulation for all scenarios
    results = run_simulation()
    
    # 2. Extract metrics and output CSV / Table
    metrics = generate_metrics(results)
    
    # 3. Generate static plots
    plot_static_results(results, metrics)
    
    # 4. Run animation for default scenario
    # Default is Rush Hour, as requested
    render_animation(results, scenario_name="Rush Hour")
