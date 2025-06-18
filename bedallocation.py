import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, t
import numpy as np
import math

class QueueSimulation:
    def __init__(self, bed_capacities,
                 arrival_rates,
                 leaving_rates,
                 urgency_points,
                 p_mtx,
                 max_patients=10000):
        self.bed_capacities = bed_capacities
        self.arrival_rate = np.sum(arrival_rates)
        self.arrival_probs = arrival_rates / self.arrival_rate
        self.leaving_rates = leaving_rates
        self.urgency_points = urgency_points
        self.max_patients = max_patients
        self.p_mtx = p_mtx
        
        self.curr_time = 0
        self.served_patients = 0
        self.blocked_patients = 0
        self.event_queue = []
    
    def sample_arrival(self):
        return np.random.exponential(1 / self.arrival_rate)
    
    def sample_departure(self, patient_type):
        return np.random.exponential(self.leaving_rates[patient_type])
    
    def schedule_event(self, event):
        self.event_queue.append(event)
        self.event_queue.sort(key=lambda x: x[1])
    
    def process_event(self, event):
        event_type, event_time, patient_type = event
        self.curr_time = event_time
        
        if event_type == 'a':  # arrival
            if self.bed_capacities[patient_type] > 0:
                self.bed_capacities[patient_type] -= 1
                self.served_patients += 1
                self.schedule_event(('d', self.curr_time + self.sample_departure(patient_type), patient_type))
            else:
                # choose another ward based on transition probabilities
                next_ward = np.random.choice(len(self.p_mtx[patient_type]), p=self.p_mtx[patient_type])
                if self.bed_capacities[next_ward] > 0:
                    self.bed_capacities[next_ward] -= 1
                    self.served_patients += 1
                    self.schedule_event(('d', self.curr_time + self.sample_departure(patient_type), next_ward))
                else:
                    self.blocked_patients += 1
            patient_type = np.random.choice(len(self.arrival_probs), p=self.arrival_probs)
            self.schedule_event(('a', self.curr_time + self.sample_arrival(), patient_type))
        
        elif event_type == 'd':  # departure
            self.bed_capacities[patient_type] += 1
    
    def run(self):
        while self.served_patients + self.blocked_patients < self.max_patients:
            if not self.event_queue:
                patient_type = np.random.choice(len(self.arrival_probs), p=self.arrival_probs)
                self.schedule_event(('a', self.sample_arrival(), patient_type))
            event = self.event_queue.pop(0)
            self.process_event(event)
        
        return {
            'served': self.served_patients,
            'blocked': self.blocked_patients,
            'blocking_probability': self.blocked_patients / (self.served_patients + self.blocked_patients),
            'current_time': self.curr_time
        }
    
def main():
    base_bed_capacities = np.array([55, 40, 30, 20, 20])
    base_arrival_rates = np.array([14.5, 11, 8, 6.5, 5])
    base_leaving_rates = np.array([2.9, 4, 4.5, 1.4, 3.9])
    base_urgency_points = np.array([7, 5, 2, 10, 5])
    base_p_mtx = np.array([
        [0.00, 0.05, 0.10, 0.05, 0.80],
        [0.20, 0.00, 0.50, 0.15, 0.15],
        [0.30, 0.20, 0.00, 0.20, 0.30],
        [0.35, 0.30, 0.05, 0.00, 0.30],
        [0.20, 0.10, 0.60, 0.10, 0.00]
    ])
    
    
    sim = QueueSimulation(
        bed_capacities=base_bed_capacities.copy(),
        arrival_rates=base_arrival_rates,
        leaving_rates=base_leaving_rates,
        urgency_points=base_urgency_points,
        p_mtx=base_p_mtx,
        max_patients=10000
    )
    results = sim.run()
    
    print(f"Total served patients: {results['served']}")
    print(f"Total blocked patients: {results['blocked']}")
    print(f"Blocking probability: {results['blocking_probability']:.4f}")
    print(f"Simulation time: {results['current_time']:.2f}")

'''
    beds_in_f = 1000
    bed_capacities=np.array([55, 40, 30, 20, 20, beds_in_f])
    arrival_rates=np.array([14.5, 11, 8, 6.5, 5, 13])
    leaving_rates=np.array([2.9, 4, 4.5, 1.4, 3.9, 2.2])
    urgency_points=np.array([7, 5, 2, 10, 5, 0])
    p_mtx = np.array([
                [0.00, 0.05, 0.10, 0.05, 0.80, 0.00],
                [0.20, 0.00, 0.50, 0.15, 0.15, 0.00],
                [0.30, 0.20, 0.00, 0.20, 0.30, 0.00],
                [0.35, 0.30, 0.05, 0.00, 0.30, 0.00],
                [0.20, 0.10, 0.60, 0.10, 0.00, 0.00],
                [0.20, 0.20, 0.20, 0.20, 0.20, 0.00]
                ])
    
    sim_f = QueueSimulation(bed_capacities, arrival_rates, leaving_rates, urgency_points, p_mtx)
    results_f = sim_f.run()
    print(f"Total served patients with F ward: {results_f['served']}")
    print(f"Total blocked patients with F ward: {results_f['blocked']}")
    print(f"Blocking probability with F ward: {results_f['blocking_probability']:.4f}")
    print(f"Simulation time with F ward: {results_f['current_time']:.2f}")
'''

if __name__ == "__main__":
    main()