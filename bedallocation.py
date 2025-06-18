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
                 max_patients=10000,
                 warmup_period=1000,
                 iterations=50):
        self.bed_capacities = bed_capacities
        self.arrival_rate = np.sum(arrival_rates)
        self.arrival_probs = arrival_rates / self.arrival_rate
        self.leaving_rates = leaving_rates
        self.urgency_points = urgency_points
        self.p_mtx = p_mtx
        self.max_patients = max_patients
        self.warmup_period = warmup_period
        self.iterations = iterations
        
        self.curr_time = 0
        self.served_patients = 0
        self.blocked_patients = 0
        self.event_queue = []

        # extra statistics
        self.arrivals = np.zeros(len(arrival_rates))
        self.transfers = np.zeros(len(arrival_rates))
        self.blocks = np.zeros(len(arrival_rates))
    
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
            self.arrivals[patient_type] += 1
            if self.bed_capacities[patient_type] > 0: # bed available in the correct ward
                self.bed_capacities[patient_type] -= 1
                self.served_patients += 1
                self.schedule_event(('d', self.curr_time + self.sample_departure(patient_type), patient_type))
            else:
                # choose another ward based on transition probabilities
                next_ward = np.random.choice(len(self.p_mtx[patient_type]), p=self.p_mtx[patient_type])
                self.transfers[patient_type] += 1 # TRY transfering to another ward
                if self.bed_capacities[next_ward] > 0:
                    self.bed_capacities[next_ward] -= 1
                    self.served_patients += 1
                    self.schedule_event(('d', self.curr_time + self.sample_departure(patient_type), next_ward))
                else:
                    self.blocked_patients += 1
                    self.blocks[patient_type] += 1
            patient_type = np.random.choice(len(self.arrival_probs), p=self.arrival_probs)
            self.schedule_event(('a', self.curr_time + self.sample_arrival(), patient_type))
        
        elif event_type == 'd':  # departure
            self.bed_capacities[patient_type] += 1
    
    def run(self):
        # warmup
        for _ in range(self.warmup_period):
            if not self.event_queue:
                patient_type = np.random.choice(len(self.arrival_probs), p=self.arrival_probs)
                self.schedule_event(('a', self.sample_arrival(), patient_type))
            event = self.event_queue.pop(0)
            self.process_event(event)

        loss_scores = []
        for _ in range(self.iterations):
            # print progress
            print(f"Iteration {_ + 1}/{self.iterations}")
            # record current statistics
            curr_blocks = self.blocks.copy()
            curr_transfers = self.transfers.copy()
            arrived_patients = 0
            while arrived_patients < self.max_patients:
                event = self.event_queue.pop(0)
                if event[0] == 'a':
                    arrived_patients += 1
                self.process_event(event)
            curr_blocks = self.blocks - curr_blocks
            curr_transfers = self.transfers - curr_transfers
            # compute loss score
            loss_scores.append((np.sum(curr_blocks * self.urgency_points) + np.sum(curr_transfers * self.urgency_points)) / self.max_patients)
        loss_scores = np.array(loss_scores)
        
        return {
            'all served': self.served_patients,
            'all blocked': self.blocked_patients,
            'all blocking_probability': self.blocked_patients / (self.served_patients + self.blocked_patients),
            'loss_scores': loss_scores
        }
    
def main():
    n_iter = 50
    n_patients = 10000
    warmup_period = 1000
    beds_in_f = 35
    beds_to_take = np.array([7, 7, 7, 7, 7, 0])
    assert np.sum(beds_to_take) == beds_in_f, "Total beds to take has to equal beds in F ward"
    bed_capacities=np.array([55, 40, 30, 20, 20, beds_in_f])
    bed_capacities = bed_capacities - beds_to_take
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
    
    sim = QueueSimulation(bed_capacities, arrival_rates, leaving_rates, urgency_points, p_mtx, n_patients, warmup_period, n_iter)
    results = sim.run()
    loss_scores = results['loss_scores']
    avg_loss_score = np.mean(loss_scores)
    std_loss_score = np.std(loss_scores)
    ci = 1.96 * std_loss_score / np.sqrt(n_iter)  # 95% CI for the mean
    print(f"Average loss score: {avg_loss_score:.4f} ± {ci:.4f} (95% CI)")
    print(f"Blocking probability: {results['all blocking_probability']:.4f}")

if __name__ == "__main__":
    main()