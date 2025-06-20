import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, t
import numpy as np
import math
import heapq

class QueueSimulation:
    def __init__(self, bed_capacities,
                 arrival_rates,
                 leaving_rates,
                 urgency_points,
                 p_mtx,
                 max_patients=10000,
                 warmup_period=1000,
                 iterations=100):
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
        self.event_queue = []

        # extra statistics
        self.arrivals = np.zeros(len(arrival_rates))
        self.transfers = np.zeros(len(arrival_rates))
        self.blocks = np.zeros(len(arrival_rates))
        self.secondary_ward_blocks = np.zeros(len(arrival_rates))
    
    def sample_arrival(self):
        return np.random.exponential(1 / self.arrival_rate)
    
    def sample_departure(self, patient_type):
        return np.random.exponential(self.leaving_rates[patient_type])
    
    def schedule_event(self, event):
        # Assuming event[1] is the time or priority
        heapq.heappush(self.event_queue, event)
    
    def process_event(self, event):
        event_time, event_type, patient_type = event
        self.curr_time = event_time
        
        if event_type == 'a':  # arrival
            self.arrivals[patient_type] += 1
            if self.bed_capacities[patient_type] > 0: # bed available in the correct ward
                self.bed_capacities[patient_type] -= 1
                self.schedule_event((self.curr_time + self.sample_departure(patient_type), 'd', patient_type))
            else:
                # choose another ward based on transition probabilities
                next_ward = np.random.choice(len(self.p_mtx[patient_type]), p=self.p_mtx[patient_type])
                self.transfers[patient_type] += 1 # TRY transfering to another ward
                if self.bed_capacities[next_ward] > 0:
                    self.bed_capacities[next_ward] -= 1
                    self.schedule_event((self.curr_time + self.sample_departure(patient_type), 'd', next_ward))
                else:
                    self.blocks[patient_type] += 1
                    self.secondary_ward_blocks[next_ward] += 1
            patient_type = np.random.choice(len(self.arrival_probs), p=self.arrival_probs)
            self.schedule_event((self.curr_time + self.sample_arrival(), 'a', patient_type))
        
        elif event_type == 'd':  # departure
            self.bed_capacities[patient_type] += 1
    
    def run(self):
        # warmup
        for _ in range(self.warmup_period):
            if not self.event_queue:
                patient_type = np.random.choice(len(self.arrival_probs), p=self.arrival_probs)
                self.schedule_event((self.sample_arrival(), 'a', patient_type))
            event = heapq.heappop(self.event_queue)
            self.process_event(event)

        loss_scores = []
        arrival_results = []
        blocks_results = []
        secondary_ward_block_results = []
        transfers_results = []
        for _ in range(self.iterations):
            # print progress
            print(f"Iteration {_ + 1}/{self.iterations}")
            # record current statistics
            curr_arrivals = self.arrivals.copy()
            curr_blocks = self.blocks.copy()
            curr_secondary_ward_blocks = self.secondary_ward_blocks.copy()
            curr_transfers = self.transfers.copy()
            arrived_patients = 0
            while arrived_patients < self.max_patients:
                event = heapq.heappop(self.event_queue)
                if event[1] == 'a':
                    arrived_patients += 1
                self.process_event(event)
            curr_arrivals = self.arrivals - curr_arrivals
            curr_blocks = self.blocks - curr_blocks
            curr_secondary_ward_blocks = self.secondary_ward_blocks - curr_secondary_ward_blocks
            curr_transfers = self.transfers - curr_transfers
            arrival_results.append(curr_arrivals)
            blocks_results.append(curr_blocks)
            secondary_ward_block_results.append(curr_secondary_ward_blocks)
            transfers_results.append(curr_transfers)
            # compute loss score
            loss_scores.append((np.sum(curr_blocks * self.urgency_points) + np.sum(curr_transfers * self.urgency_points)) / self.max_patients)
        loss_scores = np.array(loss_scores)
        
        return {
            'arrivals': np.array(arrival_results),
            'transfers': np.array(transfers_results),
            'blocks': np.array(blocks_results),
            'secondary_ward_blocks': np.array(secondary_ward_block_results),
            'loss_scores': loss_scores
        }

def round_preserve_sum(x):
    floored = np.floor(x)
    remainder = x - floored
    num_to_round_up = int(round(np.sum(x) - np.sum(floored)))
    indices = np.argsort(-remainder)  # sort descending
    result = floored.copy()
    result[indices[:num_to_round_up]] += 1
    return result.astype(int)

def main():
    n_iter = 30
    n_patients = 10000
    warmup_period = 1000
    beds_in_f = 34
    #beds_to_take = np.append(round_preserve_sum(bed_capacities[:-1] / np.sum(bed_capacities[:-1]) * beds_in_f), 0)
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
    scores_optim = []
    optim_iter = 30
    beds_to_take = np.array([7, 7, 7, 6, 7, 0])
    initial_bed_capacities=np.array([55, 40, 30, 20, 20, beds_in_f])
    best_config = beds_to_take.copy()
    min_score = np.inf
    for i in range(optim_iter):
        assert np.sum(beds_to_take) == beds_in_f, "Total beds to take has to equal beds in F ward"
        bed_capacities = initial_bed_capacities - beds_to_take
        sim = QueueSimulation(bed_capacities, arrival_rates, leaving_rates, urgency_points, p_mtx, n_patients, warmup_period, n_iter)
        results = sim.run()
        scores = np.sum(results['secondary_ward_blocks'], axis=0) * 1 + np.sum(results['transfers'], axis=0) * urgency_points
        overall_score = np.sum(scores)/(n_patients*n_iter)
        if overall_score < min_score:
            min_score = overall_score
            best_config = beds_to_take.copy()
        print(f"Optimization iteration: {i+1}/{optim_iter}")
        print(f"Current configuration: {beds_to_take}")
        print(f"Current scores: {scores}")
        print(f"Currect score: {overall_score}")
        scores_optim.append(overall_score)
        can_take_indices = np.where(initial_bed_capacities[:-1] - beds_to_take[:-1] > 0)[0]
        can_add_indices = np.where(beds_to_take > 0)[0]
        min_score_idx = np.argmin(scores[:-1][can_take_indices])
        max_score_idx = np.argmax(scores[:-1][can_add_indices])

        beds_to_take[can_take_indices[min_score_idx]] += 1
        beds_to_take[can_add_indices[max_score_idx]] -= 1

    print(f"Best configuration: {best_config}")

    plt.plot(scores_optim)
    plt.title("Optimization Scores over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()

    loss_scores = results['loss_scores']
    avg_loss_score = np.mean(loss_scores)
    std_loss_score = np.std(loss_scores)
    ci = 1.96 * std_loss_score / np.sqrt(n_iter)  # 95% CI for the mean
    print(f"Average loss score: {avg_loss_score:.4f} ± {ci:.4f} (95% CI)")
    all_arrivals = np.sum(results['arrivals'])
    all_blocks = np.sum(results['blocks'])
    print(f"Blocking probability (all): {all_blocks / all_arrivals:.4f} (total arrivals: {all_arrivals})")
    config_name = "const_beds_10"
    np.savez(f"results_{config_name}.npz", **results)

if __name__ == "__main__":
    main()