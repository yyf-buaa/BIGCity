from typing import Dict, List
import matplotlib.pyplot as plt
import os
import random
import csv

from config.logging_config import cur_log_dir


def save_loss_image(losses: Dict[str, List[float]]):
    loss_colors = ["blue", "red", "green", "purple"]
    loss_styles = ["-", "--", ":", "-."]
    
    for loss_name, loss in losses.items():
        name = f"{loss_name}_loss"
        color = random.choice(loss_colors)
        style = random.choice(loss_styles)
        
        plt.figure(figsize=(12, 8))
        plt.plot(loss, label=name, color=color, linestyle=style, linewidth=1.5)
        plt.xlabel("Iterations (batches)", fontsize=12)
        plt.ylabel("Loss Value", fontsize=12)
        plt.title(f"{name} during Training", fontsize=14)
        plt.legend()
        plt.grid(True)
        
        # Save each plot
        plot_filename = os.path.join(cur_log_dir, f"{name}.png")
        plt.savefig(plot_filename)
        plt.close()

def save_losses_to_csv(losses: Dict[str, List[float]]):
    max_length = max(len(values) for values in losses.values())
    output_file = os.path.join(cur_log_dir, "losses.csv")
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["batch_idx"] + list(losses.keys()))
        
        for i in range(max_length):
            row = [i + 1]
            for loss_name in losses:
                row.append(losses[loss_name][i] if i < len(losses[loss_name]) else "")
            writer.writerow(row)