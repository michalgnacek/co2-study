# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 02:23:12 2023

@author: m
"""

import matplotlib.pyplot as plt

def plot_eyetracking_filter(raw_signal, filtered_signal, participant_id, condition):
    # Plot the original and filtered signals
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(raw_signal, 'b-', label='Raw Signal')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Pupil size (mm)')
    ax[0].legend()

    ax[1].plot(filtered_signal, 'g-', linewidth=2, label='Filtered Signal')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Pupil size (mm)')
    ax[1].legend()
    
    plt.title('Pupil size for ' + condition + ' condition. Participant: ' + participant_id)
    plt.tight_layout()
    plt.show()