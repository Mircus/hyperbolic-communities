import matplotlib.pyplot as plt

def plot_frequency_hist(freqs):
    plt.figure(); plt.hist(freqs, bins=30); plt.xlabel('frequency'); plt.ylabel('count'); plt.tight_layout(); return plt.gcf()
