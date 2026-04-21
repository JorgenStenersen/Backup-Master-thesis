import numpy as np

def summarize_market_inputs(
    prices_cm_up,
    prices_cm_down,
    prices_dam,
    prices_eam_up,
    prices_eam_down,
    forecasted_power
):
    """
    Computes and prints summary statistics (mean, std, min, max)
    for six input lists with equal length.
    """

    data = {
        "CM Up Prices": prices_cm_up,
        "CM Down Prices": prices_cm_down,
        "DAM Prices": prices_dam,
        "EAM Up Prices": prices_eam_up,
        "EAM Down Prices": prices_eam_down,
        "Forecasted Power": forecasted_power,
    }

    print("\nSummary Statistics (n = {})".format(len(forecasted_power)))
    print("-" * 72)
    print(f"{'Variable':<20} {'Mean':>12} {'Std. Dev.':>12} {'Min':>12} {'Max':>12}")
    print("-" * 72)

    for name, values in data.items():
        values = np.asarray(values, dtype=float)

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # sample standard deviation
        min_val = np.min(values)
        max_val = np.max(values)

        print(
            f"{name:<20} "
            f"{mean_val:>12.3f} "
            f"{std_val:>12.3f} "
            f"{min_val:>12.3f} "
            f"{max_val:>12.3f}"
        )

    print("-" * 72)

