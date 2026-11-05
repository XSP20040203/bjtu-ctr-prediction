"""Use the identical training runner while keeping recent-window outputs separate."""
import run_id_regularization as experiment


if __name__ == "__main__":
    experiment.OUT = experiment.HERE / "results_recent_window"
    experiment.main()
