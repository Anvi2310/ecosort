# Reproducible experiments

Experiment scripts will consume the documented train/val/test folder layout. Record random seeds, dataset version/source, label mapping, checkpoint provenance, and package versions with each run. Keep test data untouched until model selection is complete. Report per-class CNN metrics and confusion matrix; report measured ANFIS regression metrics only against an explicitly documented target. Synthetic examples are for algorithm demonstration and must be labelled synthetic in all results.
