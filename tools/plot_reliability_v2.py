"""Descriptive repeat-one plots; no tuning or classifier refitting."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, confusion_matrix
from portable_release import discover_workspace, PROJECT


def main():
    root = discover_workspace()
    result = root / PROJECT / 'results/research_extension_v2'
    out = root / PROJECT / 'paper/research_extension_v2'
    out.mkdir(parents=True, exist_ok=True)
    thresholds = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, name in enumerate(['CALIBRATION', 'ADAPTATION']):
        predictions = pd.read_csv(result / (name + '_PREDICTIONS.csv'))
        for model, group in predictions[predictions.repeat == 1].groupby('model_id'):
            fpr, tpr, _ = roc_curve(group.y_true, group.probability)
            axes[row, 0].plot(fpr, tpr, label=model)
            observed, predicted = calibration_curve(group.y_true, group.probability, n_bins=5, strategy='quantile')
            axes[row, 1].plot(predicted, observed, marker='o', label=model)
            tn, fp, fn, tp = confusion_matrix(group.y_true, group.probability >= .5, labels=[0, 1]).ravel()
            thresholds.append(dict(cohort=name, model_id=model, repeat=1, threshold=.5,
                                   tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                                   sensitivity=tp/(tp+fn), specificity=tn/(tn+fp)))
        for col in [0, 1]:
            axes[row, col].plot([0, 1], [0, 1], '--', color='gray')
            axes[row, col].legend(fontsize=7)
            axes[row, col].set_xlim(0, 1); axes[row, col].set_ylim(0, 1)
        axes[row, 0].set(xlabel='False-positive rate', ylabel='True-positive rate', title=name + ': repeat-one ROC')
        axes[row, 1].set(xlabel='Mean predicted probability', ylabel='Observed event fraction', title=name + ': descriptive calibration')
    fig.suptitle('Retrospective follow-up: source calibration and target adaptation')
    fig.tight_layout()
    fig.savefig(out / 'ROC_CALIBRATION.png', dpi=180)
    fig.savefig(out / 'ROC_CALIBRATION.pdf')
    plt.close(fig)
    pd.DataFrame(thresholds).to_csv(out / 'THRESHOLD_05_DIAGNOSTICS.csv', index=False)


if __name__ == '__main__':
    main()
