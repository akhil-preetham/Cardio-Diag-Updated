"""Build a result-driven IEEEtran report; never fit predictive models."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/research_extension_v1"
ZAS = RESULTS / "zas"
TRANSFER = RESULTS / "transport"
FIGURES = OUT / "figures"
NAMES = {
    "historical_v3_balanced": "Historical balanced V3 (5)",
    "l2_v3_ordinary_historical_calibration": "Ordinary L2, V3 fields (5)",
    "l2_risk_symptoms17_historical_calibration": "L2 risk + symptoms (17)",
    "l2_no_symptoms28_historical_calibration": "L2 without symptoms (28)",
    "l2_full38_historical_calibration": "Equal-feature L2 (38)",
    "historical_selection": "Historical selection (30 candidates)",
    "same_components_uncalibrated_ensemble": "Corrected raw-component ensemble",
    "selection_independent_calibrated_ensemble": "Corrected calibrated ensemble",
}
FULL = "l2_full38_historical_calibration"
CAL = "selection_independent_calibrated_ensemble"
RAW = "same_components_uncalibrated_ensemble"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def fig(name, caption):
    return f"![{caption}](figures/{name}.pdf)"


def format_value(x, digits=4):
    return f"{float(x):.{digits}f}"


def plot_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.2,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})


def savefig(fig_obj, name):
    fig_obj.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig_obj.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig_obj)


def calibration_points(y, p):
    edges = np.linspace(0, 1, 6)
    bins = np.clip(np.digitize(p, edges) - 1, 0, 4)
    points = []
    for k in range(5):
        keep = bins == k
        if keep.any():
            points.append((p[keep].mean(), y[keep].mean(), keep.sum()))
    return np.array(points)


def make_figures(repeats, predictions, target, stress):
    plot_style()
    ids = list(NAMES)
    f, ax = plt.subplots(figsize=(6.8, 2.9))
    for i, model in enumerate(ids):
        values = repeats.loc[repeats.model_id == model, "roc_auc"].to_numpy()
        ax.scatter(values, [i] * len(values), marker="x", s=20, color="0.5", zorder=2)
        ax.plot([values.min(), values.max()], [i, i], color="0.55", lw=1)
        ax.scatter([values.mean()], [i], color="black", s=22, zorder=3)
    ax.set_yticks(range(len(ids)), [NAMES[x].replace("Historical ", "Hist. ").replace("Corrected ", "") for x in ids], fontsize=7.2)
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("Complete-repeat OOF ROC-AUC")
    ax.grid(axis="x", color="0.9", lw=.6)
    f.tight_layout(); savefig(f, "source_auc")

    f, axes = plt.subplots(1, 2, figsize=(6.8, 2.65))
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=.9)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed event fraction")
        ax.grid(color="0.92", lw=.5)
    for model, marker, label in [(RAW, "o", "Same-component raw"), (CAL, "s", "Independent calibration")]:
        g = predictions[(predictions.model_id == model) & (predictions.repeat == 1)]
        points = calibration_points(g.y_true.to_numpy(), g.probability.to_numpy())
        axes[0].plot(points[:, 0], points[:, 1], marker=marker, color="black" if model == CAL else ".5", label=label, lw=1)
    for model, marker, label in [("raw_full_source_fit", "o", "Raw source fit"), ("source_calibrated_ensemble", "s", "Source calibrated ensemble")]:
        points = calibration_points(target.y_base_gt50.to_numpy(), target[model].to_numpy())
        axes[1].plot(points[:, 0], points[:, 1], marker=marker, color="black" if marker == "s" else ".5", label=label, lw=1)
    axes[0].set_title("ZAS: corrected pool, repeat 1 only")
    axes[1].set_title("AngioCAD: five-core transport, 377 cases")
    for ax in axes: ax.legend(loc="lower right", frameon=False)
    f.tight_layout(); savefig(f, "calibration")

    f, ax = plt.subplots(figsize=(3.35, 2.3))
    stats = stress.groupby("mask_rate").roc_auc.agg(["mean", "min", "max"])
    ax.errorbar(stats.index * 100, stats["mean"], yerr=np.vstack([stats["mean"] - stats["min"], stats["max"] - stats["mean"]]), color="black", marker="o", capsize=3)
    ax.set(xlim=(-2, 52), ylim=(0, 1), xlabel="Independently masked input cells (%)", ylabel="Complete-repeat ROC-AUC")
    ax.set_xticks([0, 10, 25, 50]); ax.grid(color=".9", lw=.6)
    f.tight_layout(); savefig(f, "missingness")

    f, ax = plt.subplots(figsize=(6.8, 1.72))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [(0.02,.57,.23,.32,"Outer-training rows T\nThree calibration blocks"),(.38,.57,.24,.32,"Selection on D only\n6 groups x 5 families\nInner CV for all tuning"),(.75,.57,.23,.32,"Fit base model on D\nFit sigmoid on held C"),(.75,.07,.23,.32,"Keep each base + sigmoid\nAverage on outer test"),(.38,.07,.24,.32,"Identical raw\ncomponent ensemble\nCalibration contrast"),(.02,.07,.23,.32,"Separate five-core model\nFrozen AngioCAD scoring")]
    for x,y,w,h,label in boxes:
        ax.add_patch(plt.Rectangle((x,y), w,h, facecolor=".96", edgecolor="black", lw=.8))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=8)
    for a,b in [((.25,.73),(.38,.73)),((.62,.73),(.75,.73)),((.87,.57),(.87,.39)),((.75,.23),(.62,.23))]:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle":"->","lw":.9})
    ax.text(.135,.44,"Separate study track",ha="center",fontsize=7.5)
    f.tight_layout(pad=.2); savefig(f,"workflow")


def tex_escape(text):
    replacements = {"\\":r"\textbackslash{}", "&":r"\&", "%":r"\%", "$":r"\$", "#":r"\#", "_":r"\_", "{":r"\{", "}":r"\}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}"}
    return "".join(replacements.get(c,c) for c in text)


def to_latex(text):
    lines = text.splitlines()
    title = lines[0].lstrip("# ")
    preamble = [r"\documentclass[conference]{IEEEtran}", r"\usepackage[T1]{fontenc}", r"\usepackage{graphicx,booktabs,array,url}",
                r"\usepackage{amsmath}", r"\usepackage{microtype}",
                r"\hyphenation{AngioCAD CardioDiag}",
                r"\title{" + tex_escape(title) + "}",
                r"\author{\IEEEauthorblockN{Erukudinnela Akhil Preetham, Chunduri Hari Krishna, G Vennela}\IEEEauthorblockA{Department of Computer Science and Engineering\\Prasad V. Potluri Siddhartha Institute of Technology (PVPSIT)\\Vijayawada, Andhra Pradesh, India\\Project guide: K. Naga Gopi}}",
                r"\begin{document}", r"\pagestyle{plain}", r"\maketitle", r"\thispagestyle{plain}"]
    content = []
    i, special, refs = 1, None, False
    pending_caption = None
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith("## "):
            if special:
                content.append(r"\end{"+special+"}"); special=None
            value=line[3:]
            if value in ("Abstract", "Index Terms"):
                special="abstract" if value == "Abstract" else "IEEEkeywords"
                content.append(r"\begin{"+special+"}")
            elif value == "References":
                refs=True; content.extend([r"\newpage",r"\begin{thebibliography}{9}"])
            elif re.match(r"^[IVX]+\. ",value):
                content.append(r"\section{"+tex_escape(re.sub(r"^[IVX]+\. ","",value))+"}")
            else:
                content.append(r"\section*{"+tex_escape(value)+"}")
            i+=1; continue
        if line.startswith("### "):
            content.append(r"\subsection{"+tex_escape(re.sub(r"^[A-Z]\. ","",line[4:]))+"}"); i+=1;continue
        image_match=re.match(r"!\[(.*?)\]\((.*?)\)",line)
        if image_match:
            caption,path=image_match.groups(); wide=path.endswith(("calibration.pdf","workflow.pdf","source_auc.pdf"))
            env="figure*" if wide else "figure"
            content.extend([r"\begin{"+env+r"}[!t]",r"\centering",r"\includegraphics[width="+(r"\textwidth" if wide else r"\columnwidth")+"]{"+path+"}",r"\caption{"+tex_escape(caption)+"}",r"\end{"+env+"}"])
            i+=1; continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                row=[v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?",v) for v in row): rows.append(row)
                i+=1
            n=len(rows[0]); first=.42 if n>=4 else .49
            fractions=[first]+[(1-first)/(n-1)]*(n-1)
            spec="".join(r"p{\dimexpr "+f"{f:.4f}"+r"\columnwidth-2\tabcolsep\relax}" for f in fractions)
            content.extend([r"\begin{center}\begin{minipage}{\columnwidth}\centering\footnotesize"])
            if pending_caption:
                content.append(tex_escape(pending_caption)+r"\par\vspace{3pt}")
                pending_caption=None
            content.extend([r"\setlength{\tabcolsep}{2.5pt}",r"\begin{tabular}{"+spec+"}",r"\toprule"])
            for j,row in enumerate(rows):
                content.append(" & ".join(tex_escape(v) for v in row)+r" \\")
                if j==0: content.append(r"\midrule")
            content.extend([r"\bottomrule",r"\end{tabular}",r"\end{minipage}\end{center}"])
            continue
        para=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","![")):
            para.append(lines[i].strip()); i+=1
        value=" ".join(para)
        if re.match(r"^Table [IVX]+\.",value):
            pending_caption=value
            continue
        if refs:
            match=re.match(r"\[(\d+)\]\s*(.*)",value)
            if not match: raise ValueError("Unparsed bibliography record")
            content.append(r"\bibitem{r"+match[1]+"} "+tex_escape(match[2]))
        else:
            escaped=tex_escape(value)
            escaped=re.sub(r"\[(\d+)\]",lambda m:r"\cite{r"+m[1]+"}",escaped)
            content.append(escaped+"\n")
    if special: content.append(r"\end{"+special+"}")
    if refs: content.append(r"\end{thebibliography}")
    content.append(r"\end{document}")
    return "\n".join(preamble+content)+"\n"


def main():
    for required in [ZAS/"MANIFEST.json", TRANSFER/"TRANSPORT_MANIFEST.json"]:
        if not required.exists(): raise FileNotFoundError(f"Experiment incomplete: {required}")
    FIGURES.mkdir(parents=True,exist_ok=True)
    repeats=pd.read_csv(ZAS/"REPEAT_METRICS.csv")
    predictions=pd.read_csv(ZAS/"OOF_PREDICTIONS.csv")
    target=pd.read_csv(TRANSFER/"TARGET_PREDICTIONS.csv")
    target_metrics=pd.read_csv(TRANSFER/"TARGET_METRICS.csv")
    stress=pd.read_csv(TRANSFER/"SOURCE_MCAR_STRESS_REPEAT_METRICS.csv")
    means=repeats.groupby("model_id").mean(numeric_only=True)
    selections=pd.read_csv(ZAS/"SELECTIONS.csv")
    manifest=json.loads((ZAS/"MANIFEST.json").read_text(encoding="utf-8"))
    warning_rows=json.loads((ZAS/"WARNINGS.json").read_text(encoding="utf-8"))
    q=216/303
    constant={"reference":"posthoc source-prevalence constant; no fit or target adaptation", "probability":q,
              "n":len(target),"brier":brier_score_loss(target.y_base_gt50,np.repeat(q,len(target))),
              "log_loss":log_loss(target.y_base_gt50,np.repeat(q,len(target)))}
    pd.DataFrame([constant]).to_csv(RESULTS/"POSTHOC_CONSTANT_REFERENCE.csv",index=False)
    trace=[]
    for model in NAMES:
        for metric in ["roc_auc","average_precision","brier","log_loss"]:
            trace.append({"track":"ZAS","model":model,"metric":metric,"value":means.loc[model,metric],"source":"zas/REPEAT_METRICS.csv","aggregation":"mean of three complete-repeat metrics"})
    for row in target_metrics.to_dict("records"):
        for metric in ["roc_auc","average_precision","brier_score","log_loss"]:
            trace.append({"track":"transport","model":row["arm"]+":"+row["endpoint"],"metric":metric,"value":row[metric],"source":"transport/TARGET_METRICS.csv","aggregation":"fixed-model target cohort"})
    pd.DataFrame(trace).to_csv(OUT/"REPORT_NUMERIC_TRACEABILITY.csv",index=False)
    make_figures(repeats,predictions,target,stress)
    delta_selection=means.loc["historical_selection","roc_auc"]-means.loc[FULL,"roc_auc"]
    delta_cal_brier=means.loc[CAL,"brier"]-means.loc[RAW,"brier"]
    cal_direction="lower" if delta_cal_brier<0 else "higher"
    source_table="Table II. Mean complete-repeat source metrics; all rows use 303 OOF predictions per repeat. Historical-calibration controls remain explicitly limited.\n\n"+md_table(["Procedure","AUC","AP","Brier","Log loss"],[[NAMES[m]]+[format_value(means.loc[m,k]) for k in ["roc_auc","average_precision","brier","log_loss"]] for m in NAMES])
    target_table="Table III. Frozen shared-core transport results. Exclusion and upper assignment are sensitivity definitions, not recovered true >=50% labels.\n\n"+md_table(["Target scenario / arm","n / positive","AUC","Brier"],[[{"base_gt50":"Base >50%","exclude_ambiguous26_50":"Exclude boundary","upper_assignment26_50_positive":"Upper assignment"}[r.endpoint]+(" / raw" if r.arm=="raw_full_source_fit" else " / ensemble"),f"{r.n} / {r.events}",format_value(r.roc_auc),format_value(r.brier_score)] for r in target_metrics.itertuples()])
    stress_summary=stress.groupby("mask_rate").mean(numeric_only=True)
    stress_table="Table IV. Prespecified held-out source input deletion; mean complete-repeat metrics, one mask realization per rate/fold.\n\n"+md_table(["Masked cells","AUC","Brier","Log loss"],[[f"{r*100:.0f}%",format_value(g.roc_auc),format_value(g.brier_score),format_value(g.log_loss)] for r,g in stress_summary.iterrows()])
    group_counts=selections[selections.model_id==CAL].groupby("group").size().to_dict() if "group" in selections.columns else {}
    family_column="family" if "family" in selections.columns else "model_family"
    family_counts=selections[selections.model_id==CAL].groupby(family_column).size().to_dict()
    label_counts="; ".join(f"{k}: {v}/45" for k,v in family_counts.items())
    groups="; ".join(f"{k}: {v}/45" for k,v in group_counts.items())
    warning_count=sum(int(r.get("count",1)) for r in warning_rows)
    values={
        "ABSTRACT_ZAS":f"Historical selection and equal-feature logistic mean AUCs were {means.loc['historical_selection','roc_auc']:.4f} and {means.loc[FULL,'roc_auc']:.4f}; corrected raw/calibrated ensemble Brier scores were {means.loc[RAW,'brier']:.4f}/{means.loc[CAL,'brier']:.4f}.",
        "COHORT_TABLE":"Table I. Source and target evaluation records; counts do not establish verified independent patients.\n\n"+md_table(["Resource / role","Rows","Positive","Endpoint"],[["ZAS development",303,216,">=50% catalogue Cath"],["AngioCAD matched target",377,298,">50% segment rule"],["Target boundary exclusion",367,298,">50%, omit 10 ambiguous"]]),
        "WORKFLOW_FIGURE":fig("workflow","Training and evaluation separation. The full selection procedure is rerun within each development subset. The shared-core transport experiment is a separate model and study track."),
        "SOURCE_RESULTS_TEXT":f"Historical selection mean AUC was {means.loc['historical_selection','roc_auc']:.6f}, compared with {means.loc[FULL,'roc_auc']:.6f} for equal-feature L2. The paired mean selection-minus-L2 difference was {delta_selection:+.6f}. The three paired differences were "+", ".join(f"{v:+.6f}" for v in (repeats[repeats.model_id=='historical_selection'].set_index('repeat').roc_auc-repeats[repeats.model_id==FULL].set_index('repeat').roc_auc))+". These are descriptive split comparisons, not independent-patient significance results.",
        "SOURCE_TABLE":source_table,
        "SOURCE_FIGURE":fig("source_auc","ZAS complete-repeat ROC-AUC. Black points are means; crosses are the three repeat values and lines show their observed range, not confidence intervals. The horizontal axis spans 0-1."),
        "CALIBRATION_RESULTS_TEXT":f"The corrected calibrated ensemble averaged AUC {means.loc[CAL,'roc_auc']:.4f}, Brier {means.loc[CAL,'brier']:.4f} and log loss {means.loc[CAL,'log_loss']:.4f}; the identical-component raw ensemble averaged {means.loc[RAW,'roc_auc']:.4f}, {means.loc[RAW,'brier']:.4f} and {means.loc[RAW,'log_loss']:.4f}. Calibrated-minus-raw mean Brier was {delta_cal_brier:+.6f}, i.e., {cal_direction} probability loss on average. Brier improved in two repeats and worsened in one; AUC decreased in all three. This is the matched calibration-operation comparison; no improvement was assumed in the protocol.",
        "SELECTION_RESULTS_TEXT":"Across 45 corrected component searches, selected model-family counts were "+label_counts+". Selected group counts were "+groups+". The fits used smaller development subsets than the historical full-outer-training search; selection frequencies are therefore descriptive of the new procedure, not direct evidence of feature necessity.",
        "TARGET_TABLE":target_table,
        "CALIBRATION_FIGURE":fig("calibration","Descriptive reliability diagrams using five fixed probability bins. Left: corrected-pool OOF predictions from ZAS repeat 1 only, avoiding treatment of repeated rows as independent. Right: shared-core target predictions under the >50% rule. Bin estimates are sparse and do not supply individual risk guarantees."),
        "STRESS_TABLE":stress_table,
        "STRESS_FIGURE":fig("missingness","Shared-core source AUC under held-out MCAR input deletion. Error bars show the observed range over three overlapping split repeats, not patient-level uncertainty. No model is refitted after masking."),
        "DISCUSSION_SOURCE":f"The equal-feature result gives a selection-minus-simple-model AUC difference of {delta_selection:+.4f}; the simple rich-feature logistic control had higher AUC in all three observed repeats. Thus these results do not establish an automated-selection advantage. They change the interpretation of the much larger historical gain over the sparse five-field reference: additional clinical information is a substantial competing explanation. Removing the ten symptom/examination fields lowered mean AUC by 0.0709, while retaining only 17 risk-plus-symptom fields yielded AUC 0.9036 versus 0.9174 for all 38. The latter compact procedure had Brier 0.1025, close descriptively to the 38-field control's 0.1011, without an equivalence or noninferiority claim. These restrictions concern recorded information, not measurement timing or causal necessity. The corrected calibrated-versus-raw Brier difference of {delta_cal_brier:+.4f} similarly separates implementation independence from empirical probability accuracy; a proper separation design is necessary evidence hygiene, not a guarantee of improvement.",
        "RUNTIME_TEXT":f"Actual checkpoint computation summed to {manifest['checkpoint_seconds_sum']/60:.2f} minutes, with {warning_count} recorded warning occurrence(s). Warnings remain in the evidence package and were not addressed by outcome-driven parameter changes.",
        "QA_TEXT":"@FINAL_QA_PLACEHOLDER@",
        "CONCLUSION_TEXT":f"The computational extension completed information-matched controls, 45 selection-independent calibration components, three information-group ablations and a separate frozen shared-core transport study. The observed selection-minus-equal-feature AUC difference was {delta_selection:+.4f}; the corrected calibration-operation Brier difference was {delta_cal_brier:+.4f}. These results refine the historical high-score claim by identifying the roles of available information and probability construction.",
        "AUTHOR_STATEMENT":"The project authors, in order, are Erukudinnela Akhil Preetham; Chunduri Hari Krishna; G Vennela, Department of Computer Science and Engineering, Prasad V. Potluri Siddhartha Institute of Technology (PVPSIT), Vijayawada, Andhra Pradesh, India. The project guide is K. Naga Gopi. Corresponding-author designation, contributions and funding/conflict declarations remain to be finalized by the authors. No institutional ethics approval or exemption is asserted without a documented determination.",
    }
    qa_path=RESULTS/"FINAL_QA_SUMMARY.json"
    if qa_path.exists():
        qa=json.loads(qa_path.read_text(encoding='utf-8'))
        values['QA_TEXT']=qa['report_paragraph']
    text=(OUT/"REPORT_TEMPLATE.md").read_text(encoding="utf-8")
    for key,value in values.items(): text=text.replace("@"+key+"@",value)
    if re.search(r"@[A-Z_]+@",text): raise RuntimeError("Final QA or report placeholders unresolved")
    (OUT/"CARDIODIAG_RESEARCH_REPORT.md").write_text(text,encoding="utf-8")
    (OUT/"CARDIODIAG_RESEARCH_REPORT.tex").write_text(to_latex(text),encoding="utf-8")
    input_paths=[ZAS/"MANIFEST.json",ZAS/"REPEAT_METRICS.csv",ZAS/"OOF_PREDICTIONS.csv",ZAS/"SELECTIONS.csv",TRANSFER/"TRANSPORT_MANIFEST.json",TRANSFER/"TARGET_METRICS.csv",TRANSFER/"TARGET_PREDICTIONS.csv",TRANSFER/"SOURCE_MCAR_STRESS_REPEAT_METRICS.csv",qa_path,OUT/"REPORT_TEMPLATE.md",Path(__file__)]
    (OUT/"REPORT_INPUTS.json").write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in input_paths},indent=2),encoding="utf-8")
    print(f"Created report Markdown, IEEEtran source, four figure pairs and numeric traceability; words={len(text.split())}")


if __name__=="__main__":
    main()
