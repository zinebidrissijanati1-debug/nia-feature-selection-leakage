"""Print one line per finished run (quick check). Usage: python summarize.py [results_folder]"""
import json, glob, os, sys
folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
print(f"{'run':7} {'ds':4} {'exp':18} {'labels':8} {'alg':14} {'protocol':26} {'fit':10} {'balAcc':>6} {'sens':>5} {'AUC':>5} {'Brier':>6} {'slope':>6} {'nSel':>6}")
for f in sorted(glob.glob(os.path.join(folder, "R*.json"))):
    r = json.load(open(f)); u = r["run"]; m = r["metrics"]
    nsel = sum(r["n_selected"]) / len(r["n_selected"])
    print(f"{u['run_id']:7} {u['dataset']:4} {u['experiment'][:18]:18} {u['labels']:8} {u['algorithm']:14} {u['protocol']:26} {str(u['fitness'])[:10]:10} "
          f"{m['balanced_accuracy']:6.3f} {m['sensitivity']:5.3f} {m['auc']:5.3f} {m['brier']:6.3f} {m['cal_slope']:6.2f} {nsel:6.0f}")
errs = glob.glob(os.path.join(folder, "*.error.txt"))
print(f"\n{len(glob.glob(os.path.join(folder, 'R*.json')))} finished runs, {len(errs)} errors" + (": " + ", ".join(os.path.basename(e) for e in errs) if errs else ""))
