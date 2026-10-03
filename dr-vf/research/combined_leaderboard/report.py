"""Render auditable dual-proxy tables from completed frozen-forecast statistics."""
import argparse
import csv
import json
from pathlib import Path


def _fmt(value):
    return f"{value:.6g}"


def _interval(values):
    return "[" + ", ".join(map(_fmt, values)) + "]"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return ["| " + " | ".join(map(_cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
                "| " + " | ".join(map(_cell, row)) + " |" for row in rows]


def _dual_rows(result, setting, *, realized=False):
    by_proxy = {proxy: {r["model"]: r for r in rows}
                for proxy, rows in result["leaderboards"][setting].items()}
    order = sorted(by_proxy["r2"], key=lambda m: (by_proxy["r2"][m]["family"], by_proxy["r2"][m]["rank"], m))
    rows = []
    for model in order:
        r2 = by_proxy["r2"][model]
        if (r2["family"] == "R") != realized:
            continue
        rows.append([model, r2["family"], r2["rank"], _fmt(r2["loss"])] +
                    ([by_proxy["rk"][model]["rank"], _fmt(by_proxy["rk"][model]["loss"])] if "rk" in by_proxy else []))
    return rows


def _rank_tables(result, setting):
    headers = ["Model", "Family", "R2 rank", "R2 QLIKE"]
    if "rk" in result["leaderboards"][setting]:
        headers += ["RK rank", "RK QLIKE"]
    lines = ["V/E/F comparison population:", ""] + _table(headers, _dual_rows(result, setting))
    realized = _dual_rows(result, setting, realized=True)
    if realized:
        lines += ["", "Realized-information controls (ranks within R only):", ""] + _table(headers, realized)
    return lines


def _write_combined_leaderboard(result, out):
    """One model per row, with both primary proxies and the matching inference."""
    points = {p: {r["model"]: r for r in rows} for p, rows in result["leaderboards"]["primary"].items()}
    contrasts = {(r["model"], r["proxy"]): r for r in result["contrasts"]
                 if r["block"] == result["settings"]["main_block"]}
    rows = []
    for model, r2 in points["r2"].items():
        row = {k: r2[k] for k in ("model", "family", "information_group", "output_adapter", "ranking_population", "seeds")}
        for proxy in ("r2", "rk"):
            point = points[proxy][model]
            contrast = contrasts.get((model, proxy))
            row.update({proxy + "_rank": point["rank"], proxy + "_loss": point["loss"],
                        proxy + "_gain": contrast["gain"] if contrast else None,
                        proxy + "_family_ci_lower": contrast["family_interval95"][0] if contrast else None,
                        proxy + "_family_ci_upper": contrast["family_interval95"][1] if contrast else None,
                        proxy + "_holm_p": contrast["p_holm"] if contrast else None,
                        proxy + "_market_wins": contrast["markets_improved"] if contrast else None,
                        proxy + "_markets": contrast["markets"] if contrast else None,
                        proxy + "_verdict": contrast["verdict"] if contrast else "reference"})
        for name, source in (("substitutions", "substitution_counts"), ("floors", "floor_counts")):
            for label, phase in (("primary", "evaluation_primary"), ("calibration", "calibration")):
                row[label + "_" + name] = sum(s[phase] for s in result[source][model].values())
        rows.append(row)
    rows.sort(key=lambda r: (r["family"] == "R", r["r2_rank"], r["model"]))
    csv_path = out / "leaderboard.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings, sample = result["settings"], result["samples"]["primary"]
    lines = ["# Combined dual-proxy leaderboard", "",
             f"**{sample['rows']:,} common primary market-days · {sample['markets']} markets · "
             f"{sample['calendar_dates']} union-calendar dates.** Both proxies use finite, strictly positive RK origins. "
             "QLIKE is lower-is-better; gain and market wins compare the control against the mean of five VolPFN seed losses. "
             "Positive gains favour VolPFN. VolPFN is the reference and has no comparison with itself.", "",
             f"Main inference: {settings['resamples']:,} synchronized circular bootstrap draws with {settings['main_block']}-date "
             f"blocks; {settings['main_family_slots']} joint E/F proxy hypotheses" +
             (f" and a separate {settings['realized_family_slots']}-hypothesis R family." if settings["realized_family_slots"] else ".") +
             " Displayed intervals are basic Bonferroni family 95% intervals; p-values use Holm correction. "
             "Substitution and floor counts are independent model/seed forecast-record counts, summed over model seeds.", "",
             "[Detailed report, calibration, sensitivities and diagnostics](REPORT.md) · "
             "[Full precision combined CSV](leaderboard.csv)", "",
             "## V/E/F comparison population", ""]
    headers = ["Model", "Family", "Information / input", "Output adapter", "R2 rank", "R2 loss", "R2 gain",
               "R2 family 95% CI", "R2 Holm p", "R2 market wins", "RK rank", "RK loss", "RK gain",
               "RK family 95% CI", "RK Holm p", "RK market wins", "Primary substitutions", "Primary floors"]

    def cells(row):
        values = [row[k] for k in ("model", "family", "information_group", "output_adapter")]
        for proxy in ("r2", "rk"):
            values += [row[proxy + "_rank"], _fmt(row[proxy + "_loss"])]
            if row[proxy + "_gain"] is None:
                values += ["reference"] * 4
            else:
                values += [_fmt(row[proxy + "_gain"]),
                           _interval([row[proxy + "_family_ci_lower"], row[proxy + "_family_ci_upper"]]),
                           _fmt(row[proxy + "_holm_p"]),
                           f"{row[proxy + '_market_wins']}/{row[proxy + '_markets']}"]
        return values + [row["primary_substitutions"], row["primary_floors"]]

    lines += _table(headers, [cells(r) for r in rows if r["family"] != "R"])
    if any(r["family"] == "R" for r in rows):
        lines += ["", "## Realized-information controls", "",
                  "Separate information population; ranks are within R only. Gains still compare each control with VolPFN.", ""]
        lines += _table(headers, [cells(r) for r in rows if r["family"] == "R"])
    lines += ["", "A supported positive gain requires gain > 0, family interval lower bound > 0, and Holm p < 0.05. "
              "Supported negative gains require gain < 0, family interval upper bound < 0, and Holm p < 0.05. "
              "The full-precision CSV supplies both verdicts and calibration counts.", "", result["scope"], ""]
    path = out / "LEADERBOARD.md"
    path.write_text("\n".join(lines))
    return {"leaderboard_markdown": str(path.resolve()), "leaderboard_csv": str(csv_path.resolve())}


def write_report(result, out, *, heatmap=True):
    """Write Markdown, flat loss/contrast CSVs and an optional per-market heatmap."""
    if result.get("status") != "complete":
        raise ValueError("Only completed analyses can be reported")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings, samples = result["settings"], result["samples"]
    primary = samples["primary"]
    lines = ["# Combined dual-proxy volatility leaderboard", "",
             "[One combined table with both proxies, ranks, gains and inference](LEADERBOARD.md) · "
             "[Combined full-precision CSV](leaderboard.csv)", "",
             f"Primary population: **{primary['rows']:,} market-days**, {primary['markets']} markets, "
             f"{primary['calendar_dates']} distinct calendar dates. Both primary proxies use the same "
             "finite, strictly positive RK mask. Lower QLIKE is better; positive paired gains favour VolPFN.", "",
             result["scope"], "",
             "## Information and output conventions", "",
             "Ranks are descriptive within V/E/F or within the separate R control block. Their "
             "information sets differ; realized-information controls are never ranked against V/E/F "
             "and use a separate testing family.", ""]
    lines += _table(["Model", "Family", "Information group", "Output adapter"], [
        [m, spec["family"], spec.get("information_group", next(
            r["information_group"] for r in result["leaderboards"]["primary"]["r2"] if r["model"] == m)),
         spec["output_adapter"]] for m, spec in sorted(result["models"].items())])
    lines += ["", "Scores use QLIKE = y/h + log(h), which permits zero R2 targets. "
              "The R2 forecast is c × second moment and the RK forecast is c × variance. "
              "Foundation adapters explicitly set mean to zero and variance equal to second moment. "
              "VolPFN's point loss is the mean of its five seed losses, with equal weighting of markets "
              "and a time average within each market.", "", result["calibration_rule"] + ".", "",
              "## Primary calibrated ranks and losses", ""]
    lines += _rank_tables(result, "primary")
    lines += ["", "## Main paired comparisons", "",
              f"{settings['resamples']:,} synchronized circular bootstrap draws; main blocks = "
              f"{settings['main_block']} observed union-calendar dates. Missing market dates remain absent, "
              "and each draw reweights by that market's sampled observed dates. No independent resampling of markets or training seeds.", "",
              f"Holm correction includes both proxies jointly: {settings['main_family_slots']} E/F hypotheses"
              + (f" and a separate {settings['realized_family_slots']}-hypothesis realized-information family." if settings["realized_family_slots"] else ".") +
              " Basic family intervals use marginal coverage 1 − 0.05/K (Bonferroni). A positive result requires "
              "gain > 0, family interval lower bound > 0 and Holm p < 0.05. An interval containing zero does not establish equivalence.", ""]
    main = [r for r in result["contrasts"] if r["block"] == settings["main_block"]]
    lines += _table(["Control", "Proxy", "Family", "Gain", "Pointwise 95% CI", "Family 95% CI", "Holm p", "Positive seeds", "Markets improved", "Verdict"], [
        [r["model"], r["proxy"], r["testing_family"], _fmt(r["gain"]), _interval(r["interval95"]),
         _interval(r["family_interval95"]), _fmt(r["p_holm"]), f"{r['positive_seeds']}/5",
         f"{r['markets_improved']}/{r['markets']}", r["verdict"]] for r in main])
    lines += ["", "A supported negative gain requires gain < 0, family interval upper bound < 0 and Holm p < 0.05."]
    lines += ["", "## Block-length sensitivity", "",
              "These checks use the same declared comparison families within each block setting. "
              "They cannot replace the main setting or be selected by their outcomes.", ""]
    lines += _table(["Control", "Proxy", "Block", "Family 95% CI", "Holm p", "Verdict"], [
        [r["model"], r["proxy"], r["block"], _interval(r["family_interval95"]), _fmt(r["p_holm"]),
         r["verdict"]] for r in result["contrasts"]])
    lines += ["", "## Raw forecast sensitivity", "",
              "Descriptive uncalibrated losses on the identical primary mask; these are not additional hypothesis tests.", ""]
    lines += _rank_tables(result, "raw")
    lines += ["", "## Full R2 population sensitivity (S_all)", "",
              f"All {samples['S_all']['rows']:,} frozen evaluation origins are retained. The primary mask excludes "
              f"{samples['primary_excluded_rk_rows']:,} origins solely because RK is missing, nonfinite or nonpositive. "
              "This calibrated R2 sensitivity is descriptive.", ""]
    lines += _rank_tables(result, "S_all")
    lines += ["", "Uncalibrated S_all R2 sensitivity (also descriptive):", ""]
    lines += _rank_tables(result, "S_all_raw")
    lines += ["", "## Comparator complete-case sensitivity", "",
              "Descriptive calibrated primary-proxy comparisons after excluding that comparator's substituted "
              "origins (any declared comparator seed). These smaller model-specific samples are not the primary "
              "comparison and carry no hypothesis tests. Markets with no remaining origins are absent from this sensitivity only.", ""]
    lines += _table(["Control", "Proxy", "Rows", "Markets", "Substituted rows excluded", "Gain"], [
        [r["model"], r["proxy"], r["rows"], r["markets"], r["excluded_comparator_substituted_rows"],
         _fmt(r["gain"]) if r["gain"] is not None else "unavailable"]
        for r in result["secondary_contrasts"]["complete_case"]])
    lines += ["", "## Calibration, seed range, substitutions and floors", "",
              "Each scalar is fitted to the same 72 frozen development origins. Seed ranges describe the "
              "five retained training runs; they are not training-population confidence intervals. Floor activations "
              "and substitutions are counted independently; the same forecast can carry both flags. Counts refer "
              "to model/seed forecast records. Loss CSV counts sum over the declared model seeds.", ""]
    lines += _table(["Model", "Seed", "Scalar", "Calibration substitutions", "Primary substitutions", "S_all substitutions",
                     "Calibration floors", "Primary floors", "S_all floors"], [
        [m, s, _fmt(cal["scalar"]), result["substitution_counts"][m][s]["calibration"],
         result["substitution_counts"][m][s]["evaluation_primary"], result["substitution_counts"][m][s]["evaluation_S_all"],
         result["floor_counts"][m][s]["calibration"], result["floor_counts"][m][s]["evaluation_primary"],
         result["floor_counts"][m][s]["evaluation_S_all"]]
        for m, seeds in sorted(result["calibration"].items()) for s, cal in seeds.items()])
    lines += [""] + _table(["Control", "Proxy", "Five-seed gain range"], [
        [r["model"], r["proxy"], _interval(r["seed_range"])] for r in main])
    fit_diagnostics = result.get("fit_diagnostics", [])
    if fit_diagnostics:
        columns = list(dict.fromkeys(k for row in fit_diagnostics for k in row))
        lines += ["", "## Fitting diagnostics by model and market", "",
                  "Pipeline audit counts are retained as supplied below. Fit attempts and failures may include "
                  "retries and do not necessarily equal scored forecast counts. Primary forecast rows remain retained.", ""]
        lines += _table([k.replace("_", " ").capitalize() for k in columns],
                        [[r.get(k, "") for k in columns] for r in fit_diagnostics])
    lines += ["", "## Scope", "", result["scope"], "",
              "The bootstrap intervals and p-values are approximate and depend on suitable temporal regularity. "
              "Cross-market dependence is retained by synchronized dates. Availability does not alter the sample: "
              "missing required forecast rows fail the analysis, and frozen-adapter substitutions and floor activations "
              "are retained and counted separately.", "",
              "Optional model confidence sets and ratio statistics are omitted in this minimum v1 report.", ""]

    score_rows = [dict(setting=setting, proxy=proxy, **{k: r[k] for k in
                  ("model", "family", "information_group", "output_adapter", "ranking_population", "rank", "loss", "seeds")},
                  seed_loss_min=r["seed_range"][0], seed_loss_max=r["seed_range"][1],
                  substitutions=sum(s["evaluation_primary" if setting in ("primary", "raw") else "evaluation_S_all"]
                                    for s in result["substitution_counts"][r["model"]].values()),
                  floors=sum(s["evaluation_primary" if setting in ("primary", "raw") else "evaluation_S_all"]
                             for s in result["floor_counts"][r["model"]].values()))
                  for setting, proxies in result["leaderboards"].items() for proxy, rows in proxies.items() for r in rows]
    contrasts = [{k: row[k] for k in ("model", "proxy", "family", "testing_family", "family_slots", "block",
                 "block_role", "gain", "p_raw", "p_holm", "positive_seeds", "markets_improved", "markets", "directional_pass", "negative_supported", "verdict")} |
                 {"interval95_lower": row["interval95"][0], "interval95_upper": row["interval95"][1],
                  "family_interval95_lower": row["family_interval95"][0], "family_interval95_upper": row["family_interval95"][1]}
                 for row in result["contrasts"]]
    outputs = _write_combined_leaderboard(result, out)
    csv_tables = [("losses", score_rows), ("contrasts", contrasts)]
    if fit_diagnostics:
        csv_tables.append(("fit_diagnostics", fit_diagnostics))
    for name, rows in csv_tables:
        path = out / (name + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)) if rows else ["model"])
            writer.writeheader()
            writer.writerows(rows)
        outputs[name] = str(path.resolve())
    if heatmap and main:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            lines += ["The optional per-market heatmap was omitted because matplotlib is unavailable.", ""]
        else:
            names = sorted(main[0]["per_market_gain"])
            values = np.array([[r["per_market_gain"][m] for m in names] for r in main])
            limit = max(float(np.max(np.abs(values))), 1e-12)
            fig, ax = plt.subplots(figsize=(max(9, len(names) * .37), max(4, len(main) * .3)))
            image = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
            ax.set_xticks(range(len(names)), names, rotation=90)
            ax.set_yticks(range(len(main)), [f"{r['family']} · {r['model']} · {r['proxy']}" for r in main])
            ax.set_title("Calibrated primary QLIKE gain by market; positive favours VolPFN")
            fig.colorbar(image, ax=ax, label="Control loss − mean VolPFN seed loss")
            fig.tight_layout()
            path = out / "market_gains.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs["heatmap"] = str(path.resolve())
            lines += ["## Per-market gains", "", "Descriptive effect sizes; cells are not separately tested.", "",
                      "![Per-market primary QLIKE gains](market_gains.png)", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines))
    outputs["markdown"] = str(path.resolve())
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-heatmap", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_report(json.loads(args.result.read_text()), args.out, heatmap=not args.no_heatmap), indent=2))


if __name__ == "__main__":
    main()
