import pandas as pd
from lxml import etree
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram

XML_DIR = Path.home() / "your/path"
OUTPUT_DIR = Path.home() / "your/path/processed_data"
REPORT_DIR = Path.home() / "your/path/abnormal_report"
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
COLUMN_MAP = {
    1: "Sample No", 2: "Time(ms)",
    3: "column_1", 5: "column_2", 6: "column_3",
    7: "column_4", 9: "column_5", 10: "column_6",
    11: "column_7", 13: "column_8", 14: "column_9"
}


def safe_strip(text):
    return text.strip() if text and isinstance(text, str) else ""


def write_anomaly_report(file_path, anomalies):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("abnormal_total\n")
        f.write("=" * 85 + "\n")
        f.write("Sample No | data       | type_abnormal    | original_data\n")
        f.write("-" * 85 + "\n")
        for record in anomalies:
            f.write(f"{record['row']:10} | {record['column']:10} | {record['type']:10} | {record['value']}\n")


def analyze_data(df, xml_name, threshold):
    report_path = REPORT_DIR / f"{xml_name}_abnormal_report.txt"
    anomalies = []
    stats = {
        "channels": defaultdict(lambda: {"missing": 0, "outliers": 0, "total": 0}),
        "columns": defaultdict(lambda: {"missing": 0, "outliers": 0}),
        "total_data": len(df),
        "valid_data": len(df),
        "anomaly_total": 0,
        "accuracy": 0.0
    }

    numeric_cols = [col for col in COLUMN_MAP.values() if col not in ["Sample No", "Time(ms)"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').replace(0, np.nan)

    channel_def = {
        "A": ["column_1","column_2","column_3"],
        "B": ["column_4","column_5","column_6"],
        "C": ["column_7","column_8","column_9"] 
    }
    stats["valid_data"] = len(df)

    for channel, cols in channel_def.items():
        for col in cols:
            if col not in df.columns:
                continue
            series = df[col]
            stats["columns"][col]["total"] = len(df)
            stats["channels"][channel]["total"] += len(df)
            na_count = series.isna().sum()
            stats["columns"][col]["missing"] += na_count
            stats["channels"][channel]["missing"] += na_count
            stats["anomaly_total"] += na_count
            for idx in df[series.isna()].index:
                anomalies.append({
                    "row": df.loc[idx, "xml_row"],
                    "column": col,
                    "type": "MISSING",
                    "value": "NULL"
                })

            valid_data = series.dropna()
            if len(valid_data) > 1:
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (valid_data < lower_bound) | (valid_data > upper_bound)
                num_outliers = int(outlier_mask.sum())
                stats["columns"][col]["outliers"] += num_outliers
                stats["channels"][channel]["outliers"] += num_outliers
                stats["anomaly_total"] += num_outliers

                for idx in valid_data[outlier_mask].index:
                    anomalies.append({
                        "row": df.loc[idx, "xml_row"],
                        "column": col,
                        "type": "ABNORMAL_SCORE",
                        "value": f"{valid_data[idx]:.2f}"
                    })

    total_points = stats["valid_data"] * len(numeric_cols)
    stats["accuracy"] = (total_points - stats["anomaly_total"]) / total_points * 100 if total_points > 0 else 0

    min_filtered_row = df["xml_row"].min()
    anomalies = [record for record in anomalies if record["row"] >= min_filtered_row]
    write_anomaly_report(report_path, sorted(anomalies, key=lambda x: x["row"]))

    return stats


def parse_xml(xml_path):
    result = {
        "file_name": xml_path.name,
        "status": "success",
        "threshold": None,
        "valid_rows": 0
    }
    try:
        tree = etree.parse(xml_path)
        time_xpath = '//ss:Worksheet[@ss:Name="Sheet_No"]/ss:Table/ss:Row[2]/ss:Cell[3]/ss:Data/text()'
        threshold = int(safe_strip(tree.xpath(time_xpath, namespaces=NS)[0]))
        result["threshold"] = threshold

        data = []
        for _, elem in etree.iterparse(xml_path, tag="{urn:schemas-microsoft-com:office:spreadsheet}Row"):
            row_data = {}

            sample_no_cell = elem.find("ss:Cell", namespaces=NS)
            if sample_no_cell is None:
                elem.clear()
                continue
            sample_no_data = sample_no_cell.find("ss:Data", namespaces=NS)
            sample_no = safe_strip(sample_no_data.text if sample_no_data is not None else "")
            if sample_no == "" or not sample_no.isdigit():
                elem.clear()
                continue

            row_data["xml_row"] = int(sample_no)
            row_data["Sample No"] = int(sample_no)

            for cell_idx, cell in enumerate(elem.findall("ss:Cell", namespaces=NS), start=1):
                if cell_idx not in COLUMN_MAP:
                    continue
                if cell_idx == 1:
                    continue
                data_node = cell.find("ss:Data", namespaces=NS)
                row_data[COLUMN_MAP[cell_idx]] = safe_strip(data_node.text if data_node is not None else "")

            if sample_no.isdigit():
                if int(sample_no) >= threshold:
                    data.append(row_data)
            elem.clear()

        df = pd.DataFrame(data)
        if not df.empty:
            result["stats"] = analyze_data(df, xml_path.stem, threshold)
            result["valid_rows"] = len(df)
            output_path = OUTPUT_DIR / f"{xml_path.stem}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        result.update({"status": "fail", "error": str(e)})
    return result


def generate_summary_report(results):
    report_path = Path.home() / "your/path/report_total.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("report_total\n")
        f.write(f"generated time：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 85 + "\n")
        success_files = [r for r in results if r["status"] == "success"]
        f.write(f"Number of successfully processed files：{len(success_files)}\n")
        f.write(f"Number of failed processed files：{len(results) - len(success_files)}\n\n")
        total_data = sum(res["stats"]["total_data"] +res['threshold']for res in success_files if "stats" in res)
        valid_data = sum(res["stats"]["valid_data"] for res in success_files if "stats" in res)
        anomaly_total = sum(res["stats"]["anomaly_total"] for res in success_files if "stats" in res)
        total_points = sum((res["stats"]["valid_data"]+res['threshold'] )* 12 for res in success_files if "stats" in res)

        f.write("\nGlobal statistical information：\n")
        f.write("-" * 85 + "\n")
        f.write(f"total_data：{total_data} rows\n")
        f.write(f"total_points：{total_points} \n")
        f.write(f"anomaly_total：{anomaly_total} \n")
        f.write("=" * 85 + "\n\n")

        for res in success_files:
            if 'stats' not in res:
                continue
            stats = res["stats"]
            total_rows = res.get("valid_rows", 0)
            f.write(f"\n file：{res['file_name']}\n")
            f.write(f"BL：{res['threshold']} | Data rows after BL：{total_rows}\n")
            f.write(f"Validity_Rates_Thisfile：{stats['accuracy']:.2f}%\n")
            f.write("-" * 85 + "\n")

            f.write("Statistics_1：\n")
            for chan in ["A", "B", "C"]:
                chan_data = stats["channels"][chan]
                channel_total = total_rows * 3
                normal = channel_total - chan_data["missing"] - chan_data["outliers"]
                rate = (normal / channel_total) * 100 if channel_total > 0 else 0
                f.write(f"{chan:6} | Missing：{chan_data['missing']:4} | Abnormal：{chan_data['outliers']:4} | Validity：{rate:.1f}%\n")

            f.write("\nStatistics_COLUMN：\n")
            for col, counts in stats["columns"].items():
                total = counts.get("total", total_rows)
                rate = 100 - ((counts["missing"] + counts["outliers"]) / total) * 100 if total > 0 else 0
                f.write(f"{col:10} | Missing：{counts['missing']:4} | Abnormal：{counts['outliers']:4} | Validity：{rate:.1f}%\n")
            f.write("=" * 85 + "\n")



def plot_missing_data_and_efficiency(results):
    numeric_cols = [col for col in COLUMN_MAP.values() if col not in ["Sample No", "Time(ms)"]]
    valid_results = [r for r in results if r.get('status') == "success" and 'stats' in r]
    file_names = [r['file_name'] for r in valid_results]
    num_files = len(file_names)
    num_cols = len(numeric_cols)
    short_names = []
    for name in file_names:
        name_no_ext = name[:-4] if name.lower().endswith(".xml") else name
        parts = name_no_ext.split("-")
        short_name = "-".join(parts[:3])
        short_names.append(short_name)
    fig, ax = plt.subplots(figsize=(num_cols * 1.2, num_files * 0.6))
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(-0.5, num_files - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(numeric_cols, rotation=0, fontsize=10)
    ax.set_yticks(range(num_files))
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_files, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.4, zorder=1)
    plt.subplots_adjust(right=0.3)

    for i, res in enumerate(valid_results):
        stats = res['stats']
        for j, col in enumerate(numeric_cols):
            col_stats = stats["columns"].get(col)
            if col_stats is None:
                continue
            total = col_stats.get("total", 0)
            missing = col_stats.get("missing", 0)
            outliers = col_stats.get("outliers", 0)
            if total > 0:
                efficiency = 100 - ((missing + outliers) / total * 100)
            else:
                efficiency = 0
            if missing > 0:
                continue
            color = "yellow" if efficiency < 95 else "blue"
            ax.scatter(j, i, s=50, color=color, edgecolors="white")
            ax.text(j, i - 0.2, f"{efficiency:.1f}%", ha="center", va="center", fontsize=8, color="black")

    num_channels = len(numeric_cols) // 3
    for c in range(1, num_channels):
        pos = c * 3 - 0.5
        ax.axvline(x=pos, color="purple", linestyle="--", linewidth=1, zorder=2)
    channel_labels = ["A", "B", "C"]
    for k in range(num_channels):
        center_x = k * 3 + 1
        if k < len(channel_labels):
            ax.text(center_x, -1.0, channel_labels[k], ha="center", va="center", fontsize=10, color="purple")

    ax.set_title("Visualizetion of Missing Data Values and Validity Rates\n\n",fontsize=18)
    legend_handles = [
        mpatches.Patch(color="blue", label="Validity Rates ≥ 95%"),
        mpatches.Patch(color="yellow", label="Validity Rates< 95%"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="Missing：no plotted")
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.17, 1),borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(Path.home() / "Desktop/Visualizetion of Missing Data Values and Validity Rates.png", dpi=300)
    plt.show()


def parse_anomaly_report(report_path):
    anomalies = []
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[4:]:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            try:
                sample_no = int(parts[0])
            except:
                sample_no = None
            anomalies.append({
                "Sample No": sample_no,
                "data": parts[1],
                "type_abnormal": parts[2],
                "original_data": parts[3]
            })
    except Exception as e:
        print(f"Analyze the report {report_path} failed：{e}")
    return anomalies



def generate_all_vector_scatter_full_after_th():

    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No find the *.CSV file in {OUTPUT_DIR} ")
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "xml_row" not in df.columns:
            df["xml_row"] = np.arange(len(df))
        report_path = REPORT_DIR / f"{csv_file.stem}_abnormal_report.txt"
        anomaly_records = []
        if report_path.exists():
            anomaly_records = parse_anomaly_report(report_path)
        anomaly_dict = {}
        for rec in anomaly_records:
            col_name = rec["data"]
            sn = rec["Sample No"]
            if sn is None:
                continue
            anomaly_dict.setdefault(col_name, {"Missing": set(), "ABNORMAL_SCORE": set()})
            if rec["type_abnormal"] == "Foregoing":
                anomaly_dict[col_name]["Missing"].add(sn)
            elif rec["type_abnormal"] == "ABNORMAL_SCORE":
                anomaly_dict[col_name]["ABNORMAL_SCORE"].add(sn)

        numeric_cols = [col for col in df.columns if col not in ["Sample No", "Time(ms)"]]
        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce").values
            sample_nums = pd.to_numeric(df["xml_row"], errors="coerce").values
            indices = np.arange(len(values))
            missing_set = anomaly_dict.get(col, {}).get("Missing", set())
            abnormal_set = anomaly_dict.get(col, {}).get("ABNORMAL_SCORE", set())

            normal_idx = []
            missing_idx = []
            abnormal_idx = []
            for i, (sn, v) in enumerate(zip(sample_nums, values)):
                if np.isnan(v) or (sn in missing_set):
                    missing_idx.append(i)
                elif sn in abnormal_set:
                    abnormal_idx.append(i)
                else:
                    normal_idx.append(i)

            if normal_idx:
                normal_vals = values[normal_idx]
                vmin = np.min(normal_vals)
                vmax = np.max(normal_vals)
                margin = 0.1 * (vmax - vmin) if (vmax > vmin) else 1
            else:
                margin = 1
            missing_vals = np.full(len(missing_idx), (vmin - margin) if normal_idx else 0)
            normal_vals = values[normal_idx]
            abnormal_vals = values[abnormal_idx]

            plt.figure(figsize=(20, 10))
            if normal_idx:
                plt.scatter(indices[normal_idx], normal_vals, c='black', s=5, label='Normal')
            if missing_idx:
                plt.scatter(indices[missing_idx], missing_vals, c='red', s=5, label='Missing')
            if abnormal_idx:
                plt.scatter(indices[abnormal_idx], abnormal_vals, c='blue', s=5, label='ABNORMAL_SCORE')
            plt.xlabel("time_series_points", fontsize=14)
            plt.ylabel("data", fontsize=14)
            plt.title(f"{csv_file.stem} - {col} scatter diagram（ {len(values)}points at total）", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.tight_layout(
            output_svg = Path.home() / "your" /'path'/ f"{csv_file.stem}_{col}_scatter.svg"
            plt.savefig(output_svg, format="svg", dpi=300)
            plt.close()  
            print(f"Generated Vector：{output_svg}")



def read_full_xml(xml_path):
    data = []
    try:
        for _, elem in etree.iterparse(xml_path, tag="{urn:schemas-microsoft-com:office:spreadsheet}Row"):
            row_data = {}
            sample_no_cell = elem.find("ss:Cell", namespaces=NS)
            if sample_no_cell is None:
                elem.clear()
                continue
            sample_no_data = sample_no_cell.find("ss:Data", namespaces=NS)
            sample_no = safe_strip(sample_no_data.text if sample_no_data is not None else "")
            if sample_no == "" or not sample_no.isdigit():
                elem.clear()
                continue
            row_data["xml_row"] = int(sample_no)
            row_data["Sample No"] = int(sample_no)
            for cell_idx, cell in enumerate(elem.findall("ss:Cell", namespaces=NS), start=1):
                if cell_idx not in COLUMN_MAP:
                    continue
                if cell_idx == 1:
                    continue
                data_node = cell.find("ss:Data", namespaces=NS)
                row_data[COLUMN_MAP[cell_idx]] = safe_strip(data_node.text if data_node is not None else "")
            data.append(row_data)
            elem.clear()
    except Exception as e:
        print(f"Getting XML {xml_path} failed: {e}")
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    print("Start the data analysis process...")
    results = []
    for xml_file in XML_DIR.glob("*.xml"):
        print(f"Analyzing：{xml_file.name}")
        results.append(parse_xml(xml_file))
    generate_summary_report(results)
    generate_all_vector_scatter_full_after_th()
    print("\n Completed！Output_Path：")
    print(f"processed_data：{OUTPUT_DIR}")
    print(f"abnormal_report：{REPORT_DIR}")
    print(f"report_total：{Path.home()}/your/path/report_total.txt")
    print(f"Visualizetion：{Path.home()}/your/path/Visualizetion of Missing Data Values and Validity Rates.png")
