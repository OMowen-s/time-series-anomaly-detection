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

XML_DIR = Path.home() / "Desktop/paper/血流数据/新数据"
OUTPUT_DIR = Path.home() / "Desktop/processed_data"
REPORT_DIR = Path.home() / "Desktop/异常报告"
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
COLUMN_MAP = {
    1: "Sample No", 2: "Time(ms)",
    3: "血流 一", 5: "CMBC 一", 6: "V 一",
    7: "血流 二", 9: "CMBC 二", 10: "V 二",
    11: "血流 三", 13: "CMBC 三", 14: "V 三",
    15: "血流 四", 17: "CMBC 四", 18: "V 四"
}


def safe_strip(text):
    return text.strip() if text and isinstance(text, str) else ""


def write_anomaly_report(file_path, anomalies):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("多普勒血流仪异常数据报告\n")
        f.write("=" * 85 + "\n")
        f.write("Sample No | 数据列       | 异常类型    | 原始值\n")
        f.write("-" * 85 + "\n")
        for record in anomalies:
            f.write(f"{record['row']:10} | {record['column']:10} | {record['type']:10} | {record['value']}\n")


def analyze_data(df, xml_name, threshold):
    report_path = REPORT_DIR / f"{xml_name}_异常报告.txt"
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
        "通道一": ["血流 一", "CMBC 一", "V 一"],
        "通道二": ["血流 二", "CMBC 二", "V 二"],
        "通道三": ["血流 三", "CMBC 三", "V 三"],
        "通道四": ["血流 四", "CMBC 四", "V 四"]
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
                    "type": "缺失值",
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
                        "type": "分数异常",
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
        "status": "成功",
        "threshold": None,
        "valid_rows": 0
    }
    try:
        tree = etree.parse(xml_path)
        time_xpath = '//ss:Worksheet[@ss:Name="温控血流"]/ss:Table/ss:Row[2]/ss:Cell[3]/ss:Data/text()'
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
        result.update({"status": "失败", "error": str(e)})
    return result


def generate_summary_report(results):
    report_path = Path.home() / "Desktop/多普勒数据验证总报告.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("多普勒血流数据验证总报告\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 85 + "\n")
        success_files = [r for r in results if r["status"] == "成功"]
        f.write(f"成功处理文件数：{len(success_files)}\n")
        f.write(f"失败处理文件数：{len(results) - len(success_files)}\n\n")
        total_data = sum(res["stats"]["total_data"] +res['threshold']for res in success_files if "stats" in res)
        valid_data = sum(res["stats"]["valid_data"] for res in success_files if "stats" in res)
        anomaly_total = sum(res["stats"]["anomaly_total"] for res in success_files if "stats" in res)
        total_points = sum((res["stats"]["valid_data"]+res['threshold'] )* 12 for res in success_files if "stats" in res)

        f.write("\n全局统计信息：\n")
        f.write("-" * 85 + "\n")
        f.write(f"总原始数据量：{total_data} 行\n")
        f.write(f"总数据点数：{total_points} 个\n")
        f.write(f"异常数据点数：{anomaly_total} 个\n")
        f.write("=" * 85 + "\n\n")

        for res in success_files:
            if 'stats' not in res:
                continue
            stats = res["stats"]
            total_rows = res.get("valid_rows", 0)
            f.write(f"\n文件：{res['file_name']}\n")
            f.write(f"BL：{res['threshold']} | BL后数据行：{total_rows}\n")
            f.write(f"本文件有效率：{stats['accuracy']:.2f}%\n")
            f.write("-" * 85 + "\n")

            f.write("通道统计：\n")
            for chan in ["通道一", "通道二", "通道三", "通道四"]:
                chan_data = stats["channels"][chan]
                channel_total = total_rows * 3
                normal = channel_total - chan_data["missing"] - chan_data["outliers"]
                rate = (normal / channel_total) * 100 if channel_total > 0 else 0
                f.write(f"{chan:6} | 缺失：{chan_data['missing']:4} | 异常：{chan_data['outliers']:4} | 有效率：{rate:.1f}%\n")

            f.write("\n详细列统计：\n")
            for col, counts in stats["columns"].items():
                total = counts.get("total", total_rows)
                rate = 100 - ((counts["missing"] + counts["outliers"]) / total) * 100 if total > 0 else 0
                f.write(f"{col:10} | 缺失：{counts['missing']:4} | 异常：{counts['outliers']:4} | 有效率：{rate:.1f}%\n")
            f.write("=" * 85 + "\n")



def plot_missing_data_and_efficiency(results):
    numeric_cols = [col for col in COLUMN_MAP.values() if col not in ["Sample No", "Time(ms)"]]
    valid_results = [r for r in results if r.get('status') == "成功" and 'stats' in r]
    file_names = [r['file_name'] for r in valid_results]
    num_files = len(file_names)
    num_cols = len(numeric_cols)
    # 处理文件名，截取前3个“-”分隔的部分，去除.xml后缀
    short_names = []
    for name in file_names:
        name_no_ext = name[:-4] if name.lower().endswith(".xml") else name
        parts = name_no_ext.split("-")
        short_name = "-".join(parts[:3])
        short_names.append(short_name)
    # 设置图像尺寸，确保横轴和纵轴显示完整
    fig, ax = plt.subplots(figsize=(num_cols * 1.2, num_files * 0.6))
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(-0.5, num_files - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(numeric_cols, rotation=0, fontsize=10)
    ax.set_yticks(range(num_files))
    ax.set_yticklabels(short_names, fontsize=10)
    # 设置次刻度用于绘制横纵坐标交叉的网格线
    ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_files, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.4, zorder=1)
    # 为防止图例遮挡数据，调整右侧留白
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
            # 如果该列有缺失值，则不绘制点
            if missing > 0:
                continue
            # 根据效率确定点的颜色，效率小于95%标记为黄色，否则为蓝色
            color = "yellow" if efficiency < 95 else "blue"
            ax.scatter(j, i, s=50, color=color, edgecolors="white")
            # 为避免点与文本重叠，在点上方（向上偏移0.3）添加效率文本
            ax.text(j, i - 0.2, f"{efficiency:.1f}%", ha="center", va="center", fontsize=8, color="black")

    # 在横轴上每3列划分为一个通道，添加分隔线和通道标签
    num_channels = len(numeric_cols) // 3
    for c in range(1, num_channels):
        pos = c * 3 - 0.5
        ax.axvline(x=pos, color="purple", linestyle="--", linewidth=1, zorder=2)
    channel_labels = ["通道一", "通道二", "通道三", "通道四"]
    for k in range(num_channels):
        center_x = k * 3 + 1
        if k < len(channel_labels):
            ax.text(center_x, -1.0, channel_labels[k], ha="center", va="center", fontsize=10, color="purple")

    ax.set_title("数据缺失值和有效率可视化\n\n",fontsize=18)
    legend_handles = [
        mpatches.Patch(color="blue", label="有效率 ≥ 95%"),
        mpatches.Patch(color="yellow", label="有效率 < 95%"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="缺失值：未绘点")
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.17, 1),borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(Path.home() / "Desktop/数据缺失值和有效率可视化.png", dpi=300)
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
                "数据列": parts[1],
                "异常类型": parts[2],
                "原始值": parts[3]
            })
    except Exception as e:
        print(f"解析异常报告 {report_path} 出错：{e}")
    return anomalies



def generate_all_vector_scatter_full_after_th():

    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"未在 {OUTPUT_DIR} 中找到 CSV 文件")
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # 如果不存在 "xml_row" 列，则补充行号信息（原 XML 中 Sample No 对应的行号）
        if "xml_row" not in df.columns:
            df["xml_row"] = np.arange(len(df))
        # 尝试读取对应的异常报告文件（假设命名为 <stem>_异常报告.txt）
        report_path = REPORT_DIR / f"{csv_file.stem}_异常报告.txt"
        anomaly_records = []
        if report_path.exists():
            anomaly_records = parse_anomaly_report(report_path)
        anomaly_dict = {}
        for rec in anomaly_records:
            col_name = rec["数据列"]
            sn = rec["Sample No"]
            if sn is None:
                continue
            anomaly_dict.setdefault(col_name, {"缺失": set(), "分数异常": set()})
            if rec["异常类型"] == "缺失值":
                anomaly_dict[col_name]["缺失"].add(sn)
            elif rec["异常类型"] == "分数异常":
                anomaly_dict[col_name]["分数异常"].add(sn)

        # 获取所有数值列（排除 Sample No 与 Time(ms)）
        numeric_cols = [col for col in df.columns if col not in ["Sample No", "Time(ms)"]]
        # 对每个数值列分别生成图像
        for col in numeric_cols:
            # 取整列数据（CSV 已过滤时间阈值后数据）
            values = pd.to_numeric(df[col], errors="coerce").values
            sample_nums = pd.to_numeric(df["xml_row"], errors="coerce").values
            indices = np.arange(len(values))

            # 根据异常报告，构造缺失值和分数异常值的样本号集合
            missing_set = anomaly_dict.get(col, {}).get("缺失", set())
            abnormal_set = anomaly_dict.get(col, {}).get("分数异常", set())

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

            # 若存在正常数据，则计算最小值及一定间隔，用于缺失值展示
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
                plt.scatter(indices[normal_idx], normal_vals, c='black', s=5, label='正常值')
            if missing_idx:
                plt.scatter(indices[missing_idx], missing_vals, c='red', s=5, label='缺失值')
            if abnormal_idx:
                plt.scatter(indices[abnormal_idx], abnormal_vals, c='blue', s=5, label='分数异常')
            plt.xlabel("时序点", fontsize=14)
            plt.ylabel("数值", fontsize=14)
            plt.title(f"{csv_file.stem} - {col} 散点图（共 {len(values)} 个点）", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.tight_layout()
            # 保存为 SVG 格式的矢量图
            output_svg = Path.home() / "Desktop" /'plot'/ f"{csv_file.stem}_{col}_scatter.svg"
            plt.savefig(output_svg, format="svg", dpi=300)
            plt.close()  # 关闭当前图像，避免同时打开过多图像
            print(f"已生成矢量图：{output_svg}")



def read_full_xml(xml_path):
    data = []
    try:
        # 遍历所有 Row 节点
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
            # 保存 Sample No 到两个字段
            row_data["xml_row"] = int(sample_no)
            row_data["Sample No"] = int(sample_no)
            # 遍历每个单元格，根据 COLUMN_MAP 提取数据（排除温度列）
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
        print(f"读取 XML {xml_path} 错误: {e}")
    df = pd.DataFrame(data)
    return df



def generate_all_vector_scatter_full():
    xml_files = list(XML_DIR.glob("*.xml"))
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"未在 {OUTPUT_DIR} 中找到 CSV 文件")
        return

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "xml_row" not in df.columns:
            df["xml_row"] = np.arange(len(df))
        report_path = REPORT_DIR / f"{csv_file.stem}_异常报告.txt"
        anomaly_records = []
        if report_path.exists():
            anomaly_records = parse_anomaly_report(report_path)
        anomaly_dict = {}
        for rec in anomaly_records:
            col_name = rec["数据列"]
            sn = rec["Sample No"]
            if sn is None:
                continue
            anomaly_dict.setdefault(col_name, {"缺失": set(), "分数异常": set()})
            if rec["异常类型"] == "缺失值":
                anomaly_dict[col_name]["缺失"].add(sn)
            elif rec["异常类型"] == "分数异常":
                anomaly_dict[col_name]["分数异常"].add(sn)
    for xml_file in xml_files:
            # 读取全量 XML 数据，不过滤任何行
        df = read_full_xml(xml_file)
        if df.empty:
            continue

        sample_nums = pd.to_numeric(df["xml_row"], errors="coerce").values #add
        numeric_cols = [col for col in df.columns if col not in ["Sample No", "Time(ms)"]]

        try:
            tree = etree.parse(xml_file)
            threshold_xpath = '//ss:Worksheet[@ss:Name="温控血流"]/ss:Table/ss:Row[2]/ss:Cell[3]/ss:Data/text()'
            threshold = int(safe_strip(tree.xpath(threshold_xpath, namespaces=NS)[0]))
        except Exception as e:
            print(f"{csv_file.stem}: 提取阈值出错：{e}")
            threshold = None

        for col in numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce").values
            indices = sample_nums  # 横坐标使用 sample_nums（全部数据点）

            sample_nums = pd.to_numeric(df["xml_row"], errors="coerce").values
            indices = np.arange(len(values))
            missing_set = anomaly_dict.get(col, {}).get("缺失", set())
            abnormal_set = anomaly_dict.get(col, {}).get("分数异常", set())
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
                plt.scatter(indices[normal_idx], normal_vals, c='black', s=5, label='正常值')
            if missing_idx:
                plt.scatter(indices[missing_idx], missing_vals, c='red', s=5, label='缺失值')
            if abnormal_idx:
                plt.scatter(indices[abnormal_idx], abnormal_vals, c='blue', s=5, label='分数异常')
            plt.xlabel("Sample No", fontsize=14)
            plt.ylabel("数值", fontsize=14)
            plt.title(f"{csv_file.stem} - {col} （共 {len(values)} 个点）", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5)

            if threshold is not None: #add
                plt.axvline(x=threshold, color='red', linestyle='dashed', linewidth=1, label="阈值")
            plt.legend(fontsize=12)

            plt.tight_layout()
            output_svg = Path.home() / "Desktop" /'plot'/ f"{csv_file.stem}_{col}_scatter.svg"
            plt.savefig(output_svg, format="svg", dpi=300)
            plt.close()
            print(f"已生成矢量图：{output_svg}")



if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    print("开始数据分析流程...")
    results = []
    for xml_file in XML_DIR.glob("*.xml"):
        print(f"正在处理：{xml_file.name}")
        results.append(parse_xml(xml_file))
    generate_summary_report(results)
    #generate_all_vector_scatter_full()
    generate_all_vector_scatter_full_after_th()
    print("\n处理完成！输出位置：")
    print(f"清洗数据：{OUTPUT_DIR}")
    print(f"异常报告：{REPORT_DIR}")
    print(f"总报告：{Path.home()}/Desktop/多普勒数据验证总报告.txt")
    #print(f"可视化：{Path.home()}/Desktop/数据缺失值和有效率可视化.png")
