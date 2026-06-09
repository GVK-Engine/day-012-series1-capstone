# capstone_video.py
# Series 1 Perception Capstone Video
#
# Shows every image and GIF from all 11 projects.
# Each project gets a title card then all its
# visual outputs displayed one by one.
#
# Output: D:\day-012-series1-capstone\series1_capstone.mp4
#
# Vamshikrishna Gadde
# MS Robotics and Autonomous Systems, ASU

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import imageio.v2 as imageio
from pathlib import Path

OUT_DIR = Path(r"D:\day-012-series1-capstone")
OUT_DIR.mkdir(exist_ok=True)

AUTHOR = "Vamshikrishna Gadde  |  MS Robotics and Autonomous Systems, ASU"
W, H   = 1920, 1080
FPS    = 24

# all images/gifs per project in display order
PROJECTS = [
    {
        "day":     "Day 01",
        "title":   "LiDAR Obstacle Detection",
        "finding": "28 objects detected in 86ms\nVoxel-based safety boundary on real KITTI data",
        "stats":   ["86ms pipeline", "28 objects", "KITTI HDL-64E"],
        "color":   "#00C8FF",
        "visuals": [
            r"C:\Users\vamsh\Downloads\day-001-lidar-obstacle-detection\ScreenCapture_2026-05-15-21-57-10.png",
            r"C:\Users\vamsh\Downloads\day-001-lidar-obstacle-detection\ScreenCapture_2026-05-15-22-02-26.png",
            r"C:\Users\vamsh\Downloads\day-001-lidar-obstacle-detection\ScreenCapture_2026-05-15-22-02-34.png",
        ],
    },
    {
        "day":     "Day 02",
        "title":   "Stereo Camera Depth Safety",
        "finding": "MAE 1.04m at 0-10m range\nCamera becomes unreliable beyond 35m",
        "stats":   ["MAE 1.04m", "unsafe >35m", "KITTI stereo"],
        "color":   "#4FC3F7",
        "visuals": [
            r"D:\day-002-stereo-depth-analysis\results\depth_analysis_frame0001.png",
        ],
    },
    {
        "day":     "Day 03",
        "title":   "PointPillars 3D Detector",
        "finding": "98.9% training loss reduction from scratch\n2.5M parameter detector on KITTI",
        "stats":   ["2.5M params", "98.9% loss drop", "KITTI"],
        "color":   "#00E676",
        "visuals": [
            r"D:\day-003-pointpillars-3d-detector\results\training_loss_curves.png",
        ],
    },
    {
        "day":     "Day 04",
        "title":   "Multi-Camera BEV Perception",
        "finding": "178 objects from 6 cameras simultaneously\nBird's eye view fusion on nuScenes Singapore",
        "stats":   ["178 objects", "6 cameras", "nuScenes"],
        "color":   "#FFD54F",
        "visuals": [
            r"D:\day-004-bev-perception\results\bev_perception_demo.png",
            r"D:\day-004-bev-perception\results\benchmark_results.png",
            r"D:\day-004-bev-perception\results\nuscenes_bev_sample00.png",
            r"D:\day-004-bev-perception\results\nuscenes_bev_sample01.png",
            r"D:\day-004-bev-perception\results\nuscenes_bev_sample02.png",
            r"D:\day-004-bev-perception\results\nuscenes_bev_sample03.png",
            r"D:\day-004-bev-perception\results\nuscenes_bev_sample04.png",
            r"D:\day-004-bev-perception\results\bev_full_sample00.png",
            r"D:\day-004-bev-perception\results\bev_full_sample01.png",
            r"D:\day-004-bev-perception\results\bev_full_sample02.png",
        ],
    },
    {
        "day":     "Day 05",
        "title":   "Multi-Object Tracking SORT",
        "finding": "95.1% MOTP beats paper baseline 77.5%\nDetector is the bottleneck, not the tracker",
        "stats":   ["95.1% MOTP", "1158 FPS", "beats paper"],
        "color":   "#FF8A65",
        "visuals": [
            r"D:\day-005-multi-object-tracking\results\evaluation_results.png",
            r"D:\day-005-multi-object-tracking\results\sort_tracking_demo.png",
            r"D:\day-005-multi-object-tracking\results\tracking_visualization.png",
        ],
    },
    {
        "day":     "Day 06",
        "title":   "Semantic Segmentation ROS2",
        "finding": "52.6 FPS on RTX 4050 in live ROS2 pipeline\nDeepLabV3 with warmup cost measured",
        "stats":   ["52.6 FPS", "DeepLabV3", "ROS2 Humble"],
        "color":   "#CE93D8",
        "visuals": [
            r"D:\day-006-semantic-segmentation\results\segmentation_frame0001.png",
            r"D:\day-006-semantic-segmentation\results\segmentation_frame0005.png",
            r"D:\day-006-semantic-segmentation\results\segmentation_frame0010.png",
            r"D:\day-006-semantic-segmentation\results\evaluation_chart.png",
            r"D:\day-006-semantic-segmentation\results\segmentation_benchmark.png",
        ],
    },
    {
        "day":     "Day 07",
        "title":   "Adverse Weather Perception",
        "finding": "Rain safe at 100mm/hr\nFog unsafe below 75m visibility, 42 conditions tested",
        "stats":   ["42 conditions", "75m boundary", "KITTI"],
        "color":   "#EF5350",
        "visuals": [
            r"D:\day-007-adverse-weather-perception\results\safety_heatmap.png",
            r"D:\day-007-adverse-weather-perception\results\weather_benchmark.png",
            r"D:\day-007-adverse-weather-perception\results\pointcloud_comparison.png",
            r"D:\day-007-adverse-weather-perception\results\range_analysis.png",
            r"D:\day-007-adverse-weather-perception\results\weather_degradation.gif",
        ],
    },
    {
        "day":     "Day 08",
        "title":   "LiDAR-Camera Depth Completion",
        "finding": "44x MAE improvement at 0-10m range\n100% coverage from sparse LiDAR input",
        "stats":   ["44x improvement", "100% coverage", "108 frames"],
        "color":   "#26C6DA",
        "visuals": [
            r"D:\day-008-depth-completion\results\depth_comparison.png",
            r"D:\day-008-depth-completion\results\sparse_depth.png",
            r"D:\day-008-depth-completion\results\dense_depth.png",
            r"D:\day-008-depth-completion\results\lidar_on_camera.png",
            r"D:\day-008-depth-completion\results\evaluation_chart.png",
            r"D:\day-008-depth-completion\results\depth_completion_demo.gif",
        ],
    },
    {
        "day":     "Day 09",
        "title":   "Domain Shift Analysis",
        "finding": "58.4% detection drop KITTI to nuScenes\nRoot cause: sensor difference, not scene",
        "stats":   ["58.4% drop", "HDL-64E vs HDL-32E", "2 datasets"],
        "color":   "#FFA726",
        "visuals": [
            r"D:\day-009-domain-shift\results\domain_shift_analysis.png",
            r"D:\day-009-domain-shift\results\bev_comparison.png",
            r"D:\day-009-domain-shift\results\comparison_frame.png",
            r"D:\day-009-domain-shift\results\evaluation_chart.png",
            r"D:\day-009-domain-shift\results\domain_shift_demo.gif",
        ],
    },
    {
        "day":     "Day 10",
        "title":   "Neural Occupancy Network",
        "finding": "Unsafe planning boundary found at 40m\n4-state uncertainty, 3D CNN 100% accuracy",
        "stats":   ["282ms pipeline", "100% accuracy", "unsafe at 40m"],
        "color":   "#AB47BC",
        "visuals": [
            r"D:\day-010-occupancy-network\results\confidence_decay.png",
            r"D:\day-010-occupancy-network\results\uncertainty_heatmap.png",
            r"D:\day-010-occupancy-network\results\resolution_tradeoff.png",
            r"D:\day-010-occupancy-network\results\occupancy_3d.gif",
            r"D:\day-010-occupancy-network\results\demo_video.gif",
        ],
    },
    {
        "day":     "Day 11",
        "title":   "ASU Campus Perception",
        "finding": "Model scaling cannot fix domain shift\nYOLOv8x worse than nano under Arizona sun",
        "stats":   ["56.5% glare drop", "2565 frames", "8 scenarios"],
        "color":   "#FF7043",
        "visuals": [
            r"D:\day-011-asu-perception\exp1_vocabulary_gap.png",
            r"D:\day-011-asu-perception\exp2_glare_analysis.png",
            r"D:\day-011-asu-perception\exp3_distance_failure.png",
            r"D:\day-011-asu-perception\exp4_misclassification.png",
            r"D:\day-011-asu-perception\exp5_model_comparison.png",
            r"D:\day-011-asu-perception\failure_highlights.gif",
            r"D:\day-011-asu-perception\germany_vs_asu.gif",
        ],
    },
]


def fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(),
                        dtype=np.uint8)
    w_, h_ = fig.canvas.get_width_height()
    buf = buf.reshape(h_, w_, 4)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    bgr = cv2.resize(bgr, (W, H))
    plt.close(fig)
    return bgr


def make_fig(bg='#0d0d0d'):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(bg)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(bg)
    return fig, ax


def render_intro():
    fig, ax = make_fig()

    ax.add_patch(FancyBboxPatch(
        (0, 0.955), 1, 0.045,
        boxstyle="square,pad=0",
        facecolor='#00C8FF', alpha=0.9,
        transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 0.045,
        boxstyle="square,pad=0",
        facecolor='#00C8FF', alpha=0.9,
        transform=ax.transAxes))

    ax.text(0.5, 0.72, "Series 1: Perception",
            transform=ax.transAxes,
            color='#00C8FF', fontsize=76,
            fontweight='bold',
            ha='center', va='center')

    ax.axhline(y=0.60, xmin=0.2, xmax=0.8,
               color='#00C8FF', alpha=0.4,
               linewidth=2)

    ax.text(0.5, 0.50,
            "11 Projects  |  Real Data  |  Real Findings",
            transform=ax.transAxes,
            color='white', fontsize=36,
            ha='center', va='center')

    ax.text(0.5, 0.38,
            "LiDAR  |  Camera  |  Fusion  |  "
            "Deep Learning  |  Domain Shift",
            transform=ax.transAxes,
            color='#777777', fontsize=24,
            ha='center', va='center')

    ax.text(0.5, 0.22, AUTHOR,
            transform=ax.transAxes,
            color='#aaaaaa', fontsize=26,
            ha='center', va='center')

    return fig_to_bgr(fig)


def render_title_card(proj, idx, total):
    fig, ax = make_fig()
    color   = proj["color"]

    # top accent
    ax.add_patch(FancyBboxPatch(
        (0, 0.955), 1, 0.045,
        boxstyle="square,pad=0",
        facecolor=color, alpha=0.9,
        transform=ax.transAxes, zorder=3))

    # left bar
    ax.add_patch(FancyBboxPatch(
        (0.038, 0.10), 0.007, 0.76,
        boxstyle="square,pad=0",
        facecolor=color, alpha=0.8,
        transform=ax.transAxes, zorder=3))

    # day badge
    ax.add_patch(FancyBboxPatch(
        (0.07, 0.775), 0.15, 0.105,
        boxstyle="round,pad=0.01",
        facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=1.5,
        transform=ax.transAxes, zorder=3))
    ax.text(0.145, 0.83, proj["day"],
            transform=ax.transAxes,
            color=color, fontsize=30,
            fontweight='bold',
            ha='center', va='center', zorder=4)

    # title
    ax.text(0.07, 0.67, proj["title"],
            transform=ax.transAxes,
            color='white', fontsize=46,
            fontweight='bold', va='center',
            zorder=4)

    # divider
    ax.axhline(y=0.575, xmin=0.07, xmax=0.93,
               color=color, alpha=0.3,
               linewidth=1.5)

    # finding
    lines = proj["finding"].split("\n")
    for i, line in enumerate(lines):
        ax.text(0.07, 0.495 - i * 0.095,
                line,
                transform=ax.transAxes,
                color='#cccccc', fontsize=28,
                va='center', zorder=4)

    # stats pills
    pill_x = 0.07
    for stat in proj["stats"]:
        pw = max(len(stat) * 0.010 + 0.02, 0.12)
        ax.add_patch(FancyBboxPatch(
            (pill_x, 0.235), pw, 0.065,
            boxstyle="round,pad=0.01",
            facecolor='#111122',
            edgecolor=color, linewidth=1.3,
            transform=ax.transAxes, zorder=3))
        ax.text(pill_x + pw/2, 0.268, stat,
                transform=ax.transAxes,
                color=color, fontsize=19,
                fontweight='bold',
                ha='center', va='center',
                zorder=4)
        pill_x += pw + 0.022

    # visual count
    n_vis = len([v for v in proj["visuals"]
                 if os.path.exists(v)])
    ax.text(0.07, 0.14,
            f"{n_vis} visual outputs in this project",
            transform=ax.transAxes,
            color='#555555', fontsize=18,
            va='center', zorder=4)

    # progress bar
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 0.013,
        boxstyle="square,pad=0",
        facecolor='#1a1a1a',
        transform=ax.transAxes, zorder=3))
    progress = (idx + 0.4) / total
    ax.add_patch(FancyBboxPatch(
        (0, 0), progress, 0.013,
        boxstyle="square,pad=0",
        facecolor=color, alpha=0.85,
        transform=ax.transAxes, zorder=4))

    # footer
    ax.text(0.93, 0.055, AUTHOR,
            transform=ax.transAxes,
            color='#444444', fontsize=14,
            ha='right', va='center', zorder=4)
    ax.text(0.07, 0.055,
            f"Series 1  |  {idx+1} of {total}",
            transform=ax.transAxes,
            color='#444444', fontsize=14,
            ha='left', va='center', zorder=4)

    return fig_to_bgr(fig)


def render_outro():
    fig, ax = make_fig()

    ax.text(0.5, 0.93,
            "Series 1: Perception Complete",
            transform=ax.transAxes,
            color='#00C8FF', fontsize=42,
            fontweight='bold',
            ha='center', va='center')
    ax.axhline(y=0.875, xmin=0.04, xmax=0.96,
               color='#00C8FF', alpha=0.35,
               linewidth=1.5)

    rows = [
        ("Day 01", "LiDAR Obstacle Detection",
         "86ms  |  28 objects",           "#00C8FF"),
        ("Day 02", "Stereo Camera Depth",
         "MAE 1.04m  |  unsafe >35m",     "#4FC3F7"),
        ("Day 03", "PointPillars 3D Detector",
         "98.9% loss reduction",           "#00E676"),
        ("Day 04", "Multi-Camera BEV",
         "178 objects  |  6 cameras",      "#FFD54F"),
        ("Day 05", "Multi-Object Tracking",
         "95.1% MOTP  |  beats paper",    "#FF8A65"),
        ("Day 06", "Semantic Segmentation",
         "52.6 FPS  |  ROS2 Humble",      "#CE93D8"),
        ("Day 07", "Adverse Weather",
         "Rain safe  |  fog unsafe <75m",  "#EF5350"),
        ("Day 08", "Depth Completion",
         "44x MAE improvement",            "#26C6DA"),
        ("Day 09", "Domain Shift",
         "58.4% drop  |  sensor root cause","#FFA726"),
        ("Day 10", "Occupancy Network",
         "Unsafe boundary at 40m",         "#AB47BC"),
        ("Day 11", "ASU Campus Perception",
         "Scaling cannot fix domain shift", "#FF7043"),
    ]

    row_h = 0.073
    y0    = 0.835
    for i, (day, title, stat, col) in \
            enumerate(rows):
        y = y0 - i * row_h
        ax.text(0.04, y, day,
                transform=ax.transAxes,
                color=col, fontsize=17,
                fontweight='bold', va='center')
        ax.text(0.135, y, title,
                transform=ax.transAxes,
                color='white', fontsize=17,
                va='center')
        ax.text(0.50, y, stat,
                transform=ax.transAxes,
                color='#aaaaaa', fontsize=15,
                va='center')
        ax.text(0.94, y, "complete",
                transform=ax.transAxes,
                color=col, fontsize=14,
                va='center', ha='right')
        if i < len(rows) - 1:
            ax.axhline(y=y - row_h*0.44,
                       xmin=0.04, xmax=0.96,
                       color='#1e1e1e',
                       linewidth=0.8)

    ax.axhline(y=0.055, xmin=0.04, xmax=0.96,
               color='#2a2a2a', linewidth=1)
    ax.text(0.5, 0.028,
            AUTHOR + "  |  github.com/GVK-Engine",
            transform=ax.transAxes,
            color='#444444', fontsize=16,
            ha='center', va='center')

    return fig_to_bgr(fig)


def load_visual(path):
    if not os.path.exists(path):
        return None, 'missing'
    ext = Path(path).suffix.lower()
    if ext == '.gif':
        try:
            frames = imageio.mimread(
                path, memtest=False)
            result = []
            for f in frames:
                if len(f.shape) == 2:
                    f = cv2.cvtColor(
                        f, cv2.COLOR_GRAY2RGB)
                if f.shape[2] == 4:
                    f = f[:, :, :3]
                result.append(cv2.cvtColor(
                    f, cv2.COLOR_RGB2BGR))
            return result, 'gif'
        except Exception:
            return None, 'missing'
    else:
        img = cv2.imread(path)
        if img is not None:
            return img, 'image'
        return None, 'missing'


def fit_to_area(img, tw, th):
    h, w   = img.shape[:2]
    scale  = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    img    = cv2.resize(img, (nw, nh))
    canvas = np.zeros((th, tw, 3),
                      dtype=np.uint8)
    canvas[:] = (13, 13, 13)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = img
    return canvas


def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0,2,4))
    return (b, g, r)


def add_hud(frame, proj, idx, total,
            vis_name="", vis_idx=0, vis_total=0):
    out   = frame.copy()
    color = hex_to_bgr(proj["color"])
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # top bar
    cv2.rectangle(out, (0, 0), (W, 58),
                  (10, 10, 10), -1)
    cv2.rectangle(out, (0, 0), (W, 5),
                  color, -1)

    top_text = f"{proj['day']}  |  {proj['title']}"
    cv2.putText(out, top_text, (18, 40),
                font, 0.85, color, 2,
                cv2.LINE_AA)

    if vis_total > 0:
        counter = f"{vis_idx+1}/{vis_total}"
        cv2.putText(out, counter,
                    (W - 160, 40),
                    font, 0.75,
                    (120, 120, 120), 1,
                    cv2.LINE_AA)

    cv2.putText(out, AUTHOR,
                (W - 780, 40),
                font, 0.50,
                (80, 80, 80), 1,
                cv2.LINE_AA)

    # bottom bar
    cv2.rectangle(out, (0, H-72),
                  (W, H), (0, 0, 0), -1)
    lines = proj["finding"].split("\n")
    cv2.putText(out, lines[0],
                (18, H-44), font,
                0.78, color, 2, cv2.LINE_AA)
    if len(lines) > 1:
        cv2.putText(out, lines[1],
                    (18, H-14), font,
                    0.62, (150, 150, 150),
                    1, cv2.LINE_AA)

    # progress bar
    prog_w = int(W * (idx + 1) / total)
    cv2.rectangle(out, (0, H-4),
                  (W, H), (25, 25, 25), -1)
    cv2.rectangle(out, (0, H-4),
                  (prog_w, H), color, -1)

    return out


def write_n(writer, frame, n):
    for _ in range(n):
        writer.write(frame)


if __name__ == "__main__":
    out_path = str(OUT_DIR / "series1_capstone.mp4")
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (W, H))

    VIS_H = H - 132   # image area height

    print("building Series 1 Capstone Video")
    print(f"output: {out_path}\n")

    # intro
    print("intro...")
    write_n(writer, render_intro(), FPS * 4)

    for pi, proj in enumerate(PROJECTS):
        print(f"[{pi+1}/11] {proj['day']}: "
              f"{proj['title']}")

        # title card  2.5 seconds
        card = render_title_card(
            proj, pi, len(PROJECTS))
        write_n(writer, card, int(FPS * 2.5))

        # collect valid visuals
        visuals = []
        for vpath in proj["visuals"]:
            data, kind = load_visual(vpath)
            if data is not None:
                name = Path(vpath).name
                visuals.append((data, kind, name))
            else:
                print(f"  skipping: "
                      f"{Path(vpath).name}")

        if not visuals:
            write_n(writer, card, FPS * 4)
            continue

        # each visual gets equal time
        # minimum 2 seconds per image
        # GIFs get at least one full loop
        for vi, (data, kind, name) in \
                enumerate(visuals):
            print(f"  {vi+1}/{len(visuals)} "
                  f"{name}")

            if kind == 'gif':
                frames_gif = data
                # play full gif at least twice
                n_gif = max(
                    len(frames_gif) * 2,
                    FPS * 3)
                for fi in range(n_gif):
                    gf  = frames_gif[
                        fi % len(frames_gif)]
                    vis = fit_to_area(
                        gf, W, VIS_H)
                    canvas = np.zeros(
                        (H, W, 3),
                        dtype=np.uint8)
                    canvas[:] = (13, 13, 13)
                    canvas[58:58+VIS_H, :] \
                        = vis
                    canvas = add_hud(
                        canvas, proj, pi,
                        len(PROJECTS), name,
                        vi, len(visuals))
                    writer.write(canvas)
            else:
                vis    = fit_to_area(
                    data, W, VIS_H)
                canvas = np.zeros(
                    (H, W, 3), dtype=np.uint8)
                canvas[:] = (13, 13, 13)
                canvas[58:58+VIS_H, :] = vis
                canvas = add_hud(
                    canvas, proj, pi,
                    len(PROJECTS), name,
                    vi, len(visuals))
                # 3 seconds per static image
                write_n(writer, canvas, FPS*3)

    # outro
    print("\noutro...")
    write_n(writer, render_outro(), FPS * 6)

    writer.release()

    # estimate duration
    total_frames = (
        FPS * 4 +
        sum(
            int(FPS * 2.5) +
            sum(
                max(
                    len(load_visual(v)[0]) * 2
                    if load_visual(v)[1] == 'gif'
                    else FPS * 3,
                    FPS * 2
                )
                for v in p["visuals"]
                if os.path.exists(v)
            )
            for p in PROJECTS
        ) +
        FPS * 6
    )
    dur = total_frames // FPS
    print(f"\nsaved: {out_path}")
    print(f"approx duration: "
          f"{dur//60}m {dur%60}s")