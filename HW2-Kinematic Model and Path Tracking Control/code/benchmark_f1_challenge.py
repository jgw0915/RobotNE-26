"""#NewFeature: Headless benchmark utility for comparing all model/controller pairs on the three F1 challenge tracks."""
import argparse
import math
from types import SimpleNamespace

import numpy as np

from Simulation.utils import ControlState
from navigation import setup_simulator_and_controller, load_and_process_track
import PathTracking.utils as pt_utils


TRACK_REFS = {
    "Silverstone": 107.0,
    "Suzuka": 115.0,
    "Monza": 94.0,
}

K_T = 0.5
K_E = 2.0
DEFAULT_TRACKS = ["Silverstone", "Suzuka", "Monza"]
MODEL_CONTROLLER_PAIRS = [
    ("basic", "pid"),
    ("basic", "pure_pursuit"),
    ("basic", "lqr"),
    ("diff_drive", "pid"),
    ("diff_drive", "pure_pursuit"),
    ("diff_drive", "lqr"),
    ("bicycle", "pid"),
    ("bicycle", "pure_pursuit"),
    ("bicycle", "stanley"),
    ("bicycle", "lqr"),
]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def compute_track_score(track_name, elapsed_time, avg_cte):
    st = clamp(4.0 - K_T * (elapsed_time - TRACK_REFS[track_name]), 0.0, 6.0)
    se = clamp(4.0 - K_E * (avg_cte - 1.5), 0.0, 6.0)
    score = min(10.0, st + se)
    return st, se, score


def build_args(simulator, controller, track, lqr_control_state):
    return SimpleNamespace(
        simulator=simulator,
        controller=controller,
        track=track,
        lqr_control_state=lqr_control_state,
        init_shift=0.0,
    )


def compute_cte(path, nav_current_idx, state_x, state_y):
    nav_current_idx, _ = pt_utils.search_nearest_local(
        path, (state_x, state_y), nav_current_idx, lookahead=50
    )

    if nav_current_idx < len(path) - 1:
        p_i, p_i1 = path[nav_current_idx][:2], path[nav_current_idx + 1][:2]
    else:
        p_i, p_i1 = path[nav_current_idx - 1][:2], path[nav_current_idx][:2]

    vec_path = p_i1 - p_i
    vec_car = np.array([state_x, state_y]) - p_i
    path_len = np.linalg.norm(vec_path)
    if path_len > 1e-5:
        cross_2d = vec_path[0] * vec_car[1] - vec_path[1] * vec_car[0]
        cte = abs(cross_2d / path_len)
    else:
        cte = np.linalg.norm(vec_car)
    return nav_current_idx, cte


def run_once(simulator_name, controller_name, track_name, lqr_control_state, max_steps):
    args = build_args(simulator_name, controller_name, track_name, lqr_control_state)
    simulator, controller, long_controller, _ = setup_simulator_and_controller(args)
    way_points, path = load_and_process_track(track_name, 2000, 2000, simulator)

    controller.set_path(way_points)
    long_controller.set_path(way_points)

    start_yaw = np.rad2deg(np.arctan2(path[1][1] - path[0][1], path[1][0] - path[0][0]))
    start_pose = (path[0][0], path[0][1], start_yaw)
    simulator.init_pose(start_pose)
    command = ControlState(simulator_name, None, None)

    nav_current_idx = 0
    cte_history = []
    finished = False

    for step in range(max_steps):
        simulator.step(command)

        pose = (simulator.state.x, simulator.state.y, simulator.state.yaw)
        if simulator_name == "basic":
            info = {"x": pose[0], "y": pose[1], "yaw": pose[2], "v": simulator.state.v}
            next_v, _ = long_controller.feedback(info)
            next_w = controller.feedback(info)
            command = ControlState("basic", next_v, next_w)
        elif simulator_name == "diff_drive":
            info = {"x": pose[0], "y": pose[1], "yaw": pose[2], "v": simulator.state.v}
            next_v, _ = long_controller.feedback(info)
            next_w = controller.feedback(info)
            omega = np.deg2rad(next_w)
            wheel_radius = simulator.model.r
            half_wheel_distance = simulator.model.l
            next_lw = np.rad2deg((next_v - half_wheel_distance * omega) / wheel_radius)
            next_rw = np.rad2deg((next_v + half_wheel_distance * omega) / wheel_radius)
            command = ControlState("diff_drive", next_lw, next_rw)
        elif simulator_name == "bicycle":
            info = {
                "x": pose[0],
                "y": pose[1],
                "yaw": pose[2],
                "v": simulator.state.v,
                "delta": simulator.cstate.delta,
            }
            next_a, _ = long_controller.feedback(info)
            info["v"] = info["v"] + next_a * simulator.model.dt
            next_delta = controller.feedback(info)
            command = ControlState("bicycle", next_a, next_delta)
        else:
            raise ValueError(f"Unsupported simulator: {simulator_name}")

        nav_current_idx, cte = compute_cte(path, nav_current_idx, simulator.state.x, simulator.state.y)
        cte_history.append(cte)

        if nav_current_idx >= len(path) - 1:
            finished = True
            break

    elapsed_time = (step + 1) * simulator.model.dt
    avg_cte = float(np.mean(cte_history)) if cte_history else float("inf")
    return {
        "finished": finished,
        "elapsed_time": elapsed_time,
        "avg_cte": avg_cte,
        "steps": step + 1,
    }


def benchmark_combination(simulator_name, controller_name, tracks, lqr_control_state, max_steps):
    results = []
    for track_name in tracks:
        run_result = run_once(simulator_name, controller_name, track_name, lqr_control_state, max_steps)
        if run_result["finished"]:
            st, se, score = compute_track_score(track_name, run_result["elapsed_time"], run_result["avg_cte"])
        else:
            st, se, score = 0.0, 0.0, 0.0
        run_result.update({
            "track": track_name,
            "st": st,
            "se": se,
            "score": score,
        })
        results.append(run_result)

    final_score_raw = sum(result["score"] for result in results) / len(results)
    final_score = math.ceil(final_score_raw)
    return results, final_score_raw, final_score


def build_summary_text(summary_rows, tracks):
    lines = ["", "=== F1 Challenge Benchmark Summary ==="]
    track_headers = " ".join(f"{track:>12}" for track in tracks)
    header = f"{'Simulator':<12} {'Controller':<14} {'Final Score':>11} {track_headers}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in summary_rows:
        track_scores = " ".join(f"{row['track_scores'].get(track, 0.0):>12.2f}" for track in tracks)
        lines.append(
            f"{row['simulator']:<12} {row['controller']:<14} {row['final_score_raw']:>11.2f} "
            f"{track_scores}"
        )
    return "\n".join(lines)


def print_summary(summary_rows, tracks):
    print(build_summary_text(summary_rows, tracks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS, choices=DEFAULT_TRACKS)
    parser.add_argument("--max_steps", type=int, default=12000)
    parser.add_argument(
        "--lqr_control_state",
        type=str,
        default="steering_angle",
        choices=["steering_angle", "steering_angular_velocity"],
    )
    parser.add_argument("--summary_file", type=str, default="benchmark_summary.txt")
    args = parser.parse_args()

    summary_rows = []
    for simulator_name, controller_name in MODEL_CONTROLLER_PAIRS:
        print(f"\nRunning {simulator_name} + {controller_name} ...")
        try:
            track_results, final_score_raw, final_score = benchmark_combination(
                simulator_name,
                controller_name,
                args.tracks,
                args.lqr_control_state,
                args.max_steps,
            )
            track_scores = {result["track"]: result["score"] for result in track_results}
            summary_rows.append({
                "simulator": simulator_name,
                "controller": controller_name,
                "final_score_raw": final_score_raw,
                "final_score": final_score,
                "track_scores": track_scores,
            })

            for result in track_results:
                status = "finished" if result["finished"] else "timeout"
                print(
                    f"  {result['track']:<11} {status:<8} "
                    f"time={result['elapsed_time']:.2f}s "
                    f"avg_cte={result['avg_cte']:.4f} "
                    f"score={result['score']:.2f}"
                )
        except Exception as exc:
            print(f"  Failed: {exc}")
            summary_rows.append({
                "simulator": simulator_name,
                "controller": controller_name,
                "final_score_raw": 0.0,
                "final_score": 0,
                "track_scores": {track: 0.0 for track in args.tracks},
            })

    summary_rows.sort(key=lambda row: (-row["final_score_raw"], row["simulator"], row["controller"]))
    print_summary(summary_rows, args.tracks)
    with open(args.summary_file, "w", encoding="utf-8") as summary_file:
        summary_file.write(build_summary_text(summary_rows, args.tracks))
        summary_file.write("\n")
    print(f"\nSummary written to {args.summary_file}")


if __name__ == "__main__":
    main()
