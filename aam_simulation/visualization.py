import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider


STATE_COLORS = {
    "TAXI": "0.65",
    "CLIMB": "deepskyblue",
    "CRUISE": "royalblue",
    "DESCENT": "mediumpurple",
    "EVASIVE": "darkorange",
    "HOLD": "firebrick",
    "CHARGING": "seagreen",
}

class Replay:
    def __init__(self, sim):
        self.ids = sorted(sim.uavs)
        self.frames = []
        self.events = []

        # Display local coordinates in feet, relative to the first vertiport.
        first_port = next(iter(sim.airspace.vertiports.values()))
        self.origin = np.asarray(first_port.position[:2], dtype=float).copy()

        self.ports = {
            name: np.asarray(v.position[:2]) - self.origin
            for name, v in sim.airspace.vertiports.items()
        }
        self.corridors = [
            np.array([c.start_pos[:2], c.end_pos[:2]]) - self.origin
            for c in sim.airspace.corridors
        ]

    def _xyz(self, uav):
        # Altitude is stored separately; do not use position[2].
        return [
            uav.position[0] - self.origin[0],
            uav.position[1] - self.origin[1],
            uav.altitude,
        ]

    def capture(self, sim):
        """Record one post-timestep snapshot."""
        self.frames.append({
            "time": sim.time,
            "xyz": np.array([self._xyz(sim.uavs[i]) for i in self.ids]),
            "states": [sim.uavs[i].state for i in self.ids],
            "queued": sum(
                len(v.takeoff_queue) + len(v.pending_takeoff_ids)
                for v in sim.airspace.vertiports.values()
            ),
            "conflicts": sim.total_conflicts,
            "collisions": sim.total_collisions,
            "delay": (
                sim.delay_end_time is not None
                and sim.time < sim.delay_end_time
            ),
        })

    def event(self, time, kind, u1, u2):
        """Call before evasion or collision reset changes either UAV."""
        self.events.append({
            "time": time,
            "kind": kind,
            "ids": (u1.id, u2.id),
            "xyz": np.array([self._xyz(u1), self._xyz(u2)]),
        })

    def show(self):
        if len(self.frames) < 2 or not self.ids:
            raise ValueError("Replay requires UAVs and at least two frames.")

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.subplots_adjust(bottom=0.17, top=0.86)

        for segment in self.corridors:
            ax.plot(segment[:, 0], segment[:, 1], color="0.8", zorder=0)

        for name, xy in self.ports.items():
            ax.scatter(*xy, marker="s", color="black", s=45, zorder=2)
            ax.annotate(
                name, xy, xytext=(5, 5),
                textcoords="offset points", fontsize=9,
            )

        # Include all recorded positions so off-route UAVs remain visible.
        geometry = [f["xyz"][:, :2] for f in self.frames]
        geometry += list(self.ports.values())
        geometry += self.corridors
        geometry += [e["xyz"][:, :2] for e in self.events]
        all_xy = np.vstack(geometry)
        lower = all_xy.min(axis=0)
        upper = all_xy.max(axis=0)
        pad = max(float(np.max(upper - lower)) * 0.05, 100.0)

        ax.set_xlim(lower[0] - pad, upper[0] + pad)
        ax.set_ylim(lower[1] - pad, upper[1] + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Local x (ft)")
        ax.set_ylabel("Local y (ft)")
        ax.grid(alpha=0.2)

        first_xy = self.frames[0]["xyz"][:, :2]
        dots = ax.scatter(
            first_xy[:, 0], first_xy[:, 1], s=45, zorder=4
        )
        labels = [
            ax.annotate(
                "", xy, xytext=(4, 4),
                textcoords="offset points", fontsize=7, zorder=5,
            )
            for xy in first_xy
        ]

        event_lines = LineCollection([], linewidths=2.5, zorder=3)
        ax.add_collection(event_lines)
        collision_marks = ax.scatter(
            [], [], marker="x", s=120, color="red",
            linewidths=2, zorder=6,
        )

        ax.legend(
            handles=[
                Line2D(
                    [], [], marker="o", linestyle="",
                    color=color, label=state,
                )
                for state, color in STATE_COLORS.items()
            ],
            loc="upper right",
            fontsize=8,
        )

        slider_ax = fig.add_axes([0.16, 0.08, 0.68, 0.03])
        slider = Slider(
            slider_ax, "Step", 0, len(self.frames) - 1,
            valinit=0, valstep=1, valfmt="%0.0f",
        )
        fig.text(
            0.5, 0.025,
            "Space: play/pause   |   Left/Right: one step   |   "
            ", / .: previous/next event   |   Labels: ID / altitude",
            ha="center", fontsize=9,
        )

        frame_times = np.array([f["time"] for f in self.frames])
        event_times = np.array([e["time"] for e in self.events])
        jump_times = np.unique(event_times)

        def draw(value):
            frame = self.frames[int(value)]
            xyz = frame["xyz"]
            states = frame["states"]
            now = frame["time"]

            dots.set_offsets(xyz[:, :2])
            dots.set_color([STATE_COLORS.get(s, "black") for s in states])

            for label, uid, point in zip(labels, self.ids, xyz):
                label.xy = tuple(point[:2])
                label.set_text(f"{uid} / {point[2]:.0f} ft")

            # Show events from the last few seconds, at their original locations.
            lo = np.searchsorted(event_times, now - 5, side="left")
            hi = np.searchsorted(event_times, now, side="right")
            recent = self.events[lo:hi]

            event_lines.set_segments([e["xyz"][:, :2] for e in recent])
            event_lines.set_color([
                "red" if e["kind"] == "collision" else "darkorange"
                for e in recent
            ])

            collision_xy = [
                e["xyz"][:, :2]
                for e in recent if e["kind"] == "collision"
            ]
            collision_marks.set_offsets(
                np.vstack(collision_xy)
                if collision_xy else np.empty((0, 2))
            )

            airborne = int(np.count_nonzero(xyz[:, 2] > 0))
            ground_hold = sum(
                s == "HOLD" and z <= 0
                for s, z in zip(states, xyz[:, 2])
            )
            latest = (
                f"{recent[-1]['kind']} {recent[-1]['ids']}"
                if recent else "none"
            )
            ax.set_title(
                f"Post-step t={now} s | airborne={airborne}/{len(self.ids)}"
                f" | queued={frame['queued']} | ground HOLD={ground_hold}\n"
                f"Counted conflicts={frame['conflicts']}"
                f" | collisions={frame['collisions']}"
                f" | departure delay={frame['delay']}"
                f" | recent event: {latest}",
                fontsize=10,
            )
            fig.canvas.draw_idle()

        slider.on_changed(draw)
        timer = fig.canvas.new_timer(interval=100)
        playing = False

        def advance():
            nonlocal playing
            index = int(slider.val)
            if index < len(self.frames) - 1:
                slider.set_val(index + 1)
            else:
                playing = False
                timer.stop()

        timer.add_callback(advance)

        def on_key(event):
            nonlocal playing
            index = int(slider.val)

            if event.key == " ":
                playing = not playing
                if playing:
                    timer.start()
                else:
                    timer.stop()

            elif event.key in ("left", "right"):
                delta = -1 if event.key == "left" else 1
                slider.set_val(np.clip(index + delta, 0, len(self.frames) - 1))

            elif event.key in (",", ".") and len(jump_times):
                now = frame_times[index]
                if event.key == ".":
                    j = np.searchsorted(jump_times, now, side="right")
                else:
                    j = np.searchsorted(jump_times, now, side="left") - 1

                if 0 <= j < len(jump_times):
                    target = np.searchsorted(frame_times, jump_times[j])
                    slider.set_val(min(target, len(self.frames) - 1))

        fig.canvas.mpl_connect("key_press_event", on_key)

        # Retain widget/timer references for interactive backends.
        self._ui = (fig, slider, timer)
        draw(0)
        plt.show()