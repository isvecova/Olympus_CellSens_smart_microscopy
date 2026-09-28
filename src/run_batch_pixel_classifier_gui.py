"""Simple GUI for reviewing multiple z-stacks and writing one CellSens XML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from smart_acquisition.batch_workflow import (
    BatchWorkflowConfig,
    process_zstack_to_target_plan,
)
from smart_acquisition.cellsens.target_plan_io import write_combined_cellsens_xml
from smart_acquisition.planning_modes import (
    DEFAULT_PLANNING_MODE,
    GROUPING_CHOICE_LABELS,
    OPTIMIZATION_CHOICE_LABELS,
    OUTPUT_CHOICE_LABELS,
    Z_CHOICE_LABELS,
    get_planning_strategy,
    planning_choices_for_mode,
    planning_mode_from_choices,
)


DEFAULT_RF_MODEL_PATH = Path(r"L:\0_Service\SpinSR10_autoDetect\napari_models\v3_RF_normalized_background_to_muscle_peak.joblib")


@dataclass(frozen=True)
class GuiRequest:
    """Validated user choices collected from the Tk setup window."""

    template_xml: Path
    model_path: Path
    output_dir: Path
    output_xml: Path
    zstack_paths: tuple[Path, ...]
    config: BatchWorkflowConfig


def main() -> None:
    """Collect GUI settings, process all selected z-stacks, and write XML."""

    request = _collect_request()
    if request is None:
        return

    try:
        plans = []
        for index, zstack_path in enumerate(request.zstack_paths, start=1):
            print(f"Processing {index}/{len(request.zstack_paths)}: {zstack_path}")
            result = process_zstack_to_target_plan(zstack_path, request.config)
            plans.append(result.plan)
            print(f"Saved target plan: {result.plan_path}")

        write_combined_cellsens_xml(
            template_xml=request.template_xml,
            output_xml=request.output_xml,
            plans=plans,
        )
    except Exception as exc:
        _show_error("Batch processing failed", exc)
        raise

    _show_info(
        "Finished",
        (
            f"Processed {len(request.zstack_paths)} z-stacks.\n\n"
            f"Combined CellSens XML:\n{request.output_xml}"
        ),
    )


def _collect_request() -> GuiRequest | None:
    """Build the setup window and return a request after the user starts it."""

    root = tk.Tk()
    root.title("Smart acquisition batch setup")
    root.geometry("780x720")

    template_var = tk.StringVar()
    model_var = tk.StringVar(value=str(DEFAULT_RF_MODEL_PATH))
    output_dir_var = tk.StringVar()
    output_xml_var = tk.StringVar()
    default_output, default_grouping, default_z, default_optimization = (
        planning_choices_for_mode(DEFAULT_PLANNING_MODE)
    )
    output_choice_var = tk.StringVar(value=OUTPUT_CHOICE_LABELS[default_output])
    grouping_choice_var = tk.StringVar(value=GROUPING_CHOICE_LABELS[default_grouping])
    z_choice_var = tk.StringVar(value=Z_CHOICE_LABELS[default_z])
    optimization_choice_var = tk.StringVar(
        value=OPTIMIZATION_CHOICE_LABELS[default_optimization]
    )
    boundary_mode_var = tk.StringVar(value="Alpha shape")
    mode_summary_var = tk.StringVar()
    zstacks: list[Path] = []
    result: dict[str, GuiRequest | None] = {"request": None}

    numeric_defaults = {
        "overview_pixel_size_x_um": "1.3",
        "overview_pixel_size_y_um": "1.3",
        "classification_pixel_size_um": "1.3",
        "target_image_width_px": "2304",
        "target_image_height_px": "2304",
        "target_pixel_size_x_um": "0.325",
        "target_pixel_size_y_um": "0.325",
        "target_tile_overlap_fraction": "0.1",
        "group_merge_distance_factor": "1.0",
        "max_group_z_difference_um": "10.0",
        "max_extra_tile_fraction": "0.10",
        "polygon_simplify_tolerance_um": "10.0",
        "alpha_radius_tile_fraction": "0.75",
        "min_area_px": "100",
        "max_elongation": "0.8",
        "min_size_weighted_confidence": "10.0",
        "max_selected_objects": "",
    }
    numeric_vars = {
        key: tk.StringVar(value=value) for key, value in numeric_defaults.items()
    }
    use_napari_var = tk.BooleanVar(value=True)
    preview_var = tk.BooleanVar(value=True)
    ignore_zero_var = tk.BooleanVar(value=True)
    use_cache_var = tk.BooleanVar(value=True)
    write_cache_var = tk.BooleanVar(value=True)

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(1, weight=1)

    row = 0
    _path_row(
        main_frame,
        row,
        "CellSens template",
        template_var,
        lambda: _choose_file(template_var, [("CellSens XML", "*.xml"), ("All files", "*.*")]),
    )
    row += 1
    _path_row(
        main_frame,
        row,
        "RF model",
        model_var,
        lambda: _choose_file(model_var, [("Joblib model", "*.joblib"), ("All files", "*.*")]),
    )
    row += 1
    _path_row(
        main_frame,
        row,
        "Output folder",
        output_dir_var,
        lambda: _choose_directory(output_dir_var, output_xml_var),
    )
    row += 1
    _path_row(
        main_frame,
        row,
        "Output XML",
        output_xml_var,
        lambda: _choose_save_xml(output_xml_var),
    )
    row += 1

    ttk.Label(main_frame, text="Z-stacks").grid(row=row, column=0, sticky="nw", pady=4)
    zstack_list = tk.Listbox(main_frame, height=8)
    zstack_list.grid(row=row, column=1, sticky="nsew", pady=4)
    zstack_buttons = ttk.Frame(main_frame)
    zstack_buttons.grid(row=row, column=2, sticky="nw", padx=(8, 0), pady=4)
    ttk.Button(
        zstack_buttons,
        text="Add",
        command=lambda: _add_zstacks(zstacks, zstack_list),
    ).pack(fill=tk.X)
    ttk.Button(
        zstack_buttons,
        text="Remove",
        command=lambda: _remove_selected_zstacks(zstacks, zstack_list),
    ).pack(fill=tk.X, pady=(4, 0))
    row += 1

    planning_frame = ttk.LabelFrame(main_frame, text="Planning", padding=8)
    planning_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 4))
    planning_frame.columnconfigure(1, weight=1)
    planning_frame.columnconfigure(3, weight=1)
    _planning_choice_row(
        planning_frame,
        0,
        "Output",
        output_choice_var,
        OUTPUT_CHOICE_LABELS,
        "What CellSens target type should be written.",
    )
    _planning_choice_row(
        planning_frame,
        1,
        "Grouping",
        grouping_choice_var,
        GROUPING_CHOICE_LABELS,
        "Whether detections are kept separate, grouped in XY, or grouped in XY and Z.",
    )
    _planning_choice_row(
        planning_frame,
        2,
        "Z handling",
        z_choice_var,
        Z_CHOICE_LABELS,
        "How focus Z is assigned to the planned targets.",
    )
    _planning_choice_row(
        planning_frame,
        3,
        "Optimization",
        optimization_choice_var,
        OPTIMIZATION_CHOICE_LABELS,
        "Direct grouping is faster; tile-count optimization can reduce empty imaging area.",
    )
    ttk.Label(planning_frame, text="Boundary").grid(row=4, column=0, sticky="w", pady=3)
    boundary_combo = ttk.Combobox(
        planning_frame,
        textvariable=boundary_mode_var,
        values=("Alpha shape", "Mask contour / convex hull"),
        state="readonly",
    )
    boundary_combo.grid(row=4, column=1, sticky="ew", padx=(4, 12), pady=3)
    _add_tooltip(
        boundary_combo,
        "Polygon modes can use an alpha shape to avoid filling large empty gaps "
        "between nearby objects. The convex-hull fallback remains available for "
        "maximum robustness.",
    )
    ttk.Label(planning_frame, textvariable=mode_summary_var).grid(
        row=5,
        column=0,
        columnspan=4,
        sticky="w",
        pady=(6, 0),
    )
    for variable in (
        output_choice_var,
        grouping_choice_var,
        z_choice_var,
        optimization_choice_var,
    ):
        variable.trace_add(
            "write",
            lambda *_args: _refresh_planning_summary(
                mode_summary_var,
                output_choice_var,
                grouping_choice_var,
                z_choice_var,
                optimization_choice_var,
            ),
        )
    _refresh_planning_summary(
        mode_summary_var,
        output_choice_var,
        grouping_choice_var,
        z_choice_var,
        optimization_choice_var,
    )
    row += 1

    settings = ttk.LabelFrame(main_frame, text="Settings", padding=8)
    settings.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 4))
    settings.columnconfigure(1, weight=1)
    settings.columnconfigure(3, weight=1)
    _settings_grid(settings, numeric_vars)
    row += 1

    checks = ttk.Frame(main_frame)
    checks.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
    for text, variable, tooltip in (
        (
            "Use napari review",
            use_napari_var,
            "Open an interactive napari review window before final target planning.",
        ),
        (
            "Show napari tile preview",
            preview_var,
            "Show planned high-mag tiles/ROIs in napari. Preview updates can be "
            "slow for very large optimized polygon plans.",
        ),
        (
            "Ignore zero pixels",
            ignore_zero_var,
            "Exclude padded zero-valued image areas from intensity normalization.",
        ),
        (
            "Use projection cache",
            use_cache_var,
            "Reuse saved z-projection and argmax-Z files when available.",
        ),
        (
            "Write projection cache",
            write_cache_var,
            "Save z-projection and argmax-Z files for faster repeated runs.",
        ),
    ):
        checkbox = ttk.Checkbutton(checks, text=text, variable=variable)
        checkbox.pack(anchor="w")
        _add_tooltip(checkbox, tooltip)
    row += 1

    actions = ttk.Frame(main_frame)
    actions.grid(row=row, column=0, columnspan=3, sticky="e", pady=(12, 0))
    ttk.Button(actions, text="Cancel", command=root.destroy).pack(side=tk.RIGHT)
    ttk.Button(
        actions,
        text="Start",
        command=lambda: _start_from_gui(
            root,
            result,
            template_var,
            model_var,
            output_dir_var,
            output_xml_var,
            output_choice_var,
            grouping_choice_var,
            z_choice_var,
            optimization_choice_var,
            boundary_mode_var,
            zstacks,
            numeric_vars,
            use_napari_var,
            preview_var,
            ignore_zero_var,
            use_cache_var,
            write_cache_var,
        ),
    ).pack(side=tk.RIGHT, padx=(0, 8))

    root.mainloop()
    return result["request"]


def _path_row(parent, row: int, label: str, variable: tk.StringVar, command) -> None:
    """Add a labeled path entry with a Browse button."""

    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
    ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
    ttk.Button(parent, text="Browse", command=command).grid(
        row=row,
        column=2,
        sticky="ew",
        padx=(8, 0),
        pady=4,
    )


class _Tooltip:
    """Tiny Tk tooltip helper used for planning-choice explanations."""

    def __init__(self, widget: tk.Widget, text: str | Callable[[], str]):
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add=True)
        widget.bind("<Leave>", self._hide, add=True)

    def _show(self, _event=None) -> None:
        self._hide()
        text = self.text() if callable(self.text) else self.text
        if not text:
            return
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self.window,
            text=text,
            justify=tk.LEFT,
            wraplength=520,
            padding=(8, 6),
            relief=tk.SOLID,
            borderwidth=1,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


def _add_tooltip(widget: tk.Widget, text: str | Callable[[], str]) -> None:
    """Attach a hover tooltip to a Tk widget."""

    _Tooltip(widget, text)


def _planning_choice_row(
    parent: tk.Widget,
    row: int,
    label: str,
    variable: tk.StringVar,
    choices: dict[str, str],
    tooltip: str,
) -> None:
    """Add one planning-choice combobox to the planning section."""

    column = 0 if row < 2 else 2
    grid_row = row if row < 2 else row - 2
    ttk.Label(parent, text=label).grid(row=grid_row, column=column, sticky="w", pady=3)
    widget = ttk.Combobox(
        parent,
        textvariable=variable,
        values=list(choices.values()),
        state="readonly",
    )
    widget.grid(
        row=grid_row,
        column=column + 1,
        sticky="ew",
        padx=(4, 12),
        pady=3,
    )
    _add_tooltip(widget, tooltip)


def _choice_key_from_label(label: str, choices: dict[str, str], choice_name: str) -> str:
    """Translate a GUI label back to the internal planning choice key."""

    labels_to_keys = {value: key for key, value in choices.items()}
    try:
        return labels_to_keys[label]
    except KeyError as exc:
        valid_labels = ", ".join(labels_to_keys)
        raise ValueError(
            f"Unknown {choice_name} '{label}'. Choose one of: {valid_labels}."
        ) from exc


def _planning_mode_key_from_choice_vars(
    output_choice_var: tk.StringVar,
    grouping_choice_var: tk.StringVar,
    z_choice_var: tk.StringVar,
    optimization_choice_var: tk.StringVar,
) -> str:
    """Build the composed planning key from the four GUI choice variables."""

    return planning_mode_from_choices(
        output_choice=_choice_key_from_label(
            output_choice_var.get(),
            OUTPUT_CHOICE_LABELS,
            "planning output",
        ),
        grouping_choice=_choice_key_from_label(
            grouping_choice_var.get(),
            GROUPING_CHOICE_LABELS,
            "grouping",
        ),
        z_choice=_choice_key_from_label(
            z_choice_var.get(),
            Z_CHOICE_LABELS,
            "Z handling",
        ),
        optimization_choice=_choice_key_from_label(
            optimization_choice_var.get(),
            OPTIMIZATION_CHOICE_LABELS,
            "optimization",
        ),
    )


def _refresh_planning_summary(
    summary_var: tk.StringVar,
    output_choice_var: tk.StringVar,
    grouping_choice_var: tk.StringVar,
    z_choice_var: tk.StringVar,
    optimization_choice_var: tk.StringVar,
) -> None:
    """Refresh the human-readable planning summary below the comboboxes."""

    try:
        mode_key = _planning_mode_key_from_choice_vars(
            output_choice_var,
            grouping_choice_var,
            z_choice_var,
            optimization_choice_var,
        )
    except ValueError as exc:
        summary_var.set(str(exc))
        return
    strategy = get_planning_strategy(mode_key)
    summary_var.set(f"Built-in mode: {strategy.label}")


def _boundary_mode_key_from_label(label: str) -> str:
    """Translate the polygon-boundary GUI label into the planner key."""

    if label == "Alpha shape":
        return "alpha_shape"
    if label == "Mask contour / convex hull":
        return "mask_contour"
    raise ValueError(
        "Unknown polygon boundary mode. Choose Alpha shape or Mask contour / convex hull."
    )


def _settings_grid(parent, numeric_vars: dict[str, tk.StringVar]) -> None:
    """Create the numeric settings grid shared by batch workflow config."""

    labels = (
        ("Overview pixel X um", "overview_pixel_size_x_um"),
        ("Overview pixel Y um", "overview_pixel_size_y_um"),
        ("Classification pixel um", "classification_pixel_size_um"),
        ("Target width px", "target_image_width_px"),
        ("Target height px", "target_image_height_px"),
        ("Target pixel X um", "target_pixel_size_x_um"),
        ("Target pixel Y um", "target_pixel_size_y_um"),
        ("Target overlap", "target_tile_overlap_fraction"),
        ("Merge distance factor", "group_merge_distance_factor"),
        ("Max group Z diff um", "max_group_z_difference_um"),
        ("Max extra tile fraction", "max_extra_tile_fraction"),
        ("Polygon simplify um", "polygon_simplify_tolerance_um"),
        ("Alpha radius tile fraction", "alpha_radius_tile_fraction"),
        ("Min area px", "min_area_px"),
        ("Max elongation", "max_elongation"),
        ("Min weighted confidence", "min_size_weighted_confidence"),
        ("Object count", "max_selected_objects"),
    )
    for index, (label, key) in enumerate(labels):
        row = index // 2
        column = (index % 2) * 2
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=numeric_vars[key], width=14).grid(
            row=row,
            column=column + 1,
            sticky="ew",
            padx=(4, 12),
            pady=3,
        )


def _choose_file(variable: tk.StringVar, filetypes) -> None:
    """Open a file picker and store the selected path in a Tk variable."""

    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        variable.set(path)


def _choose_directory(
    output_dir_var: tk.StringVar,
    output_xml_var: tk.StringVar,
) -> None:
    """Open an output-folder picker and default the XML path inside it."""

    path = filedialog.askdirectory()
    if not path:
        return
    output_dir_var.set(path)
    if not output_xml_var.get().strip():
        output_xml_var.set(str(Path(path) / "combined_cellsens_targets.xml"))


def _choose_save_xml(variable: tk.StringVar) -> None:
    """Open a save-as dialog for the combined CellSens XML path."""

    path = filedialog.asksaveasfilename(
        defaultextension=".xml",
        filetypes=[("CellSens XML", "*.xml"), ("All files", "*.*")],
    )
    if path:
        variable.set(path)


def _add_zstacks(zstacks: list[Path], listbox: tk.Listbox) -> None:
    """Add selected VSI files to the batch list without duplicates."""

    paths = filedialog.askopenfilenames(
        filetypes=[("VSI files", "*.vsi"), ("All files", "*.*")]
    )
    for path_text in paths:
        path = Path(path_text)
        if path not in zstacks:
            zstacks.append(path)
            listbox.insert(tk.END, str(path))


def _remove_selected_zstacks(zstacks: list[Path], listbox: tk.Listbox) -> None:
    """Remove selected VSI files from the batch list."""

    for index in reversed(listbox.curselection()):
        del zstacks[index]
        listbox.delete(index)


def _start_from_gui(
    root: tk.Tk,
    result: dict[str, GuiRequest | None],
    template_var: tk.StringVar,
    model_var: tk.StringVar,
    output_dir_var: tk.StringVar,
    output_xml_var: tk.StringVar,
    output_choice_var: tk.StringVar,
    grouping_choice_var: tk.StringVar,
    z_choice_var: tk.StringVar,
    optimization_choice_var: tk.StringVar,
    boundary_mode_var: tk.StringVar,
    zstacks: list[Path],
    numeric_vars: dict[str, tk.StringVar],
    use_napari_var: tk.BooleanVar,
    preview_var: tk.BooleanVar,
    ignore_zero_var: tk.BooleanVar,
    use_cache_var: tk.BooleanVar,
    write_cache_var: tk.BooleanVar,
) -> None:
    """Validate GUI values, build `GuiRequest`, and close the setup window."""

    try:
        template_xml = _required_path(template_var.get(), "CellSens template")
        model_path = _required_path(model_var.get(), "RF model")
        output_dir = _required_path(output_dir_var.get(), "Output folder")
        output_xml = _required_path(output_xml_var.get(), "Output XML")
        if not zstacks:
            raise ValueError("Select at least one z-stack.")

        config = BatchWorkflowConfig(
            model_path=model_path,
            output_dir=output_dir,
            planning_mode=_planning_mode_key_from_choice_vars(
                output_choice_var,
                grouping_choice_var,
                z_choice_var,
                optimization_choice_var,
            ),
            overview_pixel_size_x_um=_optional_float(
                numeric_vars["overview_pixel_size_x_um"].get()
            ),
            overview_pixel_size_y_um=_optional_float(
                numeric_vars["overview_pixel_size_y_um"].get()
            ),
            classification_pixel_size_um=_required_float(
                numeric_vars["classification_pixel_size_um"].get(),
                "Classification pixel um",
            ),
            target_image_width_px=_required_int(
                numeric_vars["target_image_width_px"].get(),
                "Target width px",
            ),
            target_image_height_px=_required_int(
                numeric_vars["target_image_height_px"].get(),
                "Target height px",
            ),
            target_pixel_size_x_um=_required_float(
                numeric_vars["target_pixel_size_x_um"].get(),
                "Target pixel X um",
            ),
            target_pixel_size_y_um=_required_float(
                numeric_vars["target_pixel_size_y_um"].get(),
                "Target pixel Y um",
            ),
            target_tile_overlap_fraction=_required_float(
                numeric_vars["target_tile_overlap_fraction"].get(),
                "Target overlap",
            ),
            group_merge_distance_factor=_required_float(
                numeric_vars["group_merge_distance_factor"].get(),
                "Merge distance factor",
            ),
            max_group_z_difference_um=_required_float(
                numeric_vars["max_group_z_difference_um"].get(),
                "Max group Z diff um",
            ),
            max_extra_tile_fraction=_required_float(
                numeric_vars["max_extra_tile_fraction"].get(),
                "Max extra tile fraction",
            ),
            polygon_simplify_tolerance_um=_required_float(
                numeric_vars["polygon_simplify_tolerance_um"].get(),
                "Polygon simplify um",
            ),
            polygon_boundary_mode=_boundary_mode_key_from_label(
                boundary_mode_var.get()
            ),
            alpha_radius_tile_fraction=_required_float(
                numeric_vars["alpha_radius_tile_fraction"].get(),
                "Alpha radius tile fraction",
            ),
            min_area_px=_required_int(numeric_vars["min_area_px"].get(), "Min area px"),
            max_elongation=_required_float(
                numeric_vars["max_elongation"].get(),
                "Max elongation",
            ),
            min_size_weighted_confidence=_optional_float(
                numeric_vars["min_size_weighted_confidence"].get()
            ),
            max_selected_objects=_optional_int(
                numeric_vars["max_selected_objects"].get()
            ),
            use_napari_filter=use_napari_var.get(),
            show_napari_tiling_preview=preview_var.get(),
            ignore_zero_pixels=ignore_zero_var.get(),
            use_z_projection_cache=use_cache_var.get(),
            write_z_projection_cache=write_cache_var.get(),
        )
    except Exception as exc:
        messagebox.showerror("Invalid setup", str(exc), parent=root)
        return

    result["request"] = GuiRequest(
        template_xml=template_xml,
        model_path=model_path,
        output_dir=output_dir,
        output_xml=output_xml,
        zstack_paths=tuple(zstacks),
        config=config,
    )
    root.destroy()


def _required_path(value: str, label: str) -> Path:
    """Parse a required path field."""

    text = value.strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return Path(text)


def _required_float(value: str, label: str) -> float:
    """Parse a required floating-point field with a field-specific error."""

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _optional_float(value: str) -> float | None:
    """Parse an optional floating-point field."""

    text = value.strip()
    return None if not text else float(text)


def _required_int(value: str, label: str) -> int:
    """Parse a required integer field with a field-specific error."""

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _optional_int(value: str) -> int | None:
    """Parse an optional integer field."""

    text = value.strip()
    return None if not text else int(text)


def _show_error(title: str, exc: Exception) -> None:
    """Show a Tk error dialog including the traceback."""

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, f"{exc}\n\n{traceback.format_exc()}")
    root.destroy()


def _show_info(title: str, message: str) -> None:
    """Show a Tk informational dialog."""

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()


if __name__ == "__main__":
    main()
