"""Simple GUI for reviewing multiple z-stacks and writing one CellSens XML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from smart_acquisition.batch_workflow import (
    BatchWorkflowConfig,
    process_zstack_to_target_plan,
)
from smart_acquisition.cellsens.target_plan_io import write_combined_cellsens_xml


PLANNING_MODES = (
    "optimized_z_aware_irregular_mosaics",
    "z_aware_irregular_mosaics",
    "mixed_irregular_mosaics",
    "mixed_tilescans",
    "z_aware_tiles",
    "raw",
    "group_centers",
    "group_tiles",
)


@dataclass(frozen=True)
class GuiRequest:
    template_xml: Path
    model_path: Path
    output_dir: Path
    output_xml: Path
    zstack_paths: tuple[Path, ...]
    config: BatchWorkflowConfig


def main() -> None:
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
    root = tk.Tk()
    root.title("Smart acquisition batch setup")
    root.geometry("780x720")

    template_var = tk.StringVar()
    model_var = tk.StringVar()
    output_dir_var = tk.StringVar()
    output_xml_var = tk.StringVar()
    planning_mode_var = tk.StringVar(value=PLANNING_MODES[0])
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
        "min_area_px": "100",
        "min_size_weighted_confidence": "",
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
    estimate_z_var = tk.BooleanVar(value=True)

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

    ttk.Label(main_frame, text="Planning mode").grid(row=row, column=0, sticky="w", pady=4)
    ttk.Combobox(
        main_frame,
        textvariable=planning_mode_var,
        values=PLANNING_MODES,
        state="readonly",
    ).grid(row=row, column=1, sticky="ew", pady=4)
    row += 1

    settings = ttk.LabelFrame(main_frame, text="Settings", padding=8)
    settings.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 4))
    settings.columnconfigure(1, weight=1)
    settings.columnconfigure(3, weight=1)
    _settings_grid(settings, numeric_vars)
    row += 1

    checks = ttk.Frame(main_frame)
    checks.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
    for text, variable in (
        ("Use napari review", use_napari_var),
        ("Show napari tile preview", preview_var),
        ("Ignore zero pixels", ignore_zero_var),
        ("Use projection cache", use_cache_var),
        ("Write projection cache", write_cache_var),
        ("Estimate tile-scan Z", estimate_z_var),
    ):
        ttk.Checkbutton(checks, text=text, variable=variable).pack(anchor="w")
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
            planning_mode_var,
            zstacks,
            numeric_vars,
            use_napari_var,
            preview_var,
            ignore_zero_var,
            use_cache_var,
            write_cache_var,
            estimate_z_var,
        ),
    ).pack(side=tk.RIGHT, padx=(0, 8))

    root.mainloop()
    return result["request"]


def _path_row(parent, row: int, label: str, variable: tk.StringVar, command) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
    ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
    ttk.Button(parent, text="Browse", command=command).grid(
        row=row,
        column=2,
        sticky="ew",
        padx=(8, 0),
        pady=4,
    )


def _settings_grid(parent, numeric_vars: dict[str, tk.StringVar]) -> None:
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
        ("Min area px", "min_area_px"),
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
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        variable.set(path)


def _choose_directory(
    output_dir_var: tk.StringVar,
    output_xml_var: tk.StringVar,
) -> None:
    path = filedialog.askdirectory()
    if not path:
        return
    output_dir_var.set(path)
    if not output_xml_var.get().strip():
        output_xml_var.set(str(Path(path) / "combined_cellsens_targets.xml"))


def _choose_save_xml(variable: tk.StringVar) -> None:
    path = filedialog.asksaveasfilename(
        defaultextension=".xml",
        filetypes=[("CellSens XML", "*.xml"), ("All files", "*.*")],
    )
    if path:
        variable.set(path)


def _add_zstacks(zstacks: list[Path], listbox: tk.Listbox) -> None:
    paths = filedialog.askopenfilenames(
        filetypes=[("VSI files", "*.vsi"), ("All files", "*.*")]
    )
    for path_text in paths:
        path = Path(path_text)
        if path not in zstacks:
            zstacks.append(path)
            listbox.insert(tk.END, str(path))


def _remove_selected_zstacks(zstacks: list[Path], listbox: tk.Listbox) -> None:
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
    planning_mode_var: tk.StringVar,
    zstacks: list[Path],
    numeric_vars: dict[str, tk.StringVar],
    use_napari_var: tk.BooleanVar,
    preview_var: tk.BooleanVar,
    ignore_zero_var: tk.BooleanVar,
    use_cache_var: tk.BooleanVar,
    write_cache_var: tk.BooleanVar,
    estimate_z_var: tk.BooleanVar,
) -> None:
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
            planning_mode=planning_mode_var.get(),
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
            min_area_px=_required_int(numeric_vars["min_area_px"].get(), "Min area px"),
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
            estimate_tile_scan_z=estimate_z_var.get(),
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
    text = value.strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return Path(text)


def _required_float(value: str, label: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _optional_float(value: str) -> float | None:
    text = value.strip()
    return None if not text else float(text)


def _required_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _optional_int(value: str) -> int | None:
    text = value.strip()
    return None if not text else int(text)


def _show_error(title: str, exc: Exception) -> None:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, f"{exc}\n\n{traceback.format_exc()}")
    root.destroy()


def _show_info(title: str, message: str) -> None:
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()


if __name__ == "__main__":
    main()
