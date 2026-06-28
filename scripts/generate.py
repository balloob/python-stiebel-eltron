"""Generate the model-based Stiebel Eltron modules from the CSV maps.

Each register block in ``api/*.csv`` becomes a ``modbus_connection.model``
``Component`` of typed fields; a controller groups its components behind one
``ComponentGroup``. The CSV rows are parsed here into plain data and the Python
source is rendered from the Jinja template in ``scripts/templates/``. Run from
the repo root: ``python scripts/generate.py``.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

# Raw register value the ISG returns for an unavailable object (matches
# pystiebeleltron.UNAVAILABLE); emitted as the field ``nan`` sentinel.
UNAVAILABLE = 0x8000


@dataclass
class Columns:
    """Zero-based CSV column *indices* (they differ between WPM and LWZ exports).

    The values in these columns are strings: the ``writable`` column reads ``"r"``
    or ``"r/w"`` and the ``suffix`` column reads ``""``, ``"LOW"`` or ``"HI…"``.
    """

    name_col: int
    min_col: int
    max_col: int
    data_type_col: int
    unit_col: int
    writable_col: int
    suffix_col: int


WPM_COLUMNS = Columns(name_col=1, min_col=6, max_col=7, data_type_col=8, unit_col=9, writable_col=10, suffix_col=11)
LWZ_COLUMNS = Columns(name_col=1, min_col=5, max_col=6, data_type_col=7, unit_col=8, writable_col=9, suffix_col=10)


@dataclass
class Block:
    """One register block: a CSV file mapped to a Component."""

    name: str  # e.g. "System Values"
    path: str  # relative to api/
    space: str  # "input" or "holding"
    energy: bool = False  # apply the LOW/HI + day-and-total energy handling


@dataclass
class HeatPump:
    """A controller's component layout, docstrings and per-type extras."""

    type: str  # "Wpm" / "Lwz"
    columns: Columns
    blocks: list[Block]
    module_doc: str  # the generated module's top-level docstring
    api_doc: str  # the generated API class docstring
    operating_mode: bool = False  # emit the OperatingMode enum + helpers (LWZ)
    compressor_starts: bool = False  # emit the combined compressor_starts (LWZ)


WPM = HeatPump(
    type="Wpm",
    columns=WPM_COLUMNS,
    blocks=[
        Block("System Values", "wpm_system_values.csv", "input"),
        Block("System Parameters", "wpm_system_parameters.csv", "holding"),
        Block("System State", "wpm_system_state.csv", "input"),
        Block("Energy Data", "wpm_energy_data.csv", "input", energy=True),
        Block("Power Consumption", "wpm_power_consumption.csv", "input"),
    ],
    module_doc="Modbus api for Stiebel Eltron WPM heat pumps. This file is generated. Do not modify it manually.",
    api_doc="Stiebel Eltron WPM heat pump API over a modbus_connection ModbusUnit.",
)

LWZ = HeatPump(
    type="Lwz",
    columns=LWZ_COLUMNS,
    blocks=[
        Block("System Values", "lwz_system_values.csv", "input"),
        Block("System Parameters", "lwz_system_parameters.csv", "holding"),
        Block("System State", "lwz_system_state.csv", "input"),
        Block("Energy Data", "lwz_energy_data.csv", "input", energy=True),
    ],
    module_doc="Modbus api for Stiebel Eltron LWZ integral ventilation units (Luft-Wärme-Zentrale). This file is generated. Do not modify it manually.",
    api_doc="Stiebel Eltron LWZ integral ventilation unit (Luft-Wärme-Zentrale) API over a modbus_connection ModbusUnit.",
    operating_mode=True,
    compressor_starts=True,
)


def python_name(name: str, suffix: str = "") -> str:
    """A register's enum/attribute stem, suffix-disambiguated (pre-lowercasing)."""
    result = name.strip() if suffix == "" else name + "_" + suffix.strip()
    return result.replace(" ", "_")


def attr(name: str, suffix: str = "") -> str:
    """The snake_case attribute name for a register."""
    return python_name(name, suffix).lower()


def class_name(block_name: str) -> str:
    """The Component subclass suffix for a block (e.g. 'System Values' -> 'SystemValues')."""
    return block_name.replace(" ", "")


def field_attr(block_name: str) -> str:
    """The API attribute holding a block's component (e.g. 'system_values')."""
    return block_name.lower().replace(" ", "_")


def float_or_none(value: str) -> float | None:
    """Parse a CSV min/max cell into a float, or None when it is blank."""
    value = value.strip()
    return float(value) if value else None


def _field_line(name: str, data_type: str, wire: int, unit: str, writable: bool) -> str:
    """Render a field assignment for one register row."""
    unit_arg = f', unit="{unit}"' if unit else ""
    writable_arg = ", writable=True" if writable else ""
    if data_type == "2":
        return f"gauge({wire}, 0.1, nan=UNAVAILABLE{unit_arg}{writable_arg})"
    if data_type == "7":
        return f"gauge({wire}, 0.01, nan=UNAVAILABLE{unit_arg}{writable_arg})"
    if data_type in ("6", "8"):
        return f"integer({wire}, signed=False, nan=UNAVAILABLE{unit_arg}{writable_arg})"
    raise ValueError(f"unhandled data type {data_type!r} for {name!r}")


# The two shared components from pystiebeleltron/__init__.py, as (low, high) wire
# address ranges, so they join their space's device-wide ranges.
SHARED_INPUT_RANGE = (5000, 5001)  # EnergySystemInformation
SHARED_HOLDING_RANGE = (4000, 4002)  # EnergyManagementSettings


@dataclass
class Component:
    """A parsed component: field source lines, write bounds and energy metadata."""

    block_name: str  # e.g. "System Values"
    class_suffix: str  # e.g. "SystemValues"
    space: str
    low: int = 0  # first wire address the block covers
    high: int = 0  # last wire address the block covers
    fields: list[str] = field(default_factory=list)  # "attr = factory(...)"
    bounds: list[tuple[str, float | None, float | None]] = field(default_factory=list)
    day_and_total: list[tuple[str, str, str]] = field(default_factory=list)


def _read_rows(api_path: Path, block: Block, cols: Columns) -> list[list[str]]:
    with (api_path / block.path).open() as handle:
        return list(csv.reader(handle))[1:]


def _span(rows: list[list[str]]) -> tuple[int, int]:
    """The block's (low, high) wire address range, covering every row read."""
    addresses = [int(row[0]) - 1 for row in rows]
    return min(addresses), max(addresses)


def _plain_component(block: Block, rows: list[list[str]], cols: Columns) -> Component:
    low, high = _span(rows)
    component = Component(block.name, class_name(block.name), block.space, low, high)
    seen: set[str] = set()
    for row in rows:
        name, suffix = row[cols.name_col], row[cols.suffix_col]
        wire = int(row[0]) - 1
        attribute = attr(name, suffix)
        if attribute in seen:
            raise ValueError(f"duplicate attribute {attribute!r} in {block.name}")
        seen.add(attribute)
        # The writable column is a string ("r" / "r/w"); a "w" means writable.
        writable = "w" in row[cols.writable_col]
        factory = _field_line(name, row[cols.data_type_col], wire, row[cols.unit_col], writable)
        component.fields.append(f"{attribute} = {factory}")
        if writable:
            minimum = float_or_none(row[cols.min_col])
            maximum = float_or_none(row[cols.max_col])
            if minimum is not None or maximum is not None:
                component.bounds.append((attribute, minimum, maximum))
    return component


def _energy_component(block: Block, rows: list[list[str]], cols: Columns) -> Component:
    """Energy block: LOW/HI pairs combine in a field; DAY rows gain a running total."""
    low, high = _span(rows)
    component = Component(block.name, class_name(block.name), block.space, low, high)
    seen: set[str] = set()

    def add(attribute: str, source: str) -> None:
        if attribute in seen:
            raise ValueError(f"duplicate attribute {attribute!r} in {block.name}")
        seen.add(attribute)
        component.fields.append(f"{attribute} = {source}")

    for index, row in enumerate(rows):
        name, suffix = row[cols.name_col], row[cols.suffix_col]
        wire, unit = int(row[0]) - 1, row[cols.unit_col]
        if suffix[:2] == "HI":
            continue  # consumed by the preceding LOW row's scaled_sum
        if suffix[:3] == "LOW":
            # kWh (LOW) + MWh (HI) summed into a single counter, in kWh.
            add(attr(name, suffix[3:].strip()), f'scaled_sum({wire}, (1, 1000), unit="{unit}")')
            continue
        attribute = attr(name, suffix)
        add(attribute, _field_line(name, row[cols.data_type_col], wire, unit, False))
        if name[-3:] == "DAY":
            following = rows[index + 1]
            total = attr(following[cols.name_col], following[cols.suffix_col][3:].strip())
            running = attr(name + "_AND_TOTAL", suffix)
            component.day_and_total.append((attribute, total, running))
    return component


def _ranges_const(heatpump: HeatPump, space: str) -> str:
    """The module-level range-constant name for a controller's register space."""
    return f"{heatpump.type.upper()}_{space.upper()}_RANGES"


def _ranges_by_space(components: list[Component]) -> dict[str, tuple[tuple[int, int], ...]]:
    """The device-wide readable ranges per space (block spans plus the shared blocks)."""
    shared = {"input": SHARED_INPUT_RANGE, "holding": SHARED_HOLDING_RANGE}
    spans: dict[str, set[tuple[int, int]]] = {}
    for component in components:
        spans.setdefault(component.space, set()).add((component.low, component.high))
    for space, ranges in spans.items():
        ranges.add(shared[space])
    return {space: tuple(sorted(ranges)) for space, ranges in spans.items()}


# Convenience accessors carried over from the hand-written LWZ API.
_LWZ_HELPERS = '''
    def get_current_temp(self) -> float | None:
        """Get the current room temperature."""
        return self.system_values.actual_room_t_hc1

    def get_target_temp(self) -> float | None:
        """Get the target room temperature."""
        return self.system_parameters.room_temperature_day_hk1

    async def set_target_temp(self, temp: float) -> None:
        """Set the target room temperature (day)(HC1)."""
        await self.system_parameters.write("room_temperature_day_hk1", temp)

    def get_current_humidity(self) -> float | None:
        """Get the current room humidity."""
        return self.system_values.relative_humidity_hc1

    def get_operation(self) -> OperatingMode:
        """Return the current mode of operation."""
        op_mode = self.system_parameters.operating_mode
        if op_mode is None:
            return OperatingMode.EMERGENCY_OPERATION
        return OperatingMode(int(op_mode))

    async def set_operation(self, mode: OperatingMode) -> None:
        """Set the operation mode."""
        await self.system_parameters.write("operating_mode", mode.value)

    def get_heating_status(self) -> bool:
        """Return heater status."""
        value = self.system_state.operating_status
        if value is None:
            return False
        return bool(int(value) & (1 << 2))

    def get_cooling_status(self) -> bool:
        """Cooling status."""
        value = self.system_state.operating_status
        if value is None:
            return False
        return bool(int(value) & (1 << 3))

    def get_filter_alarm_status(self) -> bool:
        """Return filter alarm."""
        value = self.system_state.operating_status
        if value is None:
            return False
        filter_mask = (1 << 8) | (1 << 12) | (1 << 13)
        return bool(int(value) & filter_mask)'''


def _environment(scripts_path: Path) -> Environment:
    """Build the Jinja environment used to render the controller modules."""
    env = Environment(
        loader=FileSystemLoader(str(scripts_path / "templates")),
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(ranges_const=_ranges_const, field_attr=field_attr)
    return env


def generate(heatpump: HeatPump, root: Path, env: Environment) -> None:
    """Parse the CSV blocks and render the controller module from the template."""
    api_path = root / "api"
    components: list[Component] = []
    for block in heatpump.blocks:
        rows = _read_rows(api_path, block, heatpump.columns)
        if block.energy:
            component = _energy_component(block, rows, heatpump.columns)
        else:
            component = _plain_component(block, rows, heatpump.columns)
        components.append(component)

    ranges = _ranges_by_space(components)
    range_lines = [f"{_ranges_const(heatpump, space)} = {ranges[space]!r}" for space in sorted(ranges)]

    output = env.get_template("module.j2").render(
        heatpump=heatpump,
        components=components,
        range_lines=range_lines,
        module_doc=heatpump.module_doc,
        api_doc=heatpump.api_doc,
        lwz_helpers=_LWZ_HELPERS,
    )
    (root / f"pystiebeleltron/{heatpump.type.lower()}.py").write_text(output)


def main() -> None:
    """Generate every controller module, then format it with ruff."""
    root = Path.cwd()
    env = _environment(Path(__file__).parent)
    paths = []
    for heatpump in (WPM, LWZ):
        generate(heatpump, root, env)
        paths.append(str(root / f"pystiebeleltron/{heatpump.type.lower()}.py"))
    subprocess.run(["ruff", "format", *paths], check=True)
    subprocess.run(["ruff", "check", "--fix", "--quiet", *paths], check=True)
    print("Done!")


if __name__ == "__main__":
    main()
