"""PentahoAdapter: parses .ktr and .kjb Pentaho files into a DomainModel."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from vaig.core.migration.adapters.base import SourceAdapter, SourceAdapterRegistry
from vaig.core.migration.domain import Chunk, DomainModel, DomainNode, SemanticKind

__all__ = ["PentahoAdapter"]

# Map from Pentaho <type> to (SemanticKind, config_keys)
_STEP_TYPE_MAP: dict[str, tuple[SemanticKind, list[str]]] = {
    "TableInput": ("SOURCE", ["connection", "sql"]),
    "TextFileInput": ("SOURCE", ["file_name", "separator", "enclosure", "header"]),
    "CSVInput": ("SOURCE", ["filename", "separator", "enclosure", "header_present"]),
    "ExcelInput": ("SOURCE", ["filename", "sheetname"]),
    "FileExists": ("CONTROL", ["filename"]),
    "FilterRows": ("TRANSFORM", ["condition"]),
    "MergeJoin": ("TRANSFORM", ["join_type", "keys_1", "keys_2"]),
    "MergeRows": ("TRANSFORM", ["flag_field"]),
    "Calculator": ("TRANSFORM", ["calculations"]),
    "StringOperations": ("TRANSFORM", ["operations"]),
    "SelectValues": ("TRANSFORM", ["fields", "remove_fields", "meta"]),
    "SortRows": ("TRANSFORM", ["sort_fields"]),
    "RowNormaliser": ("TRANSFORM", ["normalised_field", "field"]),
    "Denormaliser": ("TRANSFORM", ["group_fields", "denormal_fields"]),
    "DatabaseLookup": ("TRANSFORM", ["connection", "sql", "keys", "return_values"]),
    "ExecSQL": ("TRANSFORM", ["connection", "sql"]),
    "TableOutput": ("SINK", ["connection", "table", "schema", "use_batch"]),
    "TextFileOutput": ("SINK", ["file_name", "separator", "enclosure", "header"]),
    "ExcelOutput": ("SINK", ["file_name", "sheetname"]),
    "SftpPut": ("SINK", ["server_name", "user_name", "remote_directory"]),
    "SftpGet": ("SOURCE", ["server_name", "user_name", "remote_directory"]),
    "Mail": ("SINK", ["server", "destination", "subject"]),
    "Abort": ("CONTROL", ["abort_option", "always_log_rows"]),
    "SetVariables": ("CONTROL", ["fields"]),
    "GetVariable": ("CONTROL", ["fields"]),
    "WriteToLog": ("CONTROL", ["log_message"]),
    "NullIf": ("TRANSFORM", ["fields"]),
    "IfNull": ("TRANSFORM", ["fields"]),
    "Mapping": ("TRANSFORM", ["filename"]),
    "MappingInput": ("SOURCE", ["input_fields"]),
    "MappingOutput": ("SINK", ["output_fields"]),
    # Job entries
    "JobEntryTrans": ("CONTROL", ["filename", "job_entry_name"]),
    "JobEntryJob": ("CONTROL", ["filename", "job_entry_name"]),
    "JobEntryShell": ("CONTROL", ["script", "filename"]),
    "JobEntryEval": ("CONTROL", ["script"]),
    "JobEntrySpecial": ("CONTROL", []),
}


def _get_text(elem: ET.Element, tag: str, default: str = "") -> str:
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else default


def _parse_ktr(path: Path) -> tuple[list[DomainNode], list[tuple[str, str]], dict[str, str]]:
    """Parse a .ktr transformation file."""
    nodes: list[DomainNode] = []
    hops: list[tuple[str, str]] = []
    parameters: dict[str, str] = {}

    try:
        tree = ET.parse(path)  # noqa: S314
    except ET.ParseError:
        return nodes, hops, parameters

    root = tree.getroot()

    # Parameters
    for param_elem in root.findall(".//parameters/parameter"):
        name = _get_text(param_elem, "name")
        default = _get_text(param_elem, "default_value")
        if name:
            parameters[name] = default

    # Steps
    for step_elem in root.findall(".//step"):
        step_name = _get_text(step_elem, "name", "unnamed")
        step_type = _get_text(step_elem, "type", "Unknown")

        # Find line number from XML position (ElementTree doesn't track it; use 0 as fallback)
        semantic_kind, config_keys = _STEP_TYPE_MAP.get(step_type, ("UNKNOWN", []))

        config: dict[str, Any] = {"raw_type": step_type}
        for key in config_keys:
            val = step_elem.findtext(key) or step_elem.findtext(f".//{key}") or ""
            if val:
                config[key] = val.strip()

        nodes.append(
            DomainNode(
                step_name=step_name,
                step_type=step_type,
                semantic_kind=semantic_kind,
                config=config,
                source_file=str(path),
                source_line=0,  # ElementTree doesn't track line numbers
            )
        )

    # Hops
    for hop_elem in root.findall(".//hop"):
        from_step = _get_text(hop_elem, "from")
        to_step = _get_text(hop_elem, "to")
        if from_step and to_step:
            hops.append((from_step, to_step))

    return nodes, hops, parameters


def _parse_kjb(path: Path) -> tuple[list[DomainNode], list[tuple[str, str]], dict[str, str]]:
    """Parse a .kjb job file."""
    nodes: list[DomainNode] = []
    hops: list[tuple[str, str]] = []
    parameters: dict[str, str] = {}

    try:
        tree = ET.parse(path)  # noqa: S314
    except ET.ParseError:
        return nodes, hops, parameters

    root = tree.getroot()

    # Parameters
    for param_elem in root.findall(".//parameters/parameter"):
        name = _get_text(param_elem, "name")
        default = _get_text(param_elem, "default_value")
        if name:
            parameters[name] = default

    # Entries
    for entry_elem in root.findall(".//entry"):
        entry_name = _get_text(entry_elem, "name", "unnamed")
        entry_type = _get_text(entry_elem, "type", "Unknown")

        semantic_kind, config_keys = _STEP_TYPE_MAP.get(entry_type, ("CONTROL", []))

        config: dict[str, Any] = {"raw_type": entry_type}
        for key in config_keys:
            val = entry_elem.findtext(key) or ""
            if val:
                config[key] = val.strip()

        nodes.append(
            DomainNode(
                step_name=entry_name,
                step_type=entry_type,
                semantic_kind=semantic_kind,
                config=config,
                source_file=str(path),
                source_line=0,
            )
        )

    # Hops
    for hop_elem in root.findall(".//hop"):
        from_entry = _get_text(hop_elem, "from")
        to_entry = _get_text(hop_elem, "to")
        if from_entry and to_entry:
            hops.append((from_entry, to_entry))

    return nodes, hops, parameters


class PentahoAdapter(SourceAdapter):
    """Adapter for Pentaho .ktr (transformation) and .kjb (job) files."""

    kind = "pentaho"
    file_globs = ("**/*.ktr", "**/*.kjb")

    def detect(self, paths: Sequence[Path]) -> float:
        all_files: list[Path] = []
        for p in paths:
            if p.is_dir():
                all_files.extend(p.rglob("*"))
            elif p.is_file():
                all_files.append(p)
        pentaho_files = [f for f in all_files if f.suffix.lower() in {".ktr", ".kjb"}]
        total_source = [f for f in all_files if f.is_file() and f.suffix.lower() in {
            ".ktr", ".kjb", ".py", ".sql", ".dtsx", ".xml", ".java"
        }]
        if not total_source:
            return 0.0
        return min(1.0, len(pentaho_files) / max(1, len(total_source)))

    def parse(self, paths: Sequence[Path]) -> DomainModel:
        all_nodes: list[DomainNode] = []
        all_hops: list[tuple[str, str]] = []
        all_params: dict[str, str] = {}
        evidence_gaps: list[str] = []

        for path in paths:
            if path.is_dir():
                source_files = list(path.rglob("*.ktr")) + list(path.rglob("*.kjb"))
            else:
                source_files = [path]

            for f in source_files:
                if f.suffix.lower() == ".ktr":
                    nodes, hops, params = _parse_ktr(f)
                elif f.suffix.lower() == ".kjb":
                    nodes, hops, params = _parse_kjb(f)
                else:
                    continue

                # Flag unknown step types
                for node in nodes:
                    if node.step_type not in _STEP_TYPE_MAP:
                        gap = f"Unknown step type '{node.step_type}' in {f.name} (step: {node.step_name})"
                        if gap not in evidence_gaps:
                            evidence_gaps.append(gap)

                all_nodes.extend(nodes)
                all_hops.extend(hops)
                all_params.update(params)

        return DomainModel(
            source_kind="pentaho",
            nodes=all_nodes,
            hops=all_hops,
            parameters=all_params,
            evidence_gaps=evidence_gaps,
        )

    def chunk(self, domain: DomainModel) -> Iterable[Chunk]:
        for node in domain.nodes:
            semantic_kind_lower = node.semantic_kind.lower()
            text = (
                f"Pentaho {node.step_type} step '{node.step_name}' "
                f"[{semantic_kind_lower}] from {node.source_file}"
            )
            if node.config.get("sql"):
                text += f"\nSQL: {node.config['sql'][:200]}"
            elif node.config.get("connection"):
                text += f"\nConnection: {node.config['connection']}"
            yield Chunk(
                node=node,
                text=text,
                tags=[semantic_kind_lower, node.step_type.lower(), "pentaho"],
            )


# Auto-register
SourceAdapterRegistry.register(PentahoAdapter)
