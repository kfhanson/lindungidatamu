"""An MCP server that lets an AI client ask about a table and get only gated aggregates back.

    lindungidatamu serve claims.parquet --policy claims.policy.json

speaks MCP over stdio (JSON-RPC 2.0, one message per line), so Claude Desktop, Claude Code or any
MCP client can run it. Two tools: `describe_table` and `aggregate` (see analysis.Analysis). The
rows never leave this process; the client sees what `describe` and the gate let through. The
state folder keeps the noise secret, the groups already answered and a log of every question.
"""

import json
import sys

from . import __version__
from .analysis import Analysis, QueryError

PROTOCOL = "2025-06-18"

INSTRUCTIONS = """This server answers questions about one table with aggregates only: counts of people
and rows, sums and means, grouped and filtered. It never returns rows. Call describe_table first to see
the columns and how each can be grouped. Every answer is gated: groups of fewer than k people are left
out, sums one or two people dominate are withheld, and numbers carry small noise (counts by up to 2,
sums and means by a little and rounded to 3 significant figures), so treat small differences as noise.
Groups that differ from an earlier answer by fewer than k people get that earlier answer."""

DP_INSTRUCTIONS = """This server answers questions about one table with aggregates only, under differential
privacy: every answer carries random noise and spends part of a fixed privacy budget (epsilon); when the
budget is used up, questions are refused. describe_table shows the budget left and the columns, but no
values. Plan before asking: one well-chosen question beats many small ones, and the same question asked
again returns the same answer for free.
- List the groups you want with "values" on every group_by column ({"column": "sex", "values": ["L", "P"]},
  months as "2024-01", quarters as "2024-Q1"): every listed group is shown. Without "values", only groups
  with many people appear.
- Sums and means need an "upper" bound (and "lower", default 0); values outside are clipped. Choose it from
  what you know about the column; a tighter bound means less noise.
- Each person counts in at most groups_per_person groups (default 1) and with at most rows_per_person rows
  (default 3). For a time series or anything where one person has rows in several groups, raise
  groups_per_person to the number of groups, or totals will be too low.
- Each answer states its noise (95% range). Do not read differences smaller than that."""

_GROUP = {"oneOf": [
    {"type": "string", "description": "a category column"},
    {"type": "object", "properties": {
        "column": {"type": "string"},
        "by": {"type": "string", "description": "for a date: day, week, month, quarter or year; for an area or "
                                                "diagnosis code column, a level describe_table lists"},
        "width": {"type": "number", "description": "for a number column: the bin width"},
        "values": {"type": "array", "description": "DP mode: the groups to show, every one of them"}},
     "required": ["column"]}]}
_MEASURE = {"oneOf": [
    {"type": "string", "enum": ["people", "rows"]},
    {"type": "object", "description": '{"sum": column} or {"mean": column}, a number column',
     "properties": {"sum": {"type": "string"}, "mean": {"type": "string"},
                    "upper": {"type": "number", "description": "DP mode: values above are clipped (required)"},
                    "lower": {"type": "number", "description": "DP mode: values below are clipped (default 0)"}}}]}
_WHERE = {"type": "object", "description": 'a column and one operator: {"column": "sex", "=": "P"}, '
                                          '{"column": "admitted", "between": ["2024-01-01", "2024-03-31"]}, '
                                          '{"column": "diagnosis", "in": ["E11.9", "I10"]}; operators: '
                                          '=, !=, in, not_in, <, <=, >, >=, between, missing (true/false)',
          "properties": {"column": {"type": "string"}}, "required": ["column"]}

TOOLS = [
    {"name": "describe_table",
     "description": "The table's columns that can be asked about (kind, role, the values held by enough "
                    "people, how to group each), the hidden columns and why, and the number of people.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "aggregate",
     "description": "Counts, sums or means over the table, grouped and filtered, through the privacy gate. "
                    "Returns a table and notes on what the gate left out.",
     "inputSchema": {"type": "object", "properties": {
         "group_by": {"type": "array", "items": _GROUP, "description": "columns to group by (none: one total)"},
         "measures": {"type": "array", "items": _MEASURE, "description": 'default ["people"]'},
         "where": {"type": "array", "items": _WHERE, "description": "conditions, all of which must hold"},
         "epsilon": {"type": "number", "description": "DP mode: the budget this question spends"},
         "groups_per_person": {"type": "integer", "description": "DP mode: groups one person may count in"},
         "rows_per_person": {"type": "integer", "description": "DP mode: rows one person may add to a group"}}}},
]


def handle(analysis: Analysis, msg: dict) -> dict | None:
    """One JSON-RPC message in, its response out (None for a notification)."""
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    try:
        if method == "initialize":
            result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "lindungidatamu", "version": __version__},
                      "instructions": DP_INSTRUCTIONS if analysis.budget is not None else INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = _call(analysis, msg.get("params", {}))
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    except Exception as e:      # a bad request must not stop the server
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _call(analysis: Analysis, params: dict) -> dict:
    name, args = params.get("name"), params.get("arguments") or {}
    try:
        if name == "describe_table":
            text = json.dumps(analysis.describe(), ensure_ascii=False, indent=1)
        elif name == "aggregate":
            options = {k: args[k] for k in ("epsilon", "groups_per_person", "rows_per_person") if k in args}
            answer = analysis.aggregate(group_by=args.get("group_by") or [],
                                        measures=args.get("measures") or ["people"],
                                        where=args.get("where") or [], **options)
            text = str(answer) + "\n\n" + json.dumps(answer.to_dict(), ensure_ascii=False)
        else:
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
    except QueryError as e:
        return {"content": [{"type": "text", "text": f"Not answered: {e}"}], "isError": True}
    return {"content": [{"type": "text", "text": text}], "isError": False}


def run(analysis: Analysis, stdin=None, stdout=None) -> None:
    """Serve until stdin closes."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "not JSON"}}
        else:
            reply = handle(analysis, msg) if isinstance(msg, dict) else \
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "batches aren't supported"}}
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
