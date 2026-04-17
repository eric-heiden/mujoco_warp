# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Determinism benchmark for repeated MJWarp runs under global Warp determinism mode.

This script complements determinism_test.py with timing and repeated-run bitwise
comparisons that are convenient for larger investigations and report generation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp import BroadphaseType
from mujoco_warp import test_data

_CONTACT_FIELDS = (
  "dist",
  "pos",
  "frame",
  "includemargin",
  "friction",
  "solref",
  "solreffriction",
  "solimp",
  "dim",
  "geom",
  "flex",
  "vert",
  "efc_address",
  "worldid",
  "type",
  "geomcollisionid",
)
_STATE_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "act",
  "qfrc_constraint",
  "sensordata",
)
_SCALAR_FIELDS = ("time", "nefc")
_CASE_CAPACITY_OVERRIDES = {
  ("humanoid/humanoid.xml", 4): {"njmax": 128},
}
_DEFAULT_CASES = (
  ("collision.xml", 1, 10),
  ("collision.xml", 4, 10),
  ("humanoid/humanoid.xml", 1, 10),
  ("humanoid/humanoid.xml", 4, 10),
)


def _capacity_overrides(path: str, nworld: int) -> dict[str, int]:
  return dict(_CASE_CAPACITY_OVERRIDES.get((path, nworld), {}))


def _fresh_case(path: str, nworld: int, nconmax=None, njmax=None):
  mjm, mjd, m, d = test_data.fixture(path=path, nworld=nworld)
  if nconmax is not None or njmax is not None:
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
  return m, d


def _collect_state_fields(d) -> dict[str, np.ndarray]:
  fields = {field: getattr(d, field).numpy().copy() for field in _STATE_FIELDS}
  fields.update({field: getattr(d, field).numpy().copy() for field in _SCALAR_FIELDS})
  return fields


def _collect_active_contact_fields(d) -> tuple[int, dict[str, np.ndarray]]:
  nacon = int(d.nacon.numpy()[0])
  return nacon, {field: getattr(d.contact, field).numpy()[:nacon].copy() for field in _CONTACT_FIELDS}


def _contacts_are_sorted(contact_fields: dict[str, np.ndarray]) -> bool:
  nacon = len(contact_fields["worldid"])
  for i in range(1, nacon):
    key_prev = (
      int(contact_fields["worldid"][i - 1]),
      int(contact_fields["geom"][i - 1, 0]),
      int(contact_fields["geom"][i - 1, 1]),
      int(contact_fields["geomcollisionid"][i - 1]),
    )
    key_curr = (
      int(contact_fields["worldid"][i]),
      int(contact_fields["geom"][i, 0]),
      int(contact_fields["geom"][i, 1]),
      int(contact_fields["geomcollisionid"][i]),
    )
    if key_prev > key_curr:
      return False
  return True


def _run_snapshot(
  *,
  path: str,
  nworld: int,
  nsteps: int,
  deterministic: bool,
  broadphase: BroadphaseType | None,
  graph_conditional: bool | None,
  nconmax=None,
  njmax=None,
) -> dict:
  m, d = _fresh_case(path=path, nworld=nworld, nconmax=nconmax, njmax=njmax)
  m.opt.deterministic = deterministic
  if broadphase is not None:
    m.opt.broadphase = broadphase
  if graph_conditional is not None:
    m.opt.graph_conditional = graph_conditional

  t0 = time.perf_counter()
  for _ in range(nsteps):
    mjw.step(m, d)
  wp.synchronize()
  elapsed_ms_per_step = (time.perf_counter() - t0) * 1000.0 / nsteps

  nacon, contacts = _collect_active_contact_fields(d)
  return {
    "nacon": nacon,
    "state": _collect_state_fields(d),
    "contacts": contacts,
    "ordered": _contacts_are_sorted(contacts),
    "time_ms_per_step": elapsed_ms_per_step,
  }


def _field_bytes_hex(arr: np.ndarray) -> str:
  flat = np.ascontiguousarray(arr)
  # Keep JSON readable while still exposing exact bit patterns.
  return flat.tobytes().hex()[:256]


def _summarize_snapshots(snapshots: list[dict]) -> dict:
  base = snapshots[0]
  field_equal = {field: True for field in _CONTACT_FIELDS}
  state_equal = {field: True for field in _STATE_FIELDS + _SCALAR_FIELDS}

  for other in snapshots[1:]:
    for field in state_equal:
      state_equal[field] &= np.array_equal(base["state"][field], other["state"][field])
    for field in field_equal:
      field_equal[field] &= np.array_equal(base["contacts"][field], other["contacts"][field])

  timing = [snapshot["time_ms_per_step"] for snapshot in snapshots]
  return {
    "nacon_values": [snapshot["nacon"] for snapshot in snapshots],
    "nacon_identical": len({snapshot["nacon"] for snapshot in snapshots}) == 1,
    "ordered": [snapshot["ordered"] for snapshot in snapshots],
    "bitwise": {
      "all_state_fields_bitwise_identical": all(state_equal.values()),
      "all_contact_fields_bitwise_identical": all(field_equal.values()),
      "all_fields_bitwise_identical": all(state_equal.values()) and all(field_equal.values()),
      "per_state_identical": state_equal,
      "per_contact_identical": field_equal,
    },
    "sample_metrics": {
      field: {
        "shape": list(base["contacts"][field].shape),
        "dtype": str(base["contacts"][field].dtype),
        "sum": float(np.asarray(base["contacts"][field], dtype=np.float64).sum()) if base["contacts"][field].size else 0.0,
        "bytes_hex": _field_bytes_hex(base["contacts"][field]),
      }
      for field in _CONTACT_FIELDS
    },
    "time_ms_per_step": {
      "samples": timing,
      "mean": statistics.mean(timing),
      "stdev": statistics.pstdev(timing) if len(timing) > 1 else 0.0,
    },
  }


def _parse_case(text: str) -> tuple[str, int, int]:
  path, nworld, nsteps = text.split("|")
  return path, int(nworld), int(nsteps)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--deterministic", action="store_true", default=False)
  parser.add_argument("--warp-deterministic", choices=("false", "run_to_run", "gpu_to_gpu"), default="false")
  parser.add_argument("--graph-conditional", choices=("true", "false"))
  parser.add_argument("--broadphase", choices=("nxn", "sap_tile", "sap_segmented"))
  parser.add_argument(
    "--case",
    action="append",
    help="Case specification: path|nworld|nsteps. May be passed multiple times.",
  )
  parser.add_argument("--output")
  args = parser.parse_args()

  wp.init()
  wp.config.deterministic = False if args.warp_deterministic == "false" else args.warp_deterministic

  broadphase = None
  if args.broadphase == "nxn":
    broadphase = BroadphaseType.NXN
  elif args.broadphase == "sap_tile":
    broadphase = BroadphaseType.SAP_TILE
  elif args.broadphase == "sap_segmented":
    broadphase = BroadphaseType.SAP_SEGMENTED

  graph_conditional = None
  if args.graph_conditional is not None:
    graph_conditional = args.graph_conditional == "true"

  cases = [_parse_case(case) for case in args.case] if args.case else list(_DEFAULT_CASES)

  result = {
    "warp_deterministic": str(wp.config.deterministic),
    "mjwarp_deterministic": args.deterministic,
    "graph_conditional": graph_conditional,
    "broadphase": None if broadphase is None else int(broadphase),
    "repeats": args.repeats,
    "cases": {},
  }

  for path, nworld, nsteps in cases:
    kwargs = _capacity_overrides(path, nworld)
    snapshots = [
      _run_snapshot(
        path=path,
        nworld=nworld,
        nsteps=nsteps,
        deterministic=args.deterministic,
        broadphase=broadphase,
        graph_conditional=graph_conditional,
        **kwargs,
      )
      for _ in range(args.repeats)
    ]
    key = f"{path}|nworld={nworld}|nsteps={nsteps}"
    result["cases"][key] = _summarize_snapshots(snapshots)

  payload = json.dumps(result, indent=2)
  if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
      f.write(payload)
  print(payload)


if __name__ == "__main__":
  main()
