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
"""Tests for GPU determinism (contact sorting and deterministic step state)."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import BroadphaseType
from mujoco_warp import test_data
from mujoco_warp._src import collision_driver

_NSTEPS = 10
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
_SENSOR_CONTACT_FIELDS = tuple(field for field in _CONTACT_FIELDS if field != "geomcollisionid")
_SCALAR_FIELDS = ("time", "nefc")
_CASE_CAPACITY_OVERRIDES = {
  ("humanoid/humanoid.xml", 4): {"njmax": 128},
}
_CONTACT_SENSOR_NETFORCE_MJCF = """
<mujoco>
  <worldbody>
    <geom name="plane" type="plane" size="10 10 .001"/>
    <body>
      <geom name="box" type="box" size=".1 .1 .1"/>
      <freejoint/>
    </body>
  </worldbody>
  <sensor>
    <contact geom1="plane" geom2="box" data="found force torque dist pos normal tangent" reduce="netforce" num="2"/>
    <contact geom1="box" geom2="plane" data="force torque" reduce="netforce"/>
  </sensor>
  <keyframe>
    <key qpos="0 0 .09 1 0 0 0"/>
  </keyframe>
</mujoco>
"""


def _run_and_collect_contacts(path, nworld, nsteps, deterministic):
  """Run simulation and return contact geom arrays from last step."""
  _, _, m, d = test_data.fixture(path=path, nworld=nworld)
  m.opt.deterministic = deterministic
  for _ in range(nsteps):
    mjw.step(m, d)
  nacon = d.nacon.numpy()[0]
  return {
    "nacon": nacon,
    "geom": d.contact.geom.numpy()[:nacon].copy(),
    "dist": d.contact.dist.numpy()[:nacon].copy(),
    "pos": d.contact.pos.numpy()[:nacon].copy(),
    "frame": d.contact.frame.numpy()[:nacon].copy(),
    "dim": d.contact.dim.numpy()[:nacon].copy(),
    "worldid": d.contact.worldid.numpy()[:nacon].copy(),
    "geomcollisionid": d.contact.geomcollisionid.numpy()[:nacon].copy(),
  }


def _copy_contact_fields(d):
  """Return copies of every contact array."""
  return {field: getattr(d.contact, field).numpy().copy() for field in _CONTACT_FIELDS}


def _write_contact_fields(d, contact_fields):
  """Write full contact arrays back to device memory."""
  for field, values in contact_fields.items():
    arr = getattr(d.contact, field)
    wp.copy(arr, wp.array(values, dtype=arr.dtype, device=arr.device))


def _permute_active_contacts(contact_fields, nacon, perm):
  """Return a copy with the active contacts permuted by `perm`."""
  permuted = {field: values.copy() for field, values in contact_fields.items()}
  for field, values in permuted.items():
    values[:nacon] = values[perm]
  return permuted


def _sorted_contact_order(contact_fields, nacon):
  """Return stable sorted indices for the active contacts."""
  geom = contact_fields["geom"]
  worldid = contact_fields["worldid"]
  geomcollisionid = contact_fields["geomcollisionid"]
  return sorted(
    range(nacon),
    key=lambda idx: (
      int(worldid[idx]),
      int(geom[idx, 0]),
      int(geom[idx, 1]),
      int(geomcollisionid[idx]),
    ),
  )


def _contacts_are_sorted(contact_fields):
  """Returns True if active contacts are sorted by the deterministic key."""
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


def _capacity_overrides(path, nworld):
  return dict(_CASE_CAPACITY_OVERRIDES.get((path, nworld), {}))


def _fresh_case(path=None, xml=None, nworld=1, nconmax=None, njmax=None, overrides=tuple()):
  """Create a fresh MJWarp case with optional custom workspace capacities."""
  mjm, mjd, m, d = test_data.fixture(path=path, xml=xml, nworld=nworld, overrides=overrides)
  if nconmax is not None or njmax is not None:
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
  return mjm, mjd, m, d


def _collect_state_fields(d):
  """Return copies of state arrays that should remain bitwise deterministic."""
  fields = {field: getattr(d, field).numpy().copy() for field in _STATE_FIELDS}
  fields.update({field: getattr(d, field).numpy().copy() for field in _SCALAR_FIELDS})
  return fields


def _collect_active_contact_fields(d):
  """Return copies of active contact arrays only."""
  nacon = int(d.nacon.numpy()[0])
  return nacon, {field: getattr(d.contact, field).numpy()[:nacon].copy() for field in _CONTACT_FIELDS}


def _run_step_snapshot(
  *,
  path,
  nworld,
  nsteps,
  deterministic,
  broadphase=None,
  graph_conditional=None,
  nconmax=None,
  njmax=None,
):
  """Run a fresh simulation and collect deterministic state/contact snapshots."""
  _, _, m, d = _fresh_case(path=path, nworld=nworld, nconmax=nconmax, njmax=njmax)
  m.opt.deterministic = deterministic
  if broadphase is not None:
    m.opt.broadphase = broadphase
  if graph_conditional is not None:
    m.opt.graph_conditional = graph_conditional

  for _ in range(nsteps):
    mjw.step(m, d)
  wp.synchronize()

  nacon, contacts = _collect_active_contact_fields(d)
  return {
    "nacon": nacon,
    "state": _collect_state_fields(d),
    "contacts": contacts,
  }


def _run_sensor_snapshot(*, xml, nworld, deterministic):
  """Run a fresh contact-sensor evaluation and collect deterministic outputs."""
  _, _, m, d = _fresh_case(xml=xml, nworld=nworld)
  m.opt.deterministic = deterministic
  d.sensordata.zero_()
  mjw.sensor_acc(m, d)
  wp.synchronize()

  nacon, contacts = _collect_active_contact_fields(d)
  return {
    "nacon": nacon,
    "state": {"sensordata": d.sensordata.numpy().copy()},
    "contacts": contacts,
  }


def _assert_snapshots_equal(
  testcase,
  snapshots,
  state_fields=_STATE_FIELDS + _SCALAR_FIELDS,
  contact_fields=_CONTACT_FIELDS,
):
  """Assert that a list of snapshots is bitwise identical."""
  base = snapshots[0]
  for run, snapshot in enumerate(snapshots[1:], start=1):
    testcase.assertEqual(base["nacon"], snapshot["nacon"], msg=f"nacon differs in run {run}")
    for field in state_fields:
      np.testing.assert_array_equal(
        base["state"][field],
        snapshot["state"][field],
        err_msg=f"{field} differs: run 0 vs run {run}",
      )
    for field in contact_fields:
      np.testing.assert_array_equal(
        base["contacts"][field],
        snapshot["contacts"][field],
        err_msg=f"{field} differs: run 0 vs run {run}",
      )


class ContactSortDeterminismTest(parameterized.TestCase):
  """Tests that contact sorting produces deterministic contact ordering."""

  @parameterized.parameters(
    ("collision.xml", 1),
    ("collision.xml", 4),
    ("humanoid/humanoid.xml", 1),
    ("humanoid/humanoid.xml", 4),
  )
  def test_contact_ordering_deterministic(self, path, nworld):
    """Contacts are bitwise identical across multiple runs."""
    nruns = 3
    results = [_run_and_collect_contacts(path, nworld, _NSTEPS, True) for _ in range(nruns)]

    self.assertGreater(results[0]["nacon"], 0, f"No contacts for {path}")

    for run in range(1, nruns):
      self.assertEqual(results[0]["nacon"], results[run]["nacon"])
      np.testing.assert_array_equal(
        results[0]["geom"],
        results[run]["geom"],
        err_msg=f"Contact geom ordering differs: run 0 vs run {run}",
      )

  @parameterized.parameters(
    ("collision.xml", 1),
    ("humanoid/humanoid.xml", 1),
  )
  def test_contact_fields_deterministic(self, path, nworld):
    """All contact fields are bitwise identical across runs."""
    nruns = 3
    results = [_run_and_collect_contacts(path, nworld, _NSTEPS, True) for _ in range(nruns)]

    self.assertGreater(results[0]["nacon"], 0)

    for run in range(1, nruns):
      self.assertEqual(results[0]["nacon"], results[run]["nacon"])
      for field in ("dist", "pos", "frame", "geom", "dim", "worldid", "geomcollisionid"):
        np.testing.assert_array_equal(
          results[0][field],
          results[run][field],
          err_msg=f"{field} differs: run 0 vs run {run}",
        )

  def test_contacts_sorted_by_geom(self):
    """Contacts are sorted by (worldid, geom0, geom1, geomcollisionid) after deterministic step."""
    result = _run_and_collect_contacts("collision.xml", 1, _NSTEPS, True)

    nacon = result["nacon"]
    self.assertGreater(nacon, 1)

    geom = result["geom"]
    worldid = result["worldid"]
    geomcollisionid = result["geomcollisionid"]

    for i in range(1, nacon):
      key_prev = (worldid[i - 1], geom[i - 1, 0], geom[i - 1, 1], geomcollisionid[i - 1])
      key_curr = (worldid[i], geom[i, 0], geom[i, 1], geomcollisionid[i])
      self.assertLessEqual(
        key_prev,
        key_curr,
        f"Contacts not sorted at index {i}: {key_prev} > {key_curr}",
      )

  def test_sort_contacts_reorders_mixed_contacts(self):
    """Sorting restores deterministic contact order after contacts are mixed."""
    _, _, m, d = test_data.fixture(path="collision.xml", nworld=4)
    m.opt.deterministic = False

    mjw.forward(m, d)

    nacon = d.nacon.numpy()[0]
    self.assertGreaterEqual(nacon, 5)
    original = _copy_contact_fields(d)
    perm = np.concatenate((np.arange(1, nacon, 2), np.arange(0, nacon, 2)))
    self.assertFalse(np.array_equal(perm, np.arange(nacon)))

    mixed = _permute_active_contacts(original, nacon, perm)
    _write_contact_fields(d, mixed)

    expected_order = _sorted_contact_order(mixed, nacon)
    expected = _permute_active_contacts(mixed, nacon, expected_order)

    collision_driver._sort_contacts(m, d)

    actual = _copy_contact_fields(d)
    self.assertEqual(d.nacon.numpy()[0], nacon)

    for field in _CONTACT_FIELDS:
      np.testing.assert_array_equal(
        actual[field][:nacon],
        expected[field][:nacon],
        err_msg=f"{field} was not permuted into deterministic order",
      )

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU determinism.")
  @parameterized.parameters(
    ("collision.xml", 1, 5),
    ("collision.xml", 4, 5),
    ("humanoid/humanoid.xml", 1, 5),
    ("humanoid/humanoid.xml", 4, 5),
  )
  def test_step_state_and_contacts_bitwise_deterministic(self, path, nworld, nsteps):
    """Repeated deterministic runs produce identical state/contact snapshots."""
    kwargs = _capacity_overrides(path, nworld)
    results = [
      _run_step_snapshot(
        path=path,
        nworld=nworld,
        nsteps=nsteps,
        deterministic=True,
        graph_conditional=False,
        **kwargs,
      )
      for _ in range(3)
    ]

    _assert_snapshots_equal(self, results)
    if results[0]["nacon"] > 1:
      self.assertTrue(_contacts_are_sorted(results[0]["contacts"]))

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU graph capture.")
  @parameterized.parameters(False, True)
  def test_step_bitwise_deterministic_with_graph_conditional(self, graph_conditional):
    """Graph-conditional and non-graph-conditional execution remain deterministic."""
    results = [
      _run_step_snapshot(
        path="collision.xml",
        nworld=4,
        nsteps=5,
        deterministic=True,
        graph_conditional=graph_conditional,
      )
      for _ in range(3)
    ]

    _assert_snapshots_equal(self, results)
    self.assertGreater(results[0]["nacon"], 0)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU determinism.")
  @parameterized.parameters(BroadphaseType.NXN, BroadphaseType.SAP_TILE, BroadphaseType.SAP_SEGMENTED)
  def test_step_bitwise_deterministic_all_broadphase_modes(self, broadphase):
    """All broadphase backends remain bitwise deterministic under the workaround."""
    results = [
      _run_step_snapshot(
        path="collision.xml",
        nworld=4,
        nsteps=5,
        deterministic=True,
        broadphase=broadphase,
        graph_conditional=False,
      )
      for _ in range(3)
    ]

    _assert_snapshots_equal(self, results)
    self.assertGreater(results[0]["nacon"], 0)
    self.assertTrue(_contacts_are_sorted(results[0]["contacts"]))

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU determinism.")
  def test_contact_sensor_netforce_bitwise_deterministic(self):
    """Sensor contact sorting stays bitwise deterministic across repeated runs."""
    results = [_run_sensor_snapshot(xml=_CONTACT_SENSOR_NETFORCE_MJCF, nworld=4, deterministic=True) for _ in range(3)]

    _assert_snapshots_equal(self, results, state_fields=("sensordata",), contact_fields=_SENSOR_CONTACT_FIELDS)
    self.assertTrue(results[0]["state"]["sensordata"].any())

  def test_deterministic_flag_default_false(self):
    """The deterministic flag defaults to False."""
    _, _, m, _ = test_data.fixture(path="collision.xml")
    self.assertFalse(m.opt.deterministic)


if __name__ == "__main__":
  absltest.main()
