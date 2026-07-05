"""Control-bit decode + schedule tests against recorded SASS-hex fixtures."""

from pathlib import Path

import pytest

from cuxray.analyze.schedule import loop_schedule
from cuxray.parse import cfgdot, ctrl, sass

REC = Path(__file__).parent / "fixtures" / "recorded"


class TestDecode:
    def test_fields_in_range(self):
        controls = ctrl.parse_sass_controls((REC / "sasshex.spill.sm_90.txt").read_text())
        table = controls["_Z6spillyPKfPfii"]
        assert len(table) > 300
        for c in table.values():
            assert 0 <= c.stall <= 15 and 0 <= c.watdb <= 63
            assert c.wrtdb <= 7 and c.readdb <= 7

    def test_scoreboard_pairing_exists(self):
        # some instruction sets a write barrier, and a later one waits on it
        controls = ctrl.parse_sass_controls((REC / "sasshex.spill.sm_90.txt").read_text())
        table = sorted(controls["_Z6spillyPKfPfii"].items())
        paired = False
        for i, (addr, c) in enumerate(table):
            if c.wrtdb != 7:
                bit = 1 << c.wrtdb
                if any(c2.watdb & bit for _, c2 in table[i + 1:i + 40]):
                    paired = True
                    break
        assert paired, "no wrtdb/watdb scoreboard pairing found — decode layout suspect"

    def test_arch_gate(self):
        assert ctrl.arch_supported("sm_86")
        assert ctrl.arch_supported("sm_90a")
        assert not ctrl.arch_supported("sm_120a")
        assert not ctrl.arch_supported("sm_75")
        assert not ctrl.arch_supported(None)


class TestSchedule:
    def test_spill_loop_schedule(self):
        dis = sass.parse_gi((REC / "nvdisasm_gi.spill.sm_90.txt").read_text())
        cfg = cfgdot.parse((REC / "nvdisasm_cfg.spill.sm_90.dot").read_text())
        controls = ctrl.parse_sass_controls((REC / "sasshex.spill.sm_90.txt").read_text())
        name = "_Z6spillyPKfPfii"
        rows = loop_schedule(dis.functions[name], cfg[name], controls[name])
        assert rows
        hot = rows[0]
        assert hot["loop_depth"] == 1
        assert hot["est_issue_stall_cycles_per_iter"] > 50
        assert hot["coverage"] > 0.9
        assert hot["top_stall_lines"][0]["line"] in (13, 14)
