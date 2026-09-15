from __future__ import annotations

from pathlib import Path
import time
import unittest
from unittest.mock import patch

import stacking_logic as stacking


class StackingDeterminismTests(unittest.TestCase):
    def test_reduction_tree_restores_logical_order_after_out_of_order_futures(self):
        leaves = [
            stacking.SubstackInfo(Path(f"leaf_{index}.fits"), 1, index)
            for index in range(4)
        ]
        observed: list[list[int]] = []

        def fake_branch(children, config, output_path, cancel_event):
            order = [child.order for child in children]
            observed.append(order)
            # Complete the right branch first to exercise as_completed().
            if output_path.name.endswith("_00.fits"):
                time.sleep(0.02)
            return stacking.SubstackInfo(output_path, sum(child.frame_count for child in children))

        with patch.object(stacking, "_process_branch", side_effect=fake_branch):
            result = stacking._reduce_substacks_tree(
                leaves,
                stacking.StackingConfig(workers=2),
                Path("."),
                2,
                None,
                None,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(observed[-1], [0, 1])
        self.assertIn([2, 3], observed)


if __name__ == "__main__":
    unittest.main()

