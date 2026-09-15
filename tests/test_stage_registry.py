from collections import defaultdict
from pathlib import Path
import unittest

from app.application.stages import STAGE_DEFINITIONS, StageContext
from views.flow_view import format_reference_provenance


ROOT = Path(__file__).parents[1]


class StageRegistryTests(unittest.TestCase):
    def test_registry_builds_all_passive_models_without_root_controller(self):
        variables = defaultdict(lambda: object())
        callbacks = defaultdict(lambda: (lambda *_args, **_kwargs: None))
        context = StageContext(variables, callbacks, {})

        self.assertEqual(
            [definition.identifier for definition in STAGE_DEFINITIONS],
            ["Calibration", "Batch", "Flow", "Align", "Stack", "HDR"],
        )
        for definition in STAGE_DEFINITIONS:
            model = definition.build_model(context)
            self.assertIsNotNone(model)
            self.assertIsNotNone(definition.view_factory)
            self.assertIsNotNone(definition.command_builder)
            self.assertEqual(len(definition.operation_controls), 2)

    def test_composition_root_uses_registry_instead_of_model_construction(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("definition.build_view", source)
        for model_name in (
            "CalibrationViewModel(", "BatchViewModel(", "FlowViewModel(",
            "AlignViewModel(", "StackViewModel(", "HDRViewModel(",
        ):
            self.assertNotIn(model_name, source)

    def test_flow_reference_cards_expose_graph_provenance(self):
        text = format_reference_provenance(
            {
                "frames": {
                    "03.fits": {
                        "status": "accepted",
                        "recovery_method": "neighbor_chain",
                        "hop_count": 2,
                        "relative_to": "02.fits",
                        "confidence": "medium",
                    }
                }
            },
            "03.fits",
        )
        self.assertIn("neighbor_chain", text)
        self.assertIn("hops=2", text)
        self.assertIn("parent=02.fits", text)


if __name__ == "__main__":
    unittest.main()
