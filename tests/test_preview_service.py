import time
import unittest
from threading import Event

from views.preview_service import PreviewService


class PreviewServiceTests(unittest.TestCase):
    def test_replacement_discards_stale_results_and_workers_return_plain_values(self):
        service = PreviewService(lambda value: (value * 2), max_workers=2, max_pending=4)
        try:
            service.replace(1, [("old", 2)])
            service.replace(2, [("new", 3)])
            deadline = time.time() + 2
            results = []
            while time.time() < deadline and not results:
                results.extend(service.drain())
                time.sleep(.01)
            self.assertTrue(results)
            self.assertTrue(all(item.generation == 2 for item in results))
            self.assertEqual(results[0].image, 6)
        finally:
            service.close()

    def test_submission_and_result_buffer_are_bounded(self):
        service = PreviewService(lambda value: value, max_workers=2, max_pending=2)
        try:
            service.replace(1, [(index, index) for index in range(20)])
            deadline = time.time() + 2
            results = []
            while time.time() < deadline:
                results.extend(service.drain())
                if len(results) >= 2:
                    break
                time.sleep(.01)
            self.assertLessEqual(len(results), 2)
            self.assertTrue({item.key for item in results}.issubset({0, 1}))
        finally:
            service.close()

    def test_close_is_idempotent_and_rejects_replacement(self):
        service = PreviewService(lambda value: value, max_workers=1, max_pending=2)
        service.close()
        service.close()
        service.replace(2, [("late", 1)])
        time.sleep(.05)
        self.assertEqual(service.drain(), [])

    def test_worker_errors_are_returned_as_plain_results(self):
        service = PreviewService(lambda _value: (_ for _ in ()).throw(RuntimeError("bad FITS")))
        try:
            service.replace(1, [("broken", None)])
            deadline = time.time() + 2
            results = []
            while time.time() < deadline and not results:
                results.extend(service.drain())
                time.sleep(.01)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].error, "bad FITS")
        finally:
            service.close()

    def test_close_discards_result_from_work_already_in_progress(self):
        entered, release = Event(), Event()

        def loader(_value):
            entered.set()
            release.wait(2)
            return "late"

        service = PreviewService(loader, max_workers=1)
        try:
            service.replace(1, [("closing", None)])
            self.assertTrue(entered.wait(1))
            service.close()
            release.set()
            time.sleep(.05)
            self.assertEqual(service.drain(), [])
        finally:
            release.set()
            service.close()


if __name__ == "__main__":
    unittest.main()
