import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.application.runner import PipelineRunner, OperationResult
from app.application.pipelines import execute_pipeline
from app.application.log_buffer import ActivityBuffer
from app.infrastructure.json_store import SettingsRepository


class RunnerTests(unittest.TestCase):
    def test_progress_coalesced_completion_once_on_consumer_thread(self):
        runner=PipelineRunner(Mock())
        def operation(log, progress, cancel):
            for i in range(10000): progress(i,10000,'work')
            return OperationResult('success','done')
        thread=runner.start('Test',operation); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(runner.busy)
        with self.assertRaises(RuntimeError): runner.start('Second',operation)
        progress, result=runner.drain()
        self.assertEqual(progress.current,9999); self.assertEqual(result.outcome,'success')
        self.assertEqual(runner.drain(),(None,None)); self.assertFalse(runner.busy)

    def test_failure_and_start_failure_unlock(self):
        log=Mock(); runner=PipelineRunner(log)
        def broken(*args): raise RuntimeError('disk full')
        runner.start('Test',broken).join(3)
        self.assertEqual(runner.drain()[1].outcome,'failed'); log.assert_called_once()
        with patch('threading.Thread.start',side_effect=RuntimeError('start failed')):
            with self.assertRaises(RuntimeError): runner.start('Test',broken)
        self.assertFalse(runner.busy)

    def test_cancel_cannot_be_reported_as_success(self):
        runner=PipelineRunner(Mock()); entered=threading.Event()
        def operation(log,progress,cancel):
            entered.set(); cancel.wait(3)
            return OperationResult('success','done')
        thread=runner.start('Test',operation); self.assertTrue(entered.wait(2))
        runner.cancel(); thread.join(3)
        self.assertEqual(runner.drain()[1].outcome,'cancelled')

    def test_partial_align_and_empty_calibration_are_not_success(self):
        with patch('astroalign_logic.process_all_alignments',return_value=(2,1)):
            result=execute_pipeline('Align',(Path('.'),Path('.'),{}),Mock(),Mock(),threading.Event())
        self.assertEqual(result.outcome,'partial')
        with patch('calibration_logic.run_calibration_pipeline',return_value=None):
            result=execute_pipeline('Calibration',({},),Mock(),Mock(),threading.Event())
        self.assertEqual(result.outcome,'failed')

    def test_activity_flood_retains_errors_and_reports_loss(self):
        buffer=ActivityBuffer(capacity=5)
        buffer.put('ERROR: disk failure')
        for i in range(50): buffer.put(f'progress {i}')
        self.assertEqual(buffer.qsize(),5)
        self.assertIn('46 mensagens',buffer.get_nowait())
        self.assertIn('ERROR',buffer.get_nowait())
        for _ in range(4): buffer.get_nowait()
        with self.assertRaises(queue.Empty): buffer.get_nowait()

    def test_settings_migrate_and_failed_replace_preserves_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'settings.json'; repo=SettingsRepository(path)
            repo.save({'AstroFlow':{'sigma':3}})
            old=path.read_bytes(); self.assertEqual(repo.load()['_schema_version'],2)
            with patch('app.infrastructure.json_store.os.replace',side_effect=OSError('disk')):
                with self.assertRaises(OSError): repo.save({'changed':True})
            self.assertEqual(path.read_bytes(),old)
            self.assertEqual(list(path.parent.glob('*.tmp')),[])


if __name__=='__main__': unittest.main()
