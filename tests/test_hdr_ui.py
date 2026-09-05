import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch
import main
from app.application.runner import OperationResult


class NativeUIArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.patches=[patch.object(main.AstroProcessManager,name) for name in
                      ('load_settings','save_settings','_start_cpu_kernel_warmup')]
        for p in self.patches: p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        try:
            self.app=main.AstroProcessManager()
        except tk.TclError as exc:
            self.skipTest(f'Tk unavailable: {exc}')
        self.app.withdraw()
        self.addCleanup(self.app.destroy)

    def test_native_hdr_fields_handoff_and_dialog_binding(self):
        app=self.app
        self.assertEqual(len(app.tab_hdr.entries),6)
        app.align_output_dir_var.set('C:/aligned')
        self.assertTrue(app.use_align_output_for_hdr())
        self.assertEqual(app.hdr_input_var.get(),'C:/aligned')
        self.assertEqual(app.notebook.select(),str(app._tab_hosts['hdr']))
        entry=app.tab_hdr.entries['Ruído por frame (unidades calibradas)']
        entry.delete(0,'end'); entry.insert(0,'2.5')
        self.assertEqual(app.hdr_noise_var.get(),2.5)
        with patch.object(main.filedialog,'asksaveasfilename',return_value='C:/output.fits'):
            app.browse_save_file(app.hdr_output_var)
        self.assertEqual(app.hdr_output_var.get(),'C:/output.fits')

    def test_all_buttons_use_one_runner_and_completion_lifecycle(self):
        app=self.app
        with patch.object(main,'execute_pipeline',return_value=OperationResult('partial','one frame failed')):
            app._start_operation('HDR',{})
            app.worker.join(3)
        for _,run,cancel in app._operation_buttons():
            self.assertEqual(str(run.cget('state')),'disabled')
        self.assertEqual(str(app.btn_cancel_hdr.cget('state')),'normal')
        app._drain_operation_events(); app.update()
        self.assertNotEqual(app.progress_var.get(),100.)
        for _,run,cancel in app._operation_buttons():
            self.assertEqual(str(run.cget('state')),'normal')
            self.assertEqual(str(cancel.cget('state')),'disabled')

    def test_scroll_forms_and_invalid_resource_input(self):
        app=self.app
        app.deiconify(); app.geometry('1000x760'); app.update_idletasks()
        self.assertEqual(len(app.notebook.tabs()),6)
        host=app._tab_hosts['align']; app.notebook.select(host); app.update_idletasks()
        host.canvas.yview_moveto(1.)
        self.assertLessEqual(host.canvas.yview()[1],1.)
        with tempfile.TemporaryDirectory() as td:
            app.batch_dir_var.set(td); app.resource_memory_var.set(-1)
            with patch.object(main.messagebox,'showerror') as error:
                app.start_flow_processing(); error.assert_called_once()
            self.assertFalse(app.runner.busy)


if __name__=='__main__': unittest.main()
