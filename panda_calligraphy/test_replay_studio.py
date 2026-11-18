"""Ink-preview and playback checks without a browser or learned-model dependency."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from callirewrite_replay import prepare, mask_rgb, tuft_geometry
from replay_studio import (ReplaySettings, ReplayPlayback, PreviewCanvas,
                          render_preview, save_preview_settings, STUDIO_MODE)
from test_callirewrite_replay import source_example


class ReplayStudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = prepare(source_example(), resolution=128)
        cls.final = len(cls.replay.time_s)-1

    def test_default_preview_is_exact_native_raster_and_ignores_disabled_ink_settings(self):
        expected = np.zeros((128,128),bool)
        for frame in range(self.final+1):
            self.replay.deposit(expected,frame)
        for settings in (ReplaySettings(), replace(ReplaySettings(),opacity=0,ink_level=0,
                            ink_color=(255,0,0),absorbency=3,footprint_scale=2)):
            rgb, mask = render_preview(self.replay,self.final,settings)
            np.testing.assert_array_equal(mask,expected)
            np.testing.assert_array_equal(rgb,mask_rgb(expected))

    def test_color_opacity_and_dip_level_change_existing_ink(self):
        settings = ReplaySettings(mode=STUDIO_MODE,ink_color=(180,30,40))
        full, mask = render_preview(self.replay,self.final,settings)
        self.assertTrue(mask.any())
        np.testing.assert_array_equal(full[mask][0], [180,30,40])
        dry,_ = render_preview(self.replay,self.final,replace(settings,ink_level=0))
        np.testing.assert_array_equal(dry,np.broadcast_to([251,247,238],dry.shape))
        half,_ = render_preview(self.replay,self.final,replace(settings,opacity=.5))
        self.assertTrue(np.all(half[mask] >= full[mask]))
        dipped,_ = render_preview(self.replay,self.final,replace(settings,ink_level=100))
        np.testing.assert_array_equal(dipped,full)

    def test_width_and_bleed_restyle_accumulated_progress_without_motion_changes(self):
        q, times, targets = self.replay.solution.q.copy(),self.replay.time_s.copy(),self.replay.target_tip.copy()
        settings = ReplaySettings(mode=STUDIO_MODE)
        canvas = PreviewCanvas(self.replay)
        normal, narrow = canvas.update(self.final,replace(settings,footprint_scale=.5))
        wide, wide_mask = canvas.update(self.final,replace(settings,footprint_scale=2))
        self.assertGreater(wide_mask.sum(),narrow.sum())
        bled,_ = canvas.update(self.final,replace(settings,footprint_scale=2,absorbency=3))
        self.assertGreater(np.count_nonzero(np.any(bled != [251,247,238],axis=2)),wide_mask.sum())
        np.testing.assert_array_equal(self.replay.solution.q,q)
        np.testing.assert_array_equal(self.replay.time_s,times)
        np.testing.assert_array_equal(self.replay.target_tip,targets)

    def test_clear_is_a_new_accumulation_window_and_does_not_rewind(self):
        state = ReplayPlayback(self.replay)
        contact_frames = np.flatnonzero(self.replay.contact_index>=0)
        state.time = float(self.replay.time_s[contact_frames[2]])
        before = state.time
        state.clear()
        self.assertEqual(state.time,before)
        canvas = PreviewCanvas(self.replay)
        _, cleared = canvas.update(state.frame,ReplaySettings(),state.paint_start)
        self.assertFalse(cleared.any())
        _, future = canvas.update(self.final,ReplaySettings(),state.paint_start)
        expected = np.zeros((128,128),bool)
        for i in range(state.paint_start,self.final+1):
            self.replay.deposit(expected,i)
        np.testing.assert_array_equal(future,expected)

    def test_step_completes_one_stroke_and_pauses_then_advances_next(self):
        state = ReplayPlayback(self.replay)
        contacts = np.flatnonzero(self.replay.contact_index>=0)
        for stroke in (0,1):
            self.assertTrue(state.step_stroke())
            self.assertEqual(state.advance(1000,ReplaySettings()),"stroke")
            expected = contacts[self.replay.source['stroke_id'][self.replay.contact_index[contacts]]==stroke][-1]
            self.assertEqual(state.frame,expected)
            self.assertFalse(state.playing)
        self.assertFalse(state.step_stroke())

    def test_speed_pause_and_restart_do_not_modify_trajectory(self):
        state = ReplayPlayback(self.replay)
        state.toggle();state.advance(.1,replace(ReplaySettings(),speed=2))
        self.assertAlmostEqual(state.time,.2)
        state.toggle();state.advance(3,ReplaySettings())
        self.assertAlmostEqual(state.time,.2)
        state.restart(playing=True)
        self.assertEqual(state.time,0);self.assertEqual(state.paint_start,0)
        self.assertEqual(state.advance(1000,ReplaySettings()),"completed")
        state.toggle()
        self.assertEqual(state.time,0);self.assertTrue(state.playing)

    def test_visual_stiffness_and_splay_change_mesh_only(self):
        args=(np.array([.55,0,.22]),np.diag([1.,-1.,-1.]),np.array([.55,0,-.005]),.003,90.,.006,True)
        default,faces,_=tuft_geometry(*args)
        stiff,_,_=tuft_geometry(*args,stiffness=3)
        splayed,_,_=tuft_geometry(*args,splay=2)
        self.assertGreater(np.max(np.abs(default-stiff)),.0001)
        self.assertGreater(np.max(np.abs(default-splayed)),.0001)
        self.assertEqual(default.shape,splayed.shape)

    def test_settings_snapshot_saved_separately_and_roundtrippable(self):
        state=ReplayPlayback(self.replay)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'preview'/'custom.json'
            save_preview_settings(path,ReplaySettings(mode=STUDIO_MODE,ink_color=(90,40,20)),state)
            data=json.loads(path.read_text())
            restored=ReplaySettings(**data['settings'])
            self.assertEqual(restored.ink_color,(90,40,20))
            self.assertIn('Display-only',data['scope'])
            self.assertEqual(list(Path(d).iterdir()),[Path(d)/'preview'])

    def test_invalid_settings_and_playback_delta_are_rejected(self):
        for kwargs in ({'opacity':float('nan')},{'footprint_scale':0},{'ink_level':101},
                       {'stiffness':0},{'ink_color':(0,0,999)},{'mode':'unknown'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                ReplaySettings(**kwargs)
        with self.assertRaises(ValueError):
            ReplayPlayback(self.replay).advance(float('nan'),ReplaySettings())


if __name__ == '__main__':
    unittest.main()
