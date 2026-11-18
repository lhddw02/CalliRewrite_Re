# Recorded Panda / CalliRewrite demonstration

Open `CalliRewrite_Panda_Viser_demo.mp4` in a normal video player. It is a silent,
captioned 150-second video at 1920 × 1080 and 30 fps, encoded as H.264/yuv420p.
`demo_poster.png` is the matching PowerPoint poster; the optional SRT repeats the
main explanatory captions already burned into the movie.

| Time | Content |
| --- | --- |
| 0:00–0:06 | Actual Glyph 6 input and integration overview |
| 0:06–1:33 | Complete recorded Panda replay, at the original capture speed |
| 1:33–1:56 | Actual side-view brush close-up |
| 1:56–2:15 | Scripted yi / legacy brush demonstration, separately labeled |
| 2:15–2:30 | Completed ink canvas, verification, and scope |

The main replay uses all 2,404 timestamped capture frames. The close-up uses all
633 frames. Each capture's elapsed timestamps are retained by an FFmpeg concat
demuxer and then sampled at 30 fps; the main clip has a final 0.008-second hold,
and the close-up has a final 1.298-second hold. No replay is accelerated. The
main recording includes the full 82.318-second trajectory and its surrounding
recorded lead-in / completion hold.

The separate yi chapter is the pre-existing hand-authored waypoint example.
It is not a CalliRewrite prediction. All robot imagery is recorded from Viser;
only crops, scaling, composition, and text were added.

`capture_provenance.json` records the input files and hashes, ordered capture
frame digest, crop rectangles, exact timestamp checks, encoding properties,
and chapter timings. The saved completed Viser RGB image and the offline
replay image are identical at 300 × 300, with zero differing pixels.

This verifies the software path from a per-glyph CalliRewrite refinement to
Panda kinematics and displayed ink. The tuft deformation is a geometric
illustration. This demonstration does not validate material mechanics,
dynamics, mesh collisions, generalization to unseen glyphs, or physical
handwriting accuracy.

To rebuild from the retained captures, run from the repository root:

```bash
python3 .local_setup/presentation_build/build_demo_video.py
```

The build script uses the installed FFmpeg 7.1.1 at
`/home/mihawk/anaconda3/envs/ros_env/bin/ffmpeg`; its concat demuxer supports the
millisecond image timebase needed to preserve capture timing. Chapter QA
frames are retained in `.local_setup/presentation_build/video_build/`.
