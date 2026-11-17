# Human label rubric (frozen with the protocol)

You will see, for each coded video: the reference image with the intended region drawn, the instruction, a 4×4 contact sheet of 16 frames, and the video itself. You will NOT know which method produced it.

Ignore lighting, shadow and reflection changes that the requested object would logically cause. Natural motion (wind, water, slight camera drift) is not collateral change unless it alters scene content.

| Field | Values | How to decide |
|---|---|---|
| presence | yes / no / unsure | Is the requested object recognisably there in at least a few frames? |
| onset_frame | 0–15 or n.a. | First contact-sheet frame where it is recognisable |
| persistence | 0.0–1.0 or n.a. | Fraction of frames after onset where it remains visible |
| transient | yes / no | Appears then disappears, or flickers |
| placement | in / partial / out / n.a. | Using the drawn region and the ≥50 % / 10–50 % / <10 % rule, judged by eye on the frame where the object is clearest |
| object_correct | yes / wrong object / wrong attribute / n.a. | Right thing, or a different thing / wrong colour-size-count |
| preservation | 3 / 2 / 1 | 3 = outside the region unchanged apart from natural motion; 2 = minor collateral change (small items altered, texture drift); 1 = major change or scene replaced |
| preservation_note | free text | What changed outside the region |
| primary_failure | none / no_change / wrong_object / wrong_location / transient / collateral_change / scene_drift / artefacts / motion_only | One label |
| secondary_tags | any of the above, comma-separated | Several may co-occur |
| scorable | yes / no + reason | e.g. generation failed, corrupt file, object too small to judge |
| within_window | yes / no / n.a. | Did onset and persistence meet the pre-registered window for this sample? |
| notes | free text | Anything that would help a second person understand the call |
