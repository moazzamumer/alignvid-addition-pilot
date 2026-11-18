# Human label rubric (frozen with the protocol)

You will see, for each coded video: the reference image with the intended region drawn, the instruction, a 4×4 contact sheet of 16 frames, and the video itself. You will NOT know which method produced it. Label from the contact sheet; watch the video once to check for anything the 16 frames miss and record it in notes.

Ignore lighting, shadow and reflection changes that the requested object would logically cause. Natural motion (wind, water, slight camera drift) is not collateral change unless it alters scene content.

| Field | Values | How to decide |
|---|---|---|
| presence | yes / no / unsure | Is the requested object recognisably there in at least a few frames? |
| onset_frame | 0–15 or n.a. | First contact-sheet frame where you are confident it is the object; if torn between two frames, take the later one and say so in notes |
| persistence | 0.0–1.0 or n.a. | Fraction of contact-sheet frames from onset to f15 inclusive in which it is visible |
| visible_f15 | yes / no / n.a. | Is it visible in the last frame? |
| transient | yes / no | Appears then disappears, or flickers (persistence < 0.75 or not visible in f15) |
| placement | in / partial / out / n.a. | Share of the object's own box that lies inside the drawn region(s) (union if two): in ≥ 50 %, partial 10–50 %, out < 10 %. Judged by eye on the frame where the object is clearest |
| object_correct | yes / wrong object / wrong attribute / n.a. | Right thing, or a different thing / wrong colour-size-count |
| preservation | 3 / 2 / 1 | 3 = outside the region unchanged apart from natural motion; 2 = minor collateral change (small items altered, texture drift); 1 = major change or scene replaced |
| preservation_note | free text | What changed outside the region |
| primary_failure | none / no_change / wrong_object / wrong_location / transient / collateral_change / scene_drift / artefacts / motion_only | One label |
| secondary_tags | any of the above, comma-separated | Several may co-occur |
| scorable | yes / no | no = the video cannot be labelled |
| scorable_reason | free text | e.g. generation failed, corrupt file, object too small to judge |
| within_window | yes / no / n.a. | Onset at or before the sample's window frame (protocol §5), persistence ≥ 0.75 and visible in f15 |
| notes | free text | Anything that would help a second person understand the call |

Each row of `labels/labels_template.csv` is one coded video; the columns match this table plus `video_code`, `labeller`, `date`.
