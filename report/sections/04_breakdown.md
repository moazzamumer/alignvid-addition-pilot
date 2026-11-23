**Labels**

- One labeller only. The second labeller planned in the protocol was not available before submission; raw agreement and κ cannot be reported.
- Partially visible subject. For the person prompt both arms produced hands and a book at the right frame edge, never a seated person. The rubric does not say when a part counts as the object. It was labelled presence = yes, object_correct = wrong attribute, primary failure = wrong_location, with a note; a rule for partial visibility should be written into the rubric.
- Scene replacement. For the stag prompt both arms cut, about 6 s in, to a different scene that contains a stag. Presence = yes is correct by the rubric, but on its own it would mislead; preservation = 1 and scene_drift carry the finding. Presence should always be read together with preservation and within_window.
- Persistence is defined from the onset, so a stag visible only in the last four frames has persistence 1.0 and within_window = no. Two fields are needed to see a late arrival; a single "appeared in window and stayed" field would read more easily.
- `transient` has no n.a. value for absent objects; it was recorded as no for the four no-change videos. Add n.a.
- Clip length. FramePack's `--length 6.0` produces 181 frames at 24 fps, 7.5 s. The 16-frame sheet spans the whole clip, so the window "onset ≤ f03" still means the first quarter, but the protocol's seconds were not FramePack's seconds. State windows in frames of the sheet, as done, and record the actual duration.

**Tools**

- Detector presence agreed with the human in 9 of 10 videos. The miss was partial visibility: three detections at ~0.53, one frame short of k = 4. Grounding DINO's confidence on a cropped person is low; a lower k would have caught it but was fixed in advance and was not changed.
- Placement rule boundary. The egg is larger than the cart region: 40–42 % of the detector box lies inside it, so the detector says partial while the human says in. The rule should be defined on the object's support (where it rests) or on the share of the region covered, or regions should be drawn to include the full extent the object may reach.
- The pre-existing check was decisive once: "tree" matched existing trees in every frame of both arms. Running the query on the reference image, as the protocol specifies, is what kept the detector honest.
- Onset differed by one frame in three videos; the human took the later frame when torn, as the rubric says. One frame of 16 is within the method's resolution.
- Region re-projection after FramePack's resize and centre crop worked for all ten videos (crop map stored in every detector JSON). One sample came out at 832×480 and the other four at 768×512, from FramePack's aspect-ratio buckets.
- Detector cost on a laptop CPU: 22 min and 11 min for five videos each (16 frames plus one reference image per video).

**Compute**

- The T4 is below the backbone's assumptions. Three numerics changes were needed and are recorded as deviations: chunked attention (no numeric change), fp16 arithmetic, and an fp32 residual stream to stop fp16 overflow. Equivalence with bf16 was checked on one clip only (PSNR 37.1 dB); the authors did not run fp16.
- 55–62 min per 6 s video, 13.0–13.2 GB GPU peak, 28–29 GB of 31 GB RAM. The RAM peak rules out a second process on the second T4 and rules out Colab free (12.7 GB RAM).
- Kaggle practicalities: 12 h per headless run, outputs kept only on clean completion, 30 h/week quota, queue waits. The orchestrator is resumable across sessions and stops starting videos near the deadline.
- Not run: the no-edit arm (≈ 5 h). The pilot therefore has no reference for how much the backbone changes a scene under a neutral prompt.
