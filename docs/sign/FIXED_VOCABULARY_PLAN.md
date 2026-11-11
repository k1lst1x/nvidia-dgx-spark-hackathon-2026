# Fixed-vocabulary sign recognition: plan and evaluation

Open-vocabulary sign translation is not good enough to ship
(`HOW2SIGN_INTEGRATION.md`). The practical target is a small vocabulary that
the app recognizes reliably and refuses everything else. This file lists the
vocabulary, how to record it, and how to score the current model on it.

## Why thresholds alone do not fix false words

`apps/api/sign_language/evaluation/evaluate_word_engine.py` replays the
app's word engine on recordings. Run on the 8 How2Sign clips (continuous
signing of unrelated sentences, about 25 s with hands visible), every word it
confirms is almost certainly wrong:

| Margin rule | Floor | Words confirmed |
|---|---|---|
| none | 0.30 | 12 |
| 0.10 (current app) | 0.30 | 11 |
| 0.20 | 0.30 | 9 |
| 0.10 | 0.60 | 5 |
| 0.10 | 0.80 | 4 |

The model is overconfident on signs it does not know: the median top
probability per window is 0.48, and one window in ten is above 0.95. Raising
the floor cuts false words, but how many real words it also blocks is unknown
until there are recordings of the vocabulary below. The app's thresholds are
left unchanged until then.

## Vocabulary (45 signs, all in the current model's 250 labels)

| Group | Signs |
|---|---|
| Answers | yes, no, not, fine, bad, same |
| Politeness | hello, bye, please, thankyou |
| Needs | water, drink, food, hungry, thirsty, potty, sleep, sick, owie (hurt) |
| Actions | wait, stay, go, look, find, give, open, close, stuck, finish |
| Talk | say, talk, hear, listen, where, who, why |
| People and place | minemy (my), yourself, police, home, outside, stairs |
| Time | now, later, tomorrow |

Gloss labels are typed in the recorder as the model's label (`THANKYOU`,
`MINEMY`); the evaluation lower-cases them and strips spaces and punctuation.

## Recording protocol (`/record` screen, web app)

- **Signers:** at least 3, ideally 5, fluent in ASL. Signer IDs are codes,
  never names. Record consent in the screen before each session.
- **Per sign:** 10 takes per signer, one sign per take, hands down between
  takes. Vary distance and lighting a little between sessions.
- **No-sign clips:** gloss `NONE`. 30 per signer: scratching, waving at
  someone, adjusting glasses, typing, talking with hands, picking up a cup.
  These measure false words, which matter more than missed ones.
- **Labels:** the signer or a fluent checker confirms each label before Stop.
  Discard any take that went wrong instead of relabelling it.
- **Storage:** samples download as JSON (landmarks only, no video). Keep them
  outside git, e.g. `datasets/visionmate_signs/<signer>/`.
- **Split:** score and tune on different signers. Never tune thresholds on the
  signers used for the final number.

## Scoring

```powershell
python apps/api/sign_language/evaluation/evaluate_word_engine.py `
  --samples datasets/visionmate_signs/all `
  --model apps/mobile/assets/models/asl_word_transformer.onnx `
  --labels apps/mobile/assets/models/asl_word_labels.json `
  --output datasets/visionmate_signs/word_engine_results.json
```

Needs `onnxruntime` and `numpy`. It prints, for each margin, how often the
right word was confirmed, a wrong word was confirmed, nothing was confirmed,
and how many words were confirmed on `NONE` clips, plus the most common
confusions. The rule-based gestures are not replayed, and recordings are at
20 fps while the live engine sees every camera frame.

## Decisions the numbers will drive

1. **Pick the floor and margin** that keep false words on `NONE` clips near
   zero, then check what that costs in correct words.
2. **Restrict output to the 45 signs.** Words outside the vocabulary are
   never confirmed, which removes most of the confusions seen above.
3. **If the 250-word model is too weak on these signs,** train a small
   classifier on the recordings with an explicit no-sign class, as a
   separate model. The existing ONNX model is not modified.
