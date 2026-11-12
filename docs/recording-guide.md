# Recording guide

The voice you get is only as good as the recording you give it. Thirty minutes of care here save hours later. The app
checks every take and tells you what to fix, but it is easier to get it right the first time.

## Before you start

| What | Do this |
| --- | --- |
| **Microphone** | Use one microphone for the whole session and for every later session: a headset or a desk/USB condenser microphone. Do not switch between a laptop microphone and a headset between parts. Turn off noise suppression, "voice enhancement", automatic gain control and echo cancellation in your system and in the microphone's software if you can. |
| **Room** | A quiet, small, soft room: a bedroom with a bed and curtains beats a bathroom or an empty office. Close the window, switch off fans, air conditioning, fridges and notifications. Avoid hard echo (bare walls, glass). Do not record next to a computer fan or with music playing. |
| **Distance** | 15-20 cm (a hand's width and a half) from the microphone, slightly off-axis (microphone a little to the side of your mouth) to avoid popping on "p" and "b". Keep the same distance all the time. |
| **Posture** | Sit comfortably, sip water first, do not lean towards and away from the microphone. |
| **Voice** | Speak the way you want your clone to speak. If you read in a flat monotone, the clone will be flat. Use your natural accent and a natural pace, with the ups and downs of real speech. Small mistakes are fine: keep talking, do not restart. |

## Levels

You do not need to measure anything, the quality check does it for you. These are the targets it works towards:

- **Peak** below about -1 dBFS (no clipping). If the check reports clipping, move back or lower the input gain and
  re-record that part: clipped audio cannot be repaired.
- **Average loudness** around -26 to -18 dBFS. If the app says "Too quiet", move closer or raise the input gain. A quiet
  take is not a disaster (it is normalised), but the quieter the take, the more of the noise you also amplify.
- **Noise floor** as low as you can get it, and the signal-to-noise ratio as high as you can get it (30 dB or better is
  good). If the check complains, find the noise: a fan, a buzzing charger, a street.
- **Silence** only the natural pauses between sentences. Do not leave long gaps at the start or the end.

In your operating system's sound settings, speak at your normal volume and watch the input meter: it should move
about halfway and never touch the top.

## Presets and how much data you need

The presets exist in English, Ukrainian, German, French, Spanish and Polish. Each language has its own texts, written
for that language (not translated), with the same parts and reading times.

| Preset | Recording time | What happens | Result |
| --- | --- | --- | --- |
| **Quick** | about 1 minute | No training. The recording becomes the reference clip for zero-shot speech. | Usable at once. Similar to you, not the same: the speaker-similarity score is about 0.75-0.80. |
| **Standard** | about 6-7 minutes (5 parts of about 1.5 minutes) | The recording is also used to train a timbre model (RVC), about 55 minutes on a 12 GB RTX 4070. This is the setup that was validated. | About 0.86-0.88, close to the 0.88 ceiling of your own recordings against each other. |
| **Pro** | about 15-20 minutes | The same, with more data. | The best similarity and the most stable result across different texts. |

Less than about a minute is too short for anything. More than 20 minutes gives little extra. **Quality beats quantity**:
six clean minutes are better than twenty noisy ones.

Notes:

- The reference clip (15-30 s) is chosen automatically from your best take. The zero-shot part reads at most the first
  15 s of it.
- Read the parts one after the other in a single session if you can, so the room, the microphone and your voice stay the
  same. If you must split it up, use the same microphone and distance. When you add a part later, it replaces the older
  take of that part.
- Your voice drifts when you are tired, ill or have just woken up. Record when you feel normal.

## Your own text

You can paste your own reading text instead of the presets. The app analyses it and tells you:

- **Length**: too short (less than about a minute of speech), enough for zero-shot, enough to train, or great.
- **Language**: detected from the text (or the one you pick): English, Ukrainian, German, French, Spanish or Polish.
- **Phoneme coverage**: how many of the sounds of that language your text contains, and which are missing (for
  example `ZH`, the sound in English "measure", or `dzʲ`, the sound in Ukrainian "дзьоб"). The more of them the text
  contains, the more complete the voice. If sounds are missing, add sentences that use them (the message shows an
  example word for each).
- The text is split into parts of about 60-90 seconds, so you can read it one part at a time.

Write in one language per text. A text with names, numbers and abbreviations written out as they are said is read better
than raw digits and symbols.

## Good habits

- Do a one-sentence test recording first, listen to it on headphones, and fix what you hear before you record the real
  thing.
- Record in the same session, with the same settings, for all parts.
- Re-record a part as soon as the check says "bad", not after training.
- Do not use recordings from a phone call, a video call or a lossy source, and never somebody else's voice (see the
  [disclaimer](../DISCLAIMER.md)).
