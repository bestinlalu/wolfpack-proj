# Body Lab — WolfHacks Master Plan

Oct 3, 2026 · @Bestin

## Overview

Body Lab is a personal scientist for your body: when your body reacts differently to a familiar situation, an AI agent investigates what was different, opens a hypothesis, and confirms it from your everyday life with no journaling.

- **Pitch:** "Health apps give you scores. Body Lab gives you a scientist."
- **Track:** Applied AI Software (Databricks). 1st prize: Fujifilm Instax Mini 12.
- **Extra MLH prizes:** Best Use of Gemini API, Best Use of ElevenLabs, Best Domain Name from GoDaddy Registry.
- **Data:** one dataset, PhysioNet BIG IDEAs (glucose + wristband + food logs for the same 16 people).
- **Scope:** four labs (Fuel, Stress, Sleep, Movement and Rhythm), one agent, passive gamification. Fuel Lab is the hero demo.
- **Mockup:** [Body Lab mockup](https://claude.ai/artifact/7PqmPr3Fmg89AdvNkDaDiV)

## Problem and why now

Millions of people now see their glucose curves, but no app tells them why one meal spiked and a similar meal did not.

- About 1 in 3 US adults has prediabetes and most do not know it (CDC, approximate). Lifestyle changes can prevent or delay type 2 diabetes.
- Over-the-counter glucose monitors launched in 2024 (Dexcom Stelo, Abbott Lingo), so raw curves are now in consumers' hands.
- Existing apps give scores and averages (Levels, Nutrisense, Signos), or need daily manual journaling (WHOOP Journal, Bearable).

**What is different about Body Lab**

| Body Lab does | Typical apps do |
| --- | --- |
| Notifies only when it has an answer: a solved case or a confirmed discovery | Alert that something was unusual (Fitbit Body Responses, high heart rate alerts on Apple Watch, Oura, WHOOP) |
| Investigates one surprising event against similar past situations | Compare with a general average |
| Treats findings as hypotheses and confirms them with natural experiments | Present insights as facts, never tested |
| Detects behavior from sensors, no journaling | Ask for daily check-ins or mood logs |
| Says "unexplained" when nothing stands out | Always produce an explanation, or none at all |
| Links findings across glucose, stress, sleep and movement | Keep each metric in its own silo |

We do not claim no one has built health insights before. The alert is only the starting point; the new part is the investigation and self-checking that follow it.

## Data

We use only the [BIG IDEAs Lab Glycemic Variability and Wearable Device Dataset](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/), because all four labs can then run on the same person.

| Item | Detail |
| --- | --- |
| People | 16 adults, aged 35 to 65, HbA1c 5.2 to 6.4% (high-normal to prediabetic) |
| Duration | 8 to 10 days each |
| Glucose | Dexcom G6, every 5 minutes |
| Wristband | Empatica E4: accelerometer 32 Hz, PPG 64 Hz (heart rate, interbeat interval), EDA 4 Hz, skin temperature 4 Hz |
| Food log | Date, time, food, amount, unit, calories, carbs, fiber, sugar, protein, fat |
| Other | Demographics.csv with HbA1c |
| Files | One folder per participant, one CSV per signal |
| License | Open Data Commons Attribution v1.0 |

**Inputs per lab**

All four labs run on the wristband. Glucose and meals are core inputs only for Fuel; elsewhere meals are a control or a limited suspect. **Measures** = what the lab watches for surprises; **Cause** = a possible explanation the agent tests; **Control** = used only to make comparisons fair.

| Data source | Fuel | Stress | Sleep | Movement and Rhythm |
| --- | --- | --- | --- | --- |
| Glucose (Dexcom) | **Measures** spike after eating | Not used | Cause: evening glucose | Not used |
| Food log | **Defines the meal** (time, carbs) | Control: after-meal hours compared with after-meal hours | Cause: dinner time (limited) | Cause: time since last meal (limited) |
| Accelerometer | Cause: steps before and after eating | Control: only still periods count; Cause: activity earlier | **Measures** sleep from night stillness; Cause: evening activity | **Measures** walks and daily activity timing |
| Heart rate and HRV | Cause: stress before eating | **Measures** stress signal | **Measures** overnight heart rate | **Measures** heart rate during and after walks |
| EDA | Cause: stress before eating | **Measures** stress signal | Cause: daytime stress | Cause: stress earlier that day |
| Skin temperature | Not used | Control: band-off check | Cause: warm nights | **Measures** daily temperature rhythm |

**What each lab compares**

| Lab | Compares | Example surprise | Example discovery |
| --- | --- | --- | --- |
| Fuel | Meal vs similar meal (similar carbs and time of day) | Same pasta: +25 vs +70 | Walking before lunch shrinks your spike |
| Stress | Hour vs the same hour on similar days | Tuesday 3pm far above your usual 3pm | Your stress signal peaks on weekday afternoons |
| Sleep (estimated) | Night vs other nights | Fell asleep 1 hour later than usual | Evening walks bring your sleep earlier |
| Movement and Rhythm | Walk vs similar walk; day vs similar day | Same 30-minute walk, heart rate 15 bpm higher | After short nights, your walks run 12 bpm higher |

One lab's result can be another's cause: last night's estimated sleep can explain a stress spike or a high-heart-rate walk the next day. Every lab skips stretches where skin temperature and EDA show the band was off.

**First-hour checks (go / no-go)**

- [ ] Count similar meal pairs per person (similar carbs and time slot). Fewer than \~3 pairs for most people = rethink Fuel Lab.
- [ ] Check overnight wristband coverage. Gaps at night = drop the Sleep Lab.
- [ ] Pick the demo participant: most complete days, most meal pairs.
- [ ] Confirm column names and timestamp formats in each CSV.

**Why not IMU50:** it has different people, no glucose or food logs, no labels, and a single 46.7 GB zip. Mention it only as future validation for the Sleep and Rhythm labs.

## Product

The user does almost nothing: the agent watches the data, opens cases on surprises, and turns repeated patterns into discovery cards.

**The investigation loop**

1. **Spot a surprise:** quietly, with no alert sent, a response far outside the person's normal range for a similar situation (for example, the same pasta lunch: +25 on Tuesday, +70 on Thursday).
2. **Check the data first:** sensor gaps, sensor warm-up, a missing snack or wrong meal time. Bad data is dismissed.
3. **Compare:** rank what was different against the person's own day-to-day variation (steps before eating, time of day, stress, carbs, starting glucose).
4. **Open a hypothesis:** the top difference becomes a lead, not a finding.
5. **Wait for natural experiments:** every later meal where the factor is present or absent counts as a test. No deliberate experiments needed.
6. **Confirm or reject:** repeated support makes a discovery card; otherwise it is logged as noise.

**Cross-lab findings** are the strongest feature because all labs use the same person: "meals eaten during a stress peak spike higher" or "late dinners raise overnight heart rate."

**Messages: answers, not alerts.** This is what separates Body Lab from Fitbit-style notifications.

- No real-time "something was unusual" alerts. The agent investigates silently.
- The user hears from it only with an answer: a solved case, a confirmed discovery, or a rejected guess.
- Every message leads with the why: "Thursday's lunch spiked because you barely moved before eating."
- Every message shows progress: "1 of 3 tests needed to confirm."
- Surprises with no clear reason close quietly and appear only as a weekly count and in the Notebook.

| Fitbit-style alert | Body Lab message |
| --- | --- |
| "Your heart rate was higher than usual during your walk." | "Your walk heart rate was 15 bpm above your usual for this walk. The biggest difference: you slept 5h 10m vs your usual 7h. Second time seen; 1 more to confirm." |
| (no follow-up) | "Confirmed: after short nights, your walks run 12 bpm higher. Held in 4 of 5 tests." |

**Gamification (passive, never tedious)**

| Feature | How it works | User effort |
| --- | --- | --- |
| Discovery cards | Each confirmed pattern becomes a card. Rarity: common, rare, legendary | None |
| Scientist rank | Intern, Lab Tech, Researcher, Professor. Rises with discoveries, never drops | None |
| Weekly quests | 1 to 3 optional quests from your own findings, completion detected by sensors | Optional |
| Weekly voice recap | 30-second summary read by ElevenLabs | Tap play |
| Discovery funnel | Shows how many surprises were dismissed as noise | None |

No competitive leaderboard: ranking people on health can backfire.

**The only manual input** is meals. For real users, Gemini estimates carbs from a meal photo; the demo uses the dataset's food logs. A small side feature shows photo logging: Gemini takes the photo and returns foods, portion, carbs, protein, fat, fiber, calories and a confidence level as JSON; the user confirms with one tap. Photo carbs are treated as rough, so similar-meal ranges are wider for them.

## Agent design

The agent's memory lives in Delta tables (its lab notebook), not in the language model; Gemini only sees a short summary of relevant rows when it handles a new event.

**Tools the agent can call**

| Tool | Input | Returns |
| --- | --- | --- |
| `check_data_quality` | event id | Sensor gaps, warm-up period, missing or suspect meal logs |
| `find_similar_situations` | event id, lab | Past events with similar carbs, time slot or activity |
| `compare_situations` | event id, list of similar ids | Each signal's difference, scored against personal variation |
| `get_personal_normal` | user, signal, context | Typical range (median and spread) for that person |
| `open_hypothesis` | claim, factor, lab | New row in `hypotheses` |
| `record_evidence` | hypothesis id, event id, supports or contradicts | Updated counts and status |
| `propose_quest` | discovery id | One optional quest the sensors can detect |

The agent chooses which tools to call and in what order. If a data check fails, it stops; if nothing stands out, it closes the case as unexplained.

**Lab notebook tables**

| Table | Key columns |
| --- | --- |
| `events` | event\_id, user, time, lab, signal, size of response, how unusual (percentile), verdict (bad data, unexplained, lead), hypothesis\_id |
| `hypotheses` | hyp\_id, user, lab, claim, factor, supports, contradicts, chances\_seen, status, opened\_at, last\_tested\_at |
| `discoveries` | card\_id, user, lab, claim, effect size, evidence (for example 4 of 5), rarity, confirmed\_at, last\_checked\_at, fading flag |
| `quests` | quest\_id, user, discovery\_id, target, progress, week |

**Hypothesis rules (example values, shortened for the 8 to 10 day dataset)**

| Outcome | Rule |
| --- | --- |
| Investigate | Event in the top 10% of unusual for that person and context |
| Confirmed | At least 3 supporting tests and at least 75% of tests agree |
| Rejected | Fails in most tests, logged as noise |
| Inconclusive | 5 chances without a clear result |
| Expired | No chance to test within the window (4 days in the demo, about 30 days in production) |
| Open slots | At most 10 open hypotheses per person, at most 2 of them about meals; the weakest is dropped first |
| Fading | A confirmed card contradicted by newer data is marked fading and rechecked |

**Data check rules (`check_data_quality`)**: starting values, tuned on the real data. If any check fails, the case closes as bad data before any comparison.

| Check | Rule | Why |
| --- | --- | --- |
| Glucose gaps | More than 15 minutes missing in the 2 hours after a meal | The peak may be missing |
| Sensor warm-up | Readings in the first 24 hours of a new sensor (look for long gaps if sensor changes are not marked) | New sensors read less accurately |
| Impossible jumps | Glucose changes more than about 15 to 20 mg/dL in 5 minutes | Sensor noise, not physiology |
| False lows | Sharp drop at night with a quick rebound | Lying on the sensor |
| Wristband off | Skin temperature well below normal, or EDA near zero | Band not on the wrist |
| Overlapping meals | Another meal or snack within 2 hours | Two curves blur into one |
| Wrong meal time | Glucose rises 15+ minutes before the logged time | Log time probably wrong |
| Unlogged meal | Big rise with no meal logged nearby | Food log incomplete |

**Normal range (`get_personal_normal`)**

- Default: a simple per-person model of expected glucose rise from carbs and time of day, fit on all of that person's meals. A surprise is a rise far from expected compared with how far off the model usually is (top or bottom 10%).
- With 4 or more truly similar meals (carbs within about ±20 g, same time slot), use their rises directly: outside the middle 80% is a surprise.
- Other signals (steps, heart rate, EDA, temperature): the person's typical value at that hour of day, weekday or weekend. This is the dashed band on the Case screen.

**Keeping meals from dominating findings**

1. **Match on meals:** Fuel compares meals with similar carbs, so "you ate more" is never the answer; Stress compares after-meal hours only with after-meal hours; Sleep compares nights with similar dinner timing when testing other causes.
2. **Known effects are baseline, not findings:** carbs raise glucose, digestion raises heart rate, exercise raises heart rate and EDA. Only what is left over can become a discovery. "An evening walk raises heart rate" is never a card; "after evening walks, your overnight heart rate is lower" can be.
3. **Separate candidate causes per lab**, with at most 2 meal-related open hypotheses.
4. **Obvious findings are common cards** worth little toward rank; rare and legendary go to patterns unusual for that person.

**Outlier handling:** one event never becomes a finding. Every meal tests open hypotheses, not just surprising ones, so the results are not biased toward confirmation.

**Wording rule:** the agent says "this was different," never "this caused." No medical advice.

## Architecture and stack

Everything except Gemini, ElevenLabs and the domain runs inside Databricks Free Edition.

&#91;embedded content: architecture · data flow from replay to app\]

The replayer stands in for a phone app uploading wristband and glucose data; everything after it is what production would run.

| Layer | Technology | Role |
| --- | --- | --- |
| Storage | Unity Catalog Volume + Delta tables | Raw CSVs, live tables, features, lab notebook |
| Streaming | Replayer notebook + Auto Loader (Structured Streaming) | Replays a week in a few minutes |
| Features | Spark SQL and pandas | Per-minute signals, `meal_features`, `personal_normal` |
| Agent | Gemini API function calling + Python tools | Investigation loop, explanations, weekly summary |
| Tracing (optional) | MLflow tracing | Logs each agent run so judges can see the tool calls |
| App | Databricks App with Streamlit and Plotly | Four screens |
| Voice | ElevenLabs text-to-speech | 30-second weekly recap |
| Domain | GoDaddy Registry | Public URL |
| Dev | GitHub + Copilot | Repo and faster coding |

Not used: Solana (health data on a public chain is a privacy problem) and Tiger Data (duplicates Delta tables).

## Screens and design

The app has four tabs, already shown in the [Body Lab mockup](https://claude.ai/artifact/7PqmPr3Fmg89AdvNkDaDiV).

| Tab | What it shows | Must have for demo |
| --- | --- | --- |
| Today | Confirmed discovery and solved case cards (each leading with the why), count of surprises closed quietly, scientist rank, weekly quest, the four labs, voice recap button | Solved case card, discovery card, quiet-close count |
| Case | Two glucose curves on one chart, data check, ranked differences against the normal band, verdict | All of it (hero screen) |
| Discoveries | Cards with rarity (legendary, rare, common, fading), one locked card close to confirmation | 4 to 6 cards |
| Notebook | Open hypotheses with evidence dots, discovery funnel, recently rejected hypotheses | Hypotheses + funnel |

**Design rules**

- Lab-notebook look: graph-paper background, mono font for numbers, one accent color.
- Glucose is always the warm color; context lines (the comparison meal) are gray and dashed.
- Every number shown comes from the Delta tables, never hard-coded in the final build.
- Built as a Databricks App in Streamlit. Charts with Plotly or Altair.

## Tasks and timeline

The plan assumes a 24-hour build with four people (Data, Agent, App, Pitch); with three, the Pitch tasks split between App and Agent.

&#91;embedded content: 24-hour build plan · 5 phases, 3 gates\]

If the first full case does not work by hour 14, cut to the Fuel Lab only and spend the rest on the Case screen.

**Before the event**

- [ ] All: create Databricks Free Edition accounts and confirm serverless notebooks and Databricks Apps work
- [ ] Agent: get a Gemini API key from Google AI Studio; test one tool-calling request
- [ ] Pitch: claim the ElevenLabs promo code; pick a voice
- [ ] Data: read the BIG IDEAs file list; download the dataset if the rules allow it
- [ ] All: sign up for the GitHub Student Developer Pack (Copilot)

**Hours 0 to 2: setup and go/no-go (all)**

- [ ] Data: upload CSVs to a Unity Catalog Volume; load one participant into Delta tables
- [ ] Data: run the first-hour checks (meal pairs, overnight coverage, demo participant)
- [ ] Agent: confirm whether serverless streaming allows only `availableNow`; plan the loop fallback
- [ ] App: hello-world Databricks App reading one Delta table
- [ ] Pitch: register the GoDaddy domain (promo MLHWH2697); start the README
- [ ] Gate: go/no-go on Fuel Lab and Sleep Lab

**Hours 2 to 8: pipeline and features**

- [ ] Data: downsample wristband signals to per-minute (steps, mean heart rate, mean EDA, mean temperature)
- [ ] Data: replayer script writing 30-minute chunks every 2 seconds into the stream folder
- [ ] Data: Auto Loader streams into `glucose_live`, `wrist_live`, `meals_live`
- [ ] Data: `meal_features` table, one row per meal: carbs, time slot, start glucose, rise and peak in 2 hours, steps/heart rate/EDA in the hour before
- [ ] Agent: `personal_normal` table (median and spread per person, signal, context)
- [ ] Agent: create the notebook tables (`events`, `hypotheses`, `discoveries`, `quests`)
- [ ] App: Today and Case screens on static sample rows

**Hours 8 to 14: agent**

- [ ] Agent: surprise detector (top 10% unusual) writing to `events`
- [ ] Agent: implement the seven tools as Python functions over Delta tables
- [ ] Agent: Gemini tool-calling loop with a system prompt (wording rule, unexplained allowed)
- [ ] Agent: hypothesis rules (confirm, reject, inconclusive, expire, 10-slot cap)
- [ ] Data: Stress Lab features (EDA peaks during still periods), estimated sleep from night stillness, and walk detection (10+ minutes of steady movement) with heart rate during and after each walk
- [ ] App: wire Case screen to real `events` and `compare_situations` output
- [ ] Gate: one full case works end to end on the demo participant

**Hours 14 to 20: app, gamification, extras**

- [ ] App: Discoveries and Notebook screens, funnel from real counts
- [ ] Agent: rarity rule (common if most participants share the pattern, rare if few do)
- [ ] Agent: quest generator and sensor-based completion check
- [ ] Agent: one cross-lab finding (stress before meal vs spike size)
- [ ] Pitch: ElevenLabs weekly recap from the agent's summary text; App: Gemini meal-photo logging side feature (photo in, foods and macros as JSON out)
- [ ] Agent: planted-test validation: add a fake walk before a meal and show the agent finds it

**Hours 20 to 24: polish and submit**

- [ ] All: code freeze at hour 22; only demo fixes after
- [ ] Pitch: slides, 2-minute video backup, Devpost write-up
- [ ] All: rehearse the demo three times with the replay at fixed speed
- [ ] Pitch: submit to Databricks track, Gemini, ElevenLabs and GoDaddy categories

**Cut list if behind:** Movement and Rhythm Lab, then Sleep Lab, then meal-photo logging, then quests, then voice recap. Never cut the Case screen or the data check.

## Team roles

Each role owns one layer end to end, so no one waits on another person for more than an hour.

| Role | Owns | Skills needed | Person |
| --- | --- | --- | --- |
| Data | Upload, replayer, streaming tables, meal and lab features | Python, pandas, Spark SQL |  |
| Agent | Surprise detector, tools, Gemini loop, hypothesis rules, quests | Python, LLM tool calling |  |
| App | Databricks App (Streamlit), four screens, charts | Python, Streamlit, Plotly |  |
| Pitch | Domain, ElevenLabs recap, slides, video, Devpost, demo script | Writing, presenting |  |

Sync every 3 hours for 5 minutes: what works, what is blocked, what gets cut.

## Demo script (3 minutes)

The demo shows one case solved live, then proves the agent rejects noise.

1. **Hook (0:00 to 0:20):** "1 in 3 US adults has prediabetes. Glucose monitors are now sold over the counter, but they show curves, not reasons."
2. **Start the replay (0:20 to 0:40):** Databricks streams one participant's week at high speed. Meals and glucose appear on the Today screen.
3. **The case (0:40 to 1:30):** no alert fires; a moment later a solved case appears: "Thursday's lunch spiked because you barely moved before eating." Open it: same pasta, +25 vs +70. Data check passes. The agent ranks the differences: about 2,000 fewer steps before eating. Verdict: a lead, hypothesis H7 opened.
4. **Confirmation (1:30 to 2:00):** the replay continues. Later meals with and without a walk add evidence dots. H7 becomes a rare discovery card; the scientist rank rises.
5. **Rigor (2:00 to 2:30):** the Notebook funnel shows most surprises dismissed as bad data or normal variation. Show the planted-test check: the agent finds a fake walk we inserted.
6. **Close (2:30 to 3:00):** play the 30-second ElevenLabs recap. "Health apps give you scores. Body Lab gives you a scientist." Show the domain.

**Backup:** a recorded 2-minute video in case the live replay fails.

## Risks and judge questions

The biggest risk is too few comparable meals; the hour-2 gate catches it before we build on it.

**Build risks**

| Risk | Fallback |
| --- | --- |
| Few similar meal pairs per person | Loosen similarity (carb range, any time slot); pick the participant with the most pairs |
| No overnight wristband data | Drop the Sleep Lab; keep three labs |
| Serverless streaming only runs `availableNow` | Run that trigger in a loop every few seconds |
| Gemini tool calling flaky | Fixed tool order with Gemini writing only the explanation |
| Databricks App deploy issues | Run Streamlit locally against Databricks SQL |
| Running out of time | Follow the cut list in Tasks and timeline |

**Prepared answers**

| Judge asks | Answer |
| --- | --- |
| Isn't this just Fitbit-style notifications? | Fitbit tells you something unusual happened. Body Lab tells you why, checks itself over the next week, and only then calls it a finding. It never sends a bare alert. |
| Isn't one surprising event just an outlier? | Maybe. That's why one event never becomes a finding. We check data first, then require the pattern to repeat. The funnel shows most surprises are dismissed. |
| Won't meals explain everything? | We match situations on meals and treat known effects as baseline, so the agent finds what else changed. At most 2 meal hypotheses can be open. |
| How do you know the explanations are right? | Planted tests: we insert a known change and the agent finds it. It also rediscovers known effects such as walking lowering spikes. |
| Doesn't Levels or WHOOP already do this? | They give scores or need journaling. We investigate single events and confirm findings from everyday life automatically. |
| Why is this agentic and not a script? | The agent picks which tools to call, stops on bad data, and can close a case as unexplained. |
| Is the data really streaming? | We replay recorded data through Databricks streaming. A phone app would feed the same pipeline live. |
| Isn't the food log unreliable? | Yes, so the agent flags suspect logs first. Real users would log by meal photo with Gemini. |
| Is this medical advice? | No. It reports what was different, never causes or treatment. |
| Only 16 people? | The agent works per person, so what matters is days per person. We say clearly it is a prototype. |

## Submission checklist and prizes

We enter one track and three MLH categories, each with a real role in the app.

| Prize | Award | Our use |
| --- | --- | --- |
| Applied AI Software (Databricks), 1st | Fujifilm Instax Mini 12 | Whole project |
| Applied AI Software, 2nd / 3rd | ELEGOO UNO R3 kit / Anker power bank | Whole project |
| Best Use of Gemini API | MLH swag kit | Agent's reasoning, tool calls and explanations |
| Best Use of ElevenLabs | Wireless earbuds | Weekly voice recap |
| Best Domain Name from GoDaddy Registry | Digital gift card | Project domain (promo MLHWH2697) |

- [ ] Public GitHub repo with README: problem, architecture, how to run, dataset citation and license
- [ ] Cite the BIG IDEAs dataset and PhysioNet as the dataset page asks
- [ ] Devpost page with screenshots, 2-minute video, and the track plus each MLH category selected
- [ ] Live domain pointing to the app or a landing page
- [ ] Slides: problem, demo, agent loop, rigor funnel, team
- [ ] Note in the README that the data is replayed, not live
- [ ] Follow the MLH Code of Conduct
