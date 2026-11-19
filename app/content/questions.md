<!--
All text shown in the app lives here, so you can change wording without touching code.

Format: each block starts with a line "## <key>" and runs until the next "## ".
Words in {curly_braces} are replaced with numbers the app computes live; if you
add a new {name} that the app doesn't know, it is shown as-is.
-->

## home.intro
This app turns the ACMAD seasonal-calibration practical into things you can move and watch change. You don't need the original notebook: everything is explained here, and the numbers are computed live from the same data.
Each lab follows the same rhythm:

1. **Predict first.** Answer a question before you touch anything.
2. **Play.** Move one or two sliders and watch the plot.
3. **Explain simply.** A short plain-language explanation, with the maths folded away.
4. **Check yourself.** A discussion question from the practical, with an answer built from the numbers on your screen.

The sidebar holds the experiment settings (see *The setup* below). Leave them at their defaults unless you want to see how much the results depend on them.

## home.pipeline
**The whole practical in one line:**
SEAS5 says how much rain to expect → **IDR** turns that single number into a full set of probabilities (a *predictive distribution*) → we cut that distribution into three **tercile probabilities** (below / near / above normal) → we **score** them against what CHIRPS actually observed, and compare with simply forecasting climatology.

## home.setup
**The question.** Can a seasonal forecast model's rainfall prediction be turned into *trustworthy probabilities* for the season ahead?

**The two ingredients**, one value each per season and location:

- **SEAS5** is ECMWF's seasonal forecast system. Here it is a single number: the October–December (**OND**) rainfall total predicted in early September, before the season starts. It is *not* the individual ensemble members.
- **CHIRPS** is what actually fell: a rainfall record built from satellite estimates blended with rain-gauge data. We treat it as the truth.

**The data.** {n_years} seasons ({y0}–{y1}) × {n_loc} locations across Africa. We don't know where each location is, so there are no maps.

**The fair test.** The model only learns from **{y0}–{split}** (training years). It is then judged on **{split1}–{y1}** (test years), seasons it has never seen, exactly as a real forecast would be. To keep things fast, we draw **{n_train} random location-season pairs** from the training years and **{n_test}** from the test years (random seed {seed}).

**One caution.** Neighbouring locations in the same season share the same weather, so these pairs are *not* thousands of independent forecasts. There are really only {n_test_years} independent test seasons.

## lab2.intro
SEAS5 gives you **one number** for the season, for example "280 mm". But the real rainfall could be 150 mm or 400 mm. What forecasters need is the **chance** of each outcome. **Isotonic distributional regression (IDR)** learns those chances from past seasons, with one sensible rule: *a wetter forecast should never make dry outcomes more likely.*

## lab2.predict.q
Imagine SEAS5 predicts more and more rain. What should happen to the **chance that the season ends up below {t_mm} mm**: go up, go down, or stay the same?

## lab2.predict.a
It should **go down** (or at least never go up). If SEAS5 is any use, wetter forecasts come with wetter outcomes, so being below a fixed amount becomes less likely. IDR *forces* this: it is the "isotonic" (order-keeping) part of the name. In the plot below, the blue curve can only go down as you move right.

## lab2.threshold.explain
**Each dot is one past season at one location.** It sits at the top (1) if the season stayed **at or below** the threshold you chose, and at the bottom (0) if it was wetter. The dots are jittered up and down slightly so they don't all overlap.

The **blue curve** is IDR's estimate of "the share of seasons that stayed below the threshold, for a forecast of this size". It is like a running average of the dots, with one rule: **it may only go down as SEAS5 increases**.

Move the threshold. Every threshold gives a different curve. **IDR is simply this, done for every possible threshold at once.** Stack the answers and you get a full probability distribution for any SEAS5 value.

## lab2.threshold.maths
For a threshold $t$, IDR estimates $F(t \mid x) = P(Y \le t \mid X = x)$ by **antitonic least squares**: among all functions that never increase in $x$, pick the one closest to the 0/1 outcomes $\mathbf{1}\{y_i \le t\}$. The pool-adjacent-violators algorithm (PAVA, next section) finds this exactly. Doing it for every observed $t$ gives a CDF for each $x$ (Henzi, Ziegel & Gneiting, 2021).

## lab2.pava.explain
How does IDR make the curve go only downward? It uses **pool adjacent violators (PAVA)**. Walk from the driest forecast to the wettest. Whenever a wetter forecast shows a *higher* share of dry outcomes than its neighbour, that breaks the rule, so **pool them**: replace both with their average. Repeat until nothing breaks the rule. Step through it below on 30 seasons.

## lab2.cdf.explain
Now fix a **SEAS5 value** and look at the full answer IDR gives: the **predictive CDF**. Read it like this: *at x mm on the horizontal axis, the curve's height is the chance the season ends up at or below x mm.*

- Where the curve crosses **0.5** is the median: a 50/50 rainfall amount.
- The steeper the curve, the more confident the forecast.

The dashed pink curve is the "obvious" alternative: a straight-line regression plus a bell-shaped (Gaussian) error. Compare the two. The Gaussian assumes the same shape of uncertainty everywhere, and it can even give probability to **negative rainfall**. IDR's shape comes from the data.

## lab2.cdf.maths
IDR CDF: $F(t \mid x)$ from the fitted model, interpolated linearly between the training SEAS5 values on either side of $x$.
Linear-Gaussian: $Y = a + bX + \varepsilon$, $\varepsilon \sim N(0, \sigma^2)$, so $F(t \mid x) = \Phi\big((t - a - bx)/\sigma\big)$.
Quantiles: $Q(p \mid x) = \inf\{t : F(t \mid x) \ge p\}$.

## lab2.check.q
*Discussion question:* At a fixed rainfall threshold, which forecast has the highest probability of a dry outcome? What shape or monotonicity restrictions do you see?

## lab2.check.a
At the threshold of **{t_mm} mm**, the four example forecasts (a dry, two middling and a wet SEAS5 value) give these probabilities of staying at or below it:
{example_table}

The **driest forecast ({x_dry} mm)** has the highest probability of a dry outcome. The restrictions are:

- **Across forecasts:** the curves never cross. A wetter forecast's CDF always lies on or below a drier one's. This is the monotonicity IDR imposes.
- **Within each curve:** it is a *step* function that only rises from 0 to 1, because it is built from the observed rainfall amounts in the training data.

## lab3.intro
Forecasters don't publish a whole curve. They publish **three numbers**: the chance the season is **below normal (BN)**, **near normal (NN)** or **above normal (AN)**. This lab shows how those three numbers come straight out of the curve from Lab 2.

## lab3.predict.q
SEAS5 is forecasting a **very wet** season. Which of the three probabilities, P(BN), P(NN) or P(AN), will grow the most?

## lab3.predict.a
**P(AN)** grows. A wetter forecast pushes the whole curve to the right, so less of it lies below the cutoffs. That means P(BN) shrinks and P(AN) = 1 − F(upper cutoff) grows. Try it with the slider.

## lab3.cdf.explain
First, define "normal": take all **training** seasons and find the two rainfall amounts that split them into three equal groups. These are the **tercile cutoffs**, q⅓ = {q1_mm} mm and q⅔ = {q2_mm} mm (the dashed vertical lines).

Then read the curve where it crosses each line:

- **P(BN)** = height of the curve at q⅓: the chance of ending at or below the lower cutoff.
- **P(NN)** = height at q⅔ minus height at q⅓: the chance of ending between the two.
- **P(AN)** = 1 − height at q⅔: everything above the upper cutoff.

The three always add up to 100%. That's it: **three probabilities read off one curve.**

## lab3.cdf.maths
$P(BN) = F(q_{1/3} \mid x)$, $\quad P(NN) = F(q_{2/3} \mid x) - F(q_{1/3} \mid x)$, $\quad P(AN) = 1 - F(q_{2/3} \mid x)$.
The categories are BN: $y \le q_{1/3}$; NN: $q_{1/3} < y \le q_{2/3}$; AN: $y > q_{2/3}$ (as in `np.digitize(..., right=True)`).
Cutoffs use training years only, so no information from the test years leaks in.

## lab3.local.explain
**This is the important twist.** The practical uses **one pair of cutoffs for all of Africa** (*pooled* cutoffs). But "normal" in the Sahara is about 0 mm, while "normal" in the humid tropics is hundreds of mm.

Pick a **dry location** with **pooled** cutoffs. Its seasons are below the pooled q⅓ almost every year, so the forecast says "BN" nearly every year, and is nearly always right. That looks like great skill, but it only shows that the place is dry, not that this year will be dry.

Switch to **local** cutoffs (this location's own training terciles). Now "below normal" means "drier than usual *for this place*", which is what a real forecast must say. The pooled IDR model was never trained for that question. In Lab 5 you'll see what that does to the scores.

## lab3.table.explain
Across all **{n_test}** held-out forecasts, the average forecast probability for each category should be close to how often that category actually happened. This is a first, rough calibration check.

## lab3.check.q
For the **{loc_label}** location, why do pooled cutoffs make the forecast look so certain, and why is that misleading?

## lab3.check.a
With pooled cutoffs, this location's 15 training seasons fell into BN / NN / AN **{pooled_counts}** times. With its own local cutoffs it is **{local_counts}**. Under pooled cutoffs the category mostly reflects **where** the location is, not **what this year** is like. A forecast can be "right" every year for a dry location just by always saying BN. Official forecasts therefore use **local** terciles from a fixed reference period (often 1991–2020). The supplied WMO mean ± standard deviation can't replace those, because a mean and a spread don't determine the tercile cutoffs.

## lab1.intro
Before any statistics, look at the raw material. For each of the {n_loc} locations and {n_years} October–December seasons there are two numbers: what **SEAS5** forecast in early September, and what **CHIRPS** says actually fell. Everything later in the practical is built from these pairs.

## lab1.predict.q
Across every location and season, SEAS5 and CHIRPS have a correlation of **r = {r_raw}**, which is very high. Does that mean SEAS5 knows whether *this* October–December will be wetter or drier than usual?

## lab1.predict.a
**Not necessarily, and mostly not.** Africa contains deserts that get almost nothing and humid regions that get hundreds of millimetres. A forecast that only knows *which place is wet and which is dry* already lines up the points along the diagonal. To see whether SEAS5 knows anything about *this year*, you have to remove each place's usual amount. Try the "Anomalies" view below.

## lab1.scatter.explain
**Raw totals:** each dot is one location in one season. The cloud hugs the diagonal, giving r = **{r_raw}**. But most of that pattern comes from the *spread between places*: desert points sit near 0, wet points far up the diagonal.

**Anomalies:** subtract each location's own 25-season average from both SEAS5 and CHIRPS. Now each dot shows "how much wetter or drier than usual" the forecast and the observation were. The correlation drops to **{r_anom}**. That is the part of the skill that is about *the season* rather than *the place*, and it is much weaker.

Keep this in mind for the rest of the labs: **a big score can come from geography alone.**

## lab1.scatter.maths
Pearson correlation $r = \dfrac{\sum (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum (x_i-\bar x)^2 \sum (y_i-\bar y)^2}}$ over all location-seasons.
Anomalies: $x'_{s,\ell} = x_{s,\ell} - \bar x_\ell$ and $y'_{s,\ell} = y_{s,\ell} - \bar y_\ell$, where $\bar x_\ell$ is location $\ell$'s mean over the 25 seasons. (This descriptive view uses all 25 seasons; the models later use training years only.)

## lab1.bias.explain
**Bias** is SEAS5 minus CHIRPS: positive means SEAS5 forecast too much rain. Each yellow point is the average bias for one tenth of all forecasts, from the driest forecasts on the left to the wettest on the right. The grey bars show the middle half of individual seasons.

- For dry forecasts SEAS5 is slightly too *dry*, around **{bias_lo} mm**. For the wettest forecasts it is far too *wet*, around **{bias_hi} mm**.
- A single correction, subtracting the overall average of **{bias_all} mm**, would make dry places worse and still leave wet places badly off.
- Even inside one group, individual seasons scatter widely (the grey bars). A corrected single number would still hide that uncertainty.

That is why the practical doesn't just subtract a bias. It builds a **full probability distribution** for each forecast value (Lab 2).

## lab1.location.explain
Pick a dry, typical or wet place and follow it through all 25 seasons. The shaded part on the right is the **test period**: the model never sees those seasons while it is being built, so they can judge it fairly.

Notice two things. First, SEAS5 usually gets the **level** about right: wet places are forecast wet, dry places dry. Second, it often misses the **ups and downs** from one season to the next. The "year-to-year r" measures exactly that second thing. The dotted line is the WMO reference mean supplied with the data: a long-term average for this location, not a forecast.

## lab1.r.explain
This histogram repeats the "one location" calculation for every location: how well SEAS5 follows the season-to-season ups and downs, place by place.

- The **median is only {r_med}**, far from the pooled **{r_raw}**.
- About **{pct_neg}** of locations have a *negative* r: in those places SEAS5 tends to go the wrong way.
- **{pct_weak}** of locations have r below 0.3, which is a weak relationship.

With only 25 seasons per location, some of this scatter is just chance. But the message is clear: **SEAS5 knows the geography very well, and the year-to-year signal much less.** Lab 5 shows what that does to the skill scores.

## lab1.check.q
*Discussion question:* Does bias vary by rainfall amount? Would subtracting the average bias capture forecast uncertainty? Why might a whole predictive distribution be more useful than one corrected rainfall value?

## lab1.check.a
- **Yes, bias depends strongly on the amount.** The average bias runs from about **{bias_lo} mm** for the driest forecasts to **{bias_hi} mm** for the wettest. The overall average, **{bias_all} mm**, describes almost no one.
- **No, subtracting the average bias can't capture uncertainty.** It moves one number up or down, but the seasons around that number still scatter widely (the grey bars), and that scatter is different for dry and wet forecasts.
- **A predictive distribution** says how likely each outcome is: the chance of a dry season, a normal one, or a very wet one. Decision-makers need those chances, for example "a 60% chance of below-normal rain". It also stays honest where the forecast is weak: at a typical location the year-to-year r is only **{r_med}**, so the distribution should stay wide.

## lab4.intro
Lab 3 turned each forecast into three probabilities. Now judge them on the **test seasons** the model never saw. No single score tells the whole story: each one answers a different question, and each one is blind to something.

## lab4.predict.q
A forecaster issues 70% for "above normal" on many occasions, and above normal happens on about 70% of them. Is that forecaster necessarily useful?

## lab4.predict.a
**Not necessarily.** That shows the probabilities are **reliable** (they mean what they say). But a forecaster who always says the long-term average, 33%, is reliable too, and tells you nothing new. A useful forecast must also **change** with the situation, issuing high probabilities when the event is likely and low ones when it isn't. That is **resolution**. Sections A and B below separate the two.

## lab4.brier.explain
The **Brier score** treats each category as a yes/no event. Every forecast gives a probability *p*; the outcome is 1 if the event happened and 0 if it didn't. The score is the average of $(p - \text{outcome})^2$. **0 is perfect** and lower is better.

On its own a Brier score is hard to read, so compare it with **climatology**: always forecasting the training-years frequency (about ⅓). The **Brier skill score (BSS)** is the improvement: 1 = perfect, 0 = no better than climatology, below 0 = worse.

The bars split the score into three parts:

- **Reliability** (a penalty): do the probabilities mean what they say?
- **Resolution** (a reward): do the forecasts *change* when the outcome changes?
- **Uncertainty**: how hard the event is to forecast. It is the same for every forecaster.

Climatology always has **zero resolution**, because it never changes. So any skill IDR has must come from resolution: compare the middle bars.

## lab4.brier.maths
$BS = \frac1n \sum_i (p_i - o_i)^2$, $\quad BSS = 1 - BS_{IDR}/BS_{clim}$.
Murphy decomposition on forecasts replaced by their bin averages $\bar p_k$: $BS_{binned} = \underbrace{\tfrac1n\sum_k n_k(\bar p_k - \bar o_k)^2}_{REL} - \underbrace{\tfrac1n\sum_k n_k(\bar o_k - \bar o)^2}_{RES} + \underbrace{\bar o(1-\bar o)}_{UNC}$.
The raw BS differs from the binned BS by a small within-bin term, shown above so the numbers visibly add up.

## lab4.reliability.explain
Group the forecasts by probability, for example all forecasts between 60% and 70%. Then ask how often the event really happened in that group. **Points on the diagonal are reliable**: 65% forecasts came true 65% of the time. Points below the diagonal mean IDR was over-confident; above means under-confident.

The **sharpness** chart on the right shows how many forecasts fell in each bin. Bins with fewer than 10 forecasts (hollow grey points) are too small to judge; here **{n_sparse}** of the **{n_bins_used}** filled bins are like that. Try more bins: the points get noisier because each bin holds fewer forecasts.

The dotted horizontal line is how often the event happened overall. A forecast with no resolution would sit on that line everywhere.

## lab4.roc.explain
Reliability asks whether the probabilities are honest. **ROC** asks something else: **can the forecast tell event seasons from non-event seasons at all?**

Imagine issuing a warning whenever the probability is above the cutoff. Each cutoff gives a **hit rate** (share of events you warned about) and a **false alarm rate** (share of non-events you warned about). Sliding the cutoff from 100% down to 0% traces the ROC curve. A curve hugging the top-left corner means good discrimination; the dotted diagonal is guessing.

The **area under the curve (AUC)** sums this up: 0.5 = chance, 1 = perfect ranking. AUC **ignores calibration completely**: multiply every probability by 0.5 and the AUC does not change, but the forecasts become badly unreliable.

## lab4.roc.maths
Hit rate $= \frac{\text{hits}}{\text{hits} + \text{misses}}$, false alarm rate $= \frac{\text{false alarms}}{\text{false alarms} + \text{correct negatives}}$.
AUC = the probability that a randomly chosen event case gets a higher forecast probability than a randomly chosen non-event case. It depends only on the *ranking* of the probabilities.

## lab4.stability.intro
Every number above came from **one** random draw of pairs and **one** particular set of ten test seasons. Two kinds of "different luck" could change them:

- **Different random pairs:** draw the training and test pairs again with 50 other seeds.
- **Different seasons:** resample the 10 test seasons (with replacement), keeping the fitted model fixed. This asks what would happen with a different mix of years.

## lab4.stability.explain
The bars on the left show the middle 90% of the reruns for each kind of luck. The right-hand chart scores each test season on its own. Season to season, the BSS for {cat} runs from **{worst_bss}** (in {worst}) to **{best_bss}**.

With only **{n_years}** test seasons, one unusual season can move the overall score noticeably. Report scores with this spread, not as a single exact number.

## lab4.check1.q
*Interpretation exercise:* Compare the three ROC AUC values with their Brier skill scores. Which category best separates event and non-event cases? Can high ROC AUC coexist with poor reliability? Why should scores pooled across thousands of locations **not** be interpreted as thousands of independent forecast years?

## lab4.check1.a
- **AUC:** BN {auc_bn} · NN {auc_nn} · AN {auc_an}. **BSS:** BN {bss_bn} · NN {bss_nn} · AN {bss_an}. **{best_auc}** separates events from non-events best and **{worst_auc}** is the hardest. Near normal is usually the hardest, because it sits between the other two and is easy to confuse with either.
- **Yes, high AUC and poor reliability can coexist.** AUC only checks the *order* of the probabilities. A forecast that ranks seasons perfectly but always says 90% when the truth is 50% has excellent AUC and terrible reliability. Check the reliability diagram too.
- **Not independent:** the {n_test} test pairs come from only **{n_years} seasons**. Within one season, locations share the same large-scale weather (for example El Niño or the Indian Ocean Dipole), so their errors move together. The honest sample size is closer to {n_years} seasons than to {n_test} forecasts; see section D and Lab 5E.

## lab4.check2.q
*Discussion question:* Which tercile has positive BSS and the highest ROC AUC? Are some probability bins poorly sampled? How could pooling across dissimilar rain regimes produce misleading regional skill?

## lab4.check2.a
- Positive BSS here: **{pos_text}** (BN {bss_bn}, NN {bss_nn}, AN {bss_an}). The highest BSS is **{best_bss}** and the highest AUC is **{best_auc}**.
- **Some bins are poorly sampled.** IDR's pooled forecasts are often near 0% or near 100% (a desert is almost certainly BN), so the middle bins can hold few forecasts. Bins with fewer than 10 forecasts, with the notebook's 8 bins: {sparse}.
- **Pooling across rain regimes:** with one set of cutoffs for all of Africa, "below normal" mostly means "a dry place". Telling the Sahara from the Congo Basin is easy, so the scores look excellent, but that says nothing about forecasting *this season* at *one place*. Lab 5A measures how much of the skill disappears.

## lab5.intro
Labs 1–4 reproduced the practical. This lab asks **what the numbers really mean**. Each tab is one trap: flip the switch and watch the headline number change.

**What comes from the notebook and what is new.** The notebook's own setting is ① in tab A: one IDR model fitted on raw totals with all locations pooled, scored against pooled cutoffs. The notebook *raises* the questions behind tabs B–E, but it does not run these experiments. Everything else in this lab was added for this app: the location-aware anomaly model (③), the local cutoffs and local climatology, the leakage switches, the shuffle controls and the season resampling.

## lab5.predict.q
The notebook's pooled IDR reaches BSS **{bss_bn}** for below normal: a huge improvement over climatology. If you judged the same forecasts against each location's *own* idea of "below normal", what do you expect?

## lab5.predict.a
**The skill largely disappears.** Judged against each location's own cutoffs, the same pooled forecasts {v2}. Tab A shows it: the big number mostly measured whether SEAS5 can tell a dry *place* from a wet one, not a dry *season* from a wet one.

## lab5.a.1
**① The notebook's setting: BN {b1}.** One pair of cutoffs for all of Africa. "Below normal" mostly means "a dry place", and SEAS5 knows the geography very well (r = 0.94 in Lab 1). Here the pooled model {v1}.
**The question it really answers:** *can SEAS5 tell a dry place from a wet one?*

## lab5.a.2
**② Same model, local cutoffs: BN {b2}.** Now "below normal" means "drier than usual *for this location*", and the yardstick is that location's own climatology. Against that yardstick the pooled model {v2}. It was never trained for this question: it learned "SEAS5 says 20 mm → probably dry", which is true everywhere in a desert and says nothing about *this* year.
**The question it answers:** *is the pooled model calibrated for local events?* Compare its score with ①: only the definition of the event changed.

## lab5.a.3
**③ Location-aware model: BN {b3}.** Fit IDR on *anomalies*: each location's SEAS5 and CHIRPS are standardised by that location's own training mean and spread, so the model can only learn "wetter than usual → wetter than usual". Scored against local cutoffs, it {v3}.
**The question it answers:** *how much year-to-year signal is there?* {ans3}

*Not in the notebook.* The notebook fits only the pooled raw-total model (①). This anomaly model was added for this app to estimate the year-to-year skill that ① cannot measure.

## lab5.a.explain
**The two-part lesson:**

1. **The big number was geography.** Changing only the *definition of the event* (① → ②) moves the BSS for below normal from **{b1}** to **{b2}**.
2. **The honest year-to-year skill is what a location-aware model finds** (③): here it {v3}.

This is why official forecasts use **local** terciles, usually from a fixed reference period such as 1991–2020.

## lab5.a.maths
Local cutoffs: $q_{1/3,\ell}, q_{2/3,\ell}$ from location $\ell$'s training seasons. Local reference: the empirical training frequencies of BN/NN/AN at $\ell$ (not exactly ⅓ when rainfall values tie).
Anomaly model: $x' = (x - \bar x_\ell)/s_{x,\ell}$, $y' = (y - \bar y_\ell)/s_{y,\ell}$ with training-year means and standard deviations; IDR is fitted on $(x', y')$, and the local cutoffs are transformed the same way.
All three rows score the same test pairs. Locations with equal cutoffs, or with training CHIRPS spread ≤ 1 mm, are left out of every row.

## lab5.b.intro
**Leakage** means information from the test seasons sneaks into the model or the yardstick. The rule in the practical is that cutoffs, climatology and the model all use **training years only**. Switch on one leak at a time and see which way the score moves. Try both the pooled and the local setting.

## lab5.b.explain
- **Cutoffs from all 25 years** change *which event is being scored*. The test seasons help define "normal", so the categories shift slightly towards the test-year outcomes.
- **Climatology from all 25 years** changes *the yardstick*. The reference now partly "knows" how often each category happened in the test years, so it becomes unrealistically good and the honest model looks **worse**. In the pooled setting the effect is tiny, because 12,424 locations average out. In the local setting it is large: each location has only 25 seasons, and 10 of them are the test.
- **Training IDR on the test pairs** lets the model see the answers, so it looks **better**.

**The lesson:** leakage can push the score either way, so you can't tell from the direction alone whether something leaked. The only safe rule is to keep test-season information out of every step.

## lab5.c.intro
Can a clever calibration method squeeze skill out of a forecast that has none? Test it: **shuffle** the training SEAS5 values, so they no longer match their seasons, then refit IDR and score it on the same real test cases. Five versions are compared below.

## lab5.c.explain
- **Real SEAS5** has high skill, as in Lab 4.
- **Shuffled SEAS5** collapses to about **{shuf}**. Almost all the skill is gone, as it should be. But it isn't exactly zero.
- **Why not zero?** IDR still *forces* the forecasts to be ordered: a wetter test forecast must get a wetter distribution. The fitted probabilities are almost flat, but they are ranked by the test-season SEAS5 value, and that value still says a lot about the place. Evidence:
  - force the order the **wrong** way (wetter → drier) and the score turns negative ({wrong});
  - shuffle the **test** SEAS5 too and the gain disappears ({both}).
- **No SEAS5 at all** (always forecast the training climatology) scores about {const}.

**The lesson:** calibration cannot create predictability that the forecast doesn't contain. IDR's one assumption, "wetter forecast → wetter outcome", is itself a small piece of knowledge, and here it is worth a few hundredths of BSS for {cat}.

## lab5.c.maths
Each of the 100 shuffles permutes the 1,200 training SEAS5 values (seeds 0–99) and refits IDR. The wrong-direction version fits IDR on $-x$ and predicts at $-x_{test}$, which forces the opposite ordering. (The library's `increasing=False` option is not used: its predictions are currently wrong.)

## lab5.d.intro
The take-home question asks: *what does it mean for a forecast to be reliable but have little skill?* Compare IDR with the laziest possible forecaster: one who **always** issues the training-years climatology.

## lab5.d.explain
**Climatology always issues the same probability for {cat}: {clim_p}.** In the test seasons {cat} happened {obs_p} of the time, so its single point sits {gap} from the diagonal. It is exactly reliable only when those two numbers agree, which depends on the test seasons, not on any skill. Its BSS against itself is exactly 0 by definition, and its resolution is zero: it never changes, so it can't warn you about anything.

**Reliability is necessary but not enough.** A useful forecast must be reliable *and* say different things in different situations. IDR's skill comes from resolution, and in the pooled setting that resolution is mostly about *place* (tab A).

## lab5.e.intro
The notebook scores **{n_test}** test pairs, which sounds like a lot of evidence. But all those pairs come from only **10 seasons**, and pairs from the same season share the same weather. How much does that matter?

## lab5.e.explain
**Left:** two ways to estimate the uncertainty of the score. *Resampling pairs* pretends all {n_pairs} pairs are independent. *Resampling seasons* keeps each season's pairs together, so it respects the fact that there are only {n_years} seasons.

**Right:** the pooled model's score for each test season on its own. It ranges from **{yr_lo}** to **{yr_hi}**; **{n_neg}** of the {n_years} seasons score below zero.

The pooled model's seasons tend to score alike, because its skill is mostly **geography**, which doesn't change from one season to the next: a desert is a desert every year. Switch to the location-aware model to see the year-to-year skill, which is much less steady.

## lab5.check.q
In your own words: why did the BSS for below normal go from **{b1}** to **{b2}** to **{b3}** across the three settings in tab A? Which one would you report for a national forecast?

## lab5.check.a
- **{b1}** answers "can SEAS5 tell a dry place from a wet one?" That is not the forecasting question.
- **{b2}** asks the pooled model a local question ("is this season drier than usual *here*?"). It was never trained for that; it {v2}.
- **{b3}** comes from a model built for the local question; it {v3}. That is the honest measure of year-to-year skill.

For a national or local forecast, report a **local-cutoff** score (like ③), together with its season-to-season spread (tab E), and say clearly how many test seasons it rests on.

## lab6.intro
Every discussion, interpretation and take-home question from the practical, in order. For each one: try your own answer first, then open the plot that answers it, then compare with the answer built from the app's live numbers.

## qb.bias_amount.q
Does bias vary by rainfall amount?

## qb.bias_amount.a
Yes. The average bias (SEAS5 − CHIRPS) runs from about **{bias_lo} mm** for the driest tenth of forecasts to **{bias_hi} mm** for the wettest. SEAS5 is slightly too dry where little rain falls and much too wet where a lot falls.

## qb.mean_bias.q
Would subtracting the average bias capture forecast uncertainty?

## qb.mean_bias.a
No. The overall average bias, **{bias_all} mm**, is wrong for most forecasts, because bias depends on the amount. And even a perfect bias correction only moves one number: it says nothing about how far the real season might fall from it. The scatter of individual seasons around the average is large, and it differs between dry and wet forecasts.

## qb.why_distribution.q
Why might a whole predictive distribution be more useful than one corrected rainfall value?

## qb.why_distribution.a
A distribution gives the **chance of every outcome**: the chance of below-normal rain, of more than 300 mm, and so on. Users plan around risks, not single numbers. It also shows how confident the forecast is: a steep CDF means confident, a gentle one means uncertain. And it can be turned directly into the tercile probabilities that seasonal outlooks publish (Lab 3).

## qb.dry_threshold.q
At a fixed rainfall threshold, which forecast has the highest probability of a dry outcome?

## qb.dry_threshold.a
The **driest SEAS5 forecast**. In Lab 2's four-forecast view, the curve for the lowest SEAS5 value is always the highest at any threshold: it has the most probability of ending at or below that amount.

## qb.monotone.q
What shape or monotonicity restrictions do you see in the predictive distributions?

## qb.monotone.a
- **Across forecasts:** the CDFs never cross. A wetter SEAS5 forecast always gives a CDF on or below a drier one, so "wetter forecast → wetter outcome" holds at every threshold. This is the isotonic constraint, enforced by PAVA (Lab 2).
- **Within one forecast:** each CDF is a **step function** rising from 0 to 1. IDR puts probability only on rainfall amounts seen in training; it assumes no bell shape. So it never gives probability to negative rainfall, unlike the linear + Gaussian model.

## qb.wmo_terciles.q
*(From the practical's operational caveat.)* Why can't the supplied WMO mean and standard deviation be used as tercile cutoffs?

## qb.wmo_terciles.a
Tercile cutoffs are the rainfall amounts with one third of past seasons below and one third above. They depend on the **whole shape** of the rainfall distribution. Seasonal rainfall is skewed (many dry seasons, a few very wet ones), and in dry places many seasons are exactly 0 mm. A mean and a standard deviation can't tell you where the thirds fall: two places with the same mean and spread can have very different terciles. Proper local cutoffs need the actual reference-period observations, as computed in Lab 3's "Local" mode (pooled cutoffs here are q⅓ = {q1} mm, q⅔ = {q2} mm).

## qb.auc_vs_bss.q
Compare the three ROC AUC values with their Brier skill scores. Which category best separates event and non-event cases?

## qb.auc_vs_bss.a
AUC: BN **{auc_bn}**, NN **{auc_nn}**, AN **{auc_an}**. BSS: BN **{bss_bn}**, NN **{bss_nn}**, AN **{bss_an}**. **{best_auc}** separates events best and **{worst_auc}** is the hardest. Near normal is usually hardest, because it sits between the other two. The two scores measure different things: AUC only the ranking, BSS calibration too.

## qb.auc_reliability.q
Can high ROC AUC coexist with poor reliability?

## qb.auc_reliability.a
**Yes.** AUC depends only on the *order* of the probabilities. Halve every probability, or push them all towards 0 or 1: the order stays the same, so AUC is unchanged, but the probabilities no longer match observed frequencies and the reliability diagram falls off the diagonal. That's why the practical checks both.

## qb.independent.q
Why should scores pooled across thousands of locations **not** be interpreted as thousands of independent forecast years?

## qb.independent.a
The {n_test} test pairs come from only **{n_years} seasons**. In any one season, large-scale drivers (El Niño, the Indian Ocean Dipole) shape rainfall over huge areas, so neighbouring locations' errors move together. You really have {n_years} draws of "what kind of season was it". Scores vary noticeably from season to season (Lab 4D, Lab 5E), and a couple of unusual seasons could change the picture.

## qb.positive_bss.q
Which tercile has positive BSS and the highest ROC AUC?

## qb.positive_bss.a
In the notebook's pooled setting, positive BSS: **{pos_text}** (BN **{bss_bn}**, NN **{bss_nn}**, AN **{bss_an}**). Highest BSS: **{best_bss}**; highest AUC: **{best_auc}**. But remember tab A of Lab 5: with each location's own cutoffs, the same model {v2} for BN.

## qb.sparse_bins.q
Are some probability bins poorly sampled?

## qb.sparse_bins.a
Yes. Pooled IDR issues many probabilities near 0% and 100% (a desert location is almost certainly BN), so the middle bins are thin. Bins with fewer than 10 forecasts, using the notebook's 8 bins: **{sparse}**. Points from such bins in the reliability diagram are mostly noise. With more bins (try 15) the problem gets worse.

## qb.pooling_regional.q
How could pooling across dissimilar rain regimes produce misleading regional skill?

## qb.pooling_regional.a
With one set of cutoffs for all of Africa (q⅓ = {q1} mm, q⅔ = {q2} mm), a desert location is "below normal" almost every year, and a forecast that says "BN" there is almost always right, without knowing anything about the season (Lab 3, dry place). Those easy wins dominate the score. Judged against each location's own cutoffs, the BSS for BN moves from **{b1}** to **{b2}**: the same model {v2}. A location-aware model {v3}.

## qb.reliable_no_skill.q
*(Take-home.)* What does it mean for a probabilistic forecast to be *reliable* but have little *skill*?

## qb.reliable_no_skill.a
Reliable: when it says 30%, the event happens about 30% of the time. Little skill: it hardly beats climatology, usually because it rarely **changes** its probabilities (low resolution).

The extreme case is climatology itself. Its BSS against itself is exactly 0 by definition, and it has zero resolution. It is exactly reliable only if the event happens as often in the test seasons as it forecasts. Here it forecasts {clim_bn} for BN and BN happened {test_bn} of the time: close, but not exactly equal. Either way, it can never warn anyone about anything.

## qb.clim_no_test.q
*(Take-home.)* Why must climatology reference values be estimated without test-year observations?

## qb.clim_no_test.a
Because the reference would then partly **know the answers**. With local cutoffs, letting the 10 test seasons into each location's reference moves the honest model's BSS for BN from **{leak_honest}** to **{leak_ref}**, because the yardstick becomes unrealistically good. In real forecasting, the future isn't available when you build the climatology. Leakage can also flatter a model: training IDR on the test pairs too moves the same score to **{leak_model}**. Either way the comparison is no longer fair.

## qb.create_predictability.q
*(Take-home.)* Does calibration create predictability where none existed?

## qb.create_predictability.a
**No.** With SEAS5 shuffled, so it carries no information about the season, IDR's BSS for BN collapses from {bss_bn} to about **{shuf}**. The small remainder comes from IDR's monotone assumption ranking the test forecasts; force the wrong direction and it becomes **{wrong}**. Forecasting climatology gives **{const}**. Calibration can only *unlock* skill that is already in the forecast, by fixing biases and turning it into honest probabilities.

## glossary
**SEAS5**: ECMWF's fifth-generation seasonal forecast system. In this practical: one forecast October–December rainfall total per location and season, issued in early September.

**CHIRPS**: a rainfall record that blends satellite estimates with rain-gauge observations. Here it plays the role of "what actually fell".

**OND**: October–November–December, the season studied.

**Location-season (pair)**: one location in one season: a SEAS5 value and the matching CHIRPS value.

**Training years / test years**: the model learns only from training years (2001–2015 by default) and is judged on test years (2016–2025) that it never saw.

**Bias**: forecast minus observation. Positive = the forecast was too wet.

**Anomaly**: how much a value differs from that location's usual (average) value.

**Correlation (r)**: how closely two quantities rise and fall together, from −1 to +1. 0 means no linear relationship.

**Climatology**: the long-term behaviour of the weather at a place. As a forecast: always issue the historical frequencies (about ⅓ for each tercile).

**CDF (cumulative distribution function)**: a curve giving, for every rainfall amount, the chance of ending at or below it.

**Predictive distribution**: the full set of probabilities a forecast gives for every possible outcome, here shown as a CDF.

**IDR (isotonic distributional regression)**: a method that learns a predictive CDF for each SEAS5 value from past pairs, with one rule: a wetter forecast can never make dry outcomes more likely.

**Isotonic / monotone**: never changing direction, e.g. only going down as SEAS5 increases.

**PAVA (pool adjacent violators algorithm)**: the procedure that enforces monotonicity: whenever neighbours are in the wrong order, replace them with their average.

**Tercile**: one of the two cutoffs (q⅓ and q⅔) that split past seasons into three equal-sized groups.

**BN / NN / AN**: below normal (≤ q⅓), near normal (between), above normal (> q⅔).

**Pooled vs local cutoffs**: pooled = one pair of cutoffs for all of Africa (the notebook). Local = each location's own cutoffs, as official forecasts use.

**Brier score (BS)**: the average squared difference between the forecast probability and the outcome (1 or 0). 0 is perfect.

**Brier skill score (BSS)**: improvement over climatology: 1 − BS/BS(climatology). Above 0 = better than climatology.

**Reliability**: whether forecast probabilities match observed frequencies (70% forecasts come true 70% of the time).

**Resolution**: whether forecasts change when outcomes change. Climatology has none.

**Sharpness**: how often a forecast issues confident probabilities (near 0% or 100%).

**ROC curve / AUC**: shows how well forecasts separate events from non-events at every warning cutoff. AUC 0.5 = chance, 1 = perfect.

**Hit rate / false alarm rate**: share of events that were warned / share of non-events that were warned.

**Leakage**: test-period information getting into the model, the cutoffs or the reference, which makes the evaluation unfair.

**Bootstrap**: estimating uncertainty by recomputing a score on many resampled versions of the data.

## lab5.e.explain_local
**Left:** two ways to estimate the uncertainty of the score. *Resampling pairs* pretends all {n_pairs} pairs are independent. *Resampling seasons* keeps each season's pairs together, so it respects the fact that there are only {n_years} seasons.

**Right:** the location-aware model's score for each test season on its own. It ranges from **{yr_lo}** to **{yr_hi}**, and **{n_neg}** of the {n_years} seasons score below zero.

Overall the model {v_now}, but individual seasons disagree. Compare the two intervals on the left: with the pairs spread thinly over 12,424 locations they can be similar. That doesn't make the result safe: with only {n_years} seasons, resampling can only reshuffle the seasons that happened, not show what a very different season (a strong El Niño, for instance) would do. Report the score together with this season-to-season spread.

## qb.create_predictability.a_pending
**No.** Calibration can only *unlock* skill that is already in the forecast, by fixing biases and turning it into honest probabilities. The test: shuffle SEAS5 so it carries no information about the season, refit, and score again. Press the button to run it, and the numbers fill in here.
