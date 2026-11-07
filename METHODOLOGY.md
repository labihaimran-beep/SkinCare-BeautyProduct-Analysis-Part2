# Methodology notes

**Simulator ground truth.** Each platform has a base CPM, CTR, conversion rate, order value and a saturation exponent GAMMA.
A day's platform intensity m scales impressions (hence spend) by m and CTR and conversion rate by m^(-GAMMA/2) each, so
daily revenue ~ spend^(1-GAMMA). The true elasticity is therefore 1 - GAMMA (Facebook 0.70, Instagram 0.75, Google 0.55,
TikTok 0.85, YouTube 0.80). Audience effects (age, gender, city, device), campaign effects and month effects multiply CTR/CVR.

**Why a trust region.** Response curves are estimated from daily spend levels that actually occurred; outside that range
they are extrapolation. The default 0.5x-2x of the current share roughly matches the range of observed daily spend.

**Why greedy is optimal here.** For concave revenue curves (b < 1) giving each next small increment to the platform with the
highest marginal revenue converges to the constrained optimum (equal marginal revenue per rupee, KKT conditions); a unit test checks this.

**Rescaling.** The simulated history covers billions of rupees; the question is about Rs 1M. History is rescaled by one common factor
so that average monthly spend equals the budget baseline. ROAS and elasticities do not change under a common scaling.

**What the checks mean.** The bootstrap resamples days and refits; it measures sampling noise in the curves, not simulator
or model misspecification. The ground-truth check is only possible because the data is simulated.
