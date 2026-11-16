"""
Reproducible analysis of rag_ablation_judged_v2.csv (read-only; the data are never modified).
Usage: python3 rag_ablation_analysis.py /path/to/rag_ablation_judged_v2.csv
All numbers in the write-up come from this script. Random seed fixed for resampling procedures.
"""
import sys, json, re, itertools
import numpy as np, pandas as pd
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/rag_ablation_judged_v2.csv'
RNG = np.random.default_rng(20261110)
NBOOT, NPERM = 20000, 200000
df = pd.read_csv(PATH, encoding='utf-8-sig')

def hdr(t): print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100)
def desc(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1), median=np.median(x),
                q1=np.percentile(x, 25), q3=np.percentile(x, 75), min=x.min(), max=x.max())
def fmt(d): return ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in d.items())
def holm(pvals):
    p = np.array(pvals); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h
def cp_upper(k, n, alpha=0.05):  # one-sided 95% Clopper-Pearson upper bound
    return 1.0 if k == n else stats.beta.ppf(1 - alpha, k + 1, n - k)
def boot_ci(f, *arrs, n=NBOOT):
    N = len(arrs[0]); out = np.empty(n)
    for b in range(n):
        idx = RNG.integers(0, N, N); out[b] = f(*[a[idx] for a in arrs])
    return np.percentile(out, [2.5, 97.5])
def cliffs(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))

# ------------------------------------------------------------------ A. VALIDATION
hdr('A. DATASET VALIDATION')
print('rows, cols:', df.shape); print('columns:', list(df.columns))
print('model values:', df.model.value_counts().to_dict())
print('condition counts:', df.condition.value_counts().to_dict())
print('language counts:', df.language.value_counts().to_dict())
print(df.groupby(['condition', 'language']).size().to_string())
print('missing values per column:\n', df.isna().sum().to_string())
on = df[df.condition == 'rag_on'].set_index('id').sort_index()
off = df[df.condition == 'rag_off'].set_index('id').sort_index()
print('unique ids on/off:', on.index.nunique(), off.index.nunique(), '| duplicate ids within condition:', on.index.duplicated().sum(), off.index.duplicated().sum())
print('ids in only one condition:', sorted(set(on.index) ^ set(off.index)))
print('pair language identical:', (on.language == off.loc[on.index].language).all(), '| pair message identical:', (on.message == off.loc[on.index].message).all())
print('id range:', df.id.min(), df.id.max(), '| duplicate message texts:', on.message.duplicated().sum())
print('rag_on empty/missing context:', (on.context.fillna('').str.strip() == '').sum(), '| rag_off non-empty context:', off.context.notna().sum())
print('rag_on context length chars:', fmt(desc(on.context.str.len())))
print('response_relevance values:', sorted(df.response_relevance.unique()), '| retrieval_relevance values (rag_on):', sorted(on.retrieval_relevance.dropna().unique()))
print('retrieval_relevance non-null by condition:', df.groupby('condition').retrieval_relevance.apply(lambda s: s.notna().sum()).to_dict())
print('faithfulness labels by condition:\n', pd.crosstab(df.condition, df.faithfulness).to_string())
print('safety values by condition:\n', pd.crosstab(df.condition, df.safety).to_string(), '| safety_reason non-null:', df.safety_reason.notna().sum())
VALID_FAITH = {'supported', 'partially_supported', 'unsupported', 'no_factual_claims', 'not_applicable'}
print('unexpected faithfulness labels:', sorted(set(df.faithfulness) - VALID_FAITH))
print('RR outside 1-5:', ((df.response_relevance < 1) | (df.response_relevance > 5)).sum(), '| RetRel outside 1-5:', ((on.retrieval_relevance < 1) | (on.retrieval_relevance > 5)).sum(), '| non-integer RetRel:', (on.retrieval_relevance % 1 != 0).sum())
hang = lambda s: (lambda L: sum('\uac00' <= c <= '\ud7a3' for c in L) / max(1, len(L)))([c for c in s if c.isalpha()])
df['hangul_frac'] = df.response.map(hang); on['ctx_hangul'] = on.context.map(hang)
print('hangul fraction of response: EN msgs max=%.3f ; KO msgs min=%.3f' % (df[df.language == 'en'].hangul_frac.max(), df[df.language == 'ko'].hangul_frac.min()))
print('max hangul fraction in any retrieved context:', on.ctx_hangul.max())
print('identical response text in on vs off pairs:', (on.response == off.loc[on.index].response).sum())
print('unique context strings (rag_on):', on.context.nunique(), '| by language:', on.groupby('language').context.nunique().to_dict())
cjk = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff]')
for c, g in df.groupby('condition'): print(f'responses containing CJK ideograph/kana ({c}):', g.response.str.contains(cjk).sum(), g[g.response.str.contains(cjk)].id.tolist())
print('response length chars:', df.groupby('condition').response.apply(lambda s: s.str.len().mean()).round(1).to_dict())
crisis_en = re.compile(r'suicid|kill myself|end my life|self[- ]harm|hurt myself|want to die|no reason to live', re.I)
crisis_ko = re.compile(r'자살|죽고 ?싶|죽을|자해|목숨|사라지고 ?싶')
m = on.message
print('messages matching crude crisis keywords: EN', m[on.language == 'en'].str.contains(crisis_en).sum(), '| KO', m[on.language == 'ko'].str.contains(crisis_ko).sum())

# ------------------------------------------------------------------ B. RESPONSE RELEVANCE
hdr('B. RESPONSE RELEVANCE')
pair = pd.DataFrame({'on': on.response_relevance, 'off': off.loc[on.index].response_relevance, 'lang': on.language})
pair['d'] = pair.on - pair.off
for nm, g in [('ALL', df), ('EN', df[df.language == 'en']), ('KO', df[df.language == 'ko'])]:
    for c in ['rag_on', 'rag_off']:
        print(f'{nm:3s} {c:8s}', fmt(desc(g[g.condition == c].response_relevance)))
print('score frequency by condition:\n', pd.crosstab(df.condition, df.response_relevance).to_string())
print('score frequency by condition x language:\n', pd.crosstab([df.language, df.condition], df.response_relevance).to_string())

def paired_block(d, label, on_s=None, off_s=None):
    d = np.asarray(d, float); n = len(d); out = {}
    nz = d[d != 0]
    out['n_pairs'] = n; out['mean_diff'] = d.mean(); out['sd_diff'] = d.std(ddof=1); out['median_diff'] = float(np.median(d))
    t = stats.ttest_1samp(d, 0); se = d.std(ddof=1) / np.sqrt(n); tcrit = stats.t.ppf(.975, n - 1)
    out['t'] = t.statistic; out['df'] = n - 1; out['p_t'] = t.pvalue; out['ci_t'] = (d.mean() - tcrit * se, d.mean() + tcrit * se)
    out['ci_boot'] = tuple(boot_ci(lambda x: x.mean(), d))
    out['dz'] = d.mean() / d.std(ddof=1)
    # Hedges-style small-sample correction for dz
    out['dz_hedges'] = out['dz'] * (1 - 3 / (4 * (n - 1) - 1))
    out['dz_ci_boot'] = tuple(boot_ci(lambda x: x.mean() / x.std(ddof=1), d))
    w = stats.wilcoxon(d, zero_method='wilcox', correction=True, method='approx'); out['wilcoxon_W'] = w.statistic; out['p_wilcoxon'] = w.pvalue
    out['n_nonzero'] = len(nz); out['n_pos'] = int((d > 0).sum()); out['n_neg'] = int((d < 0).sum()); out['n_zero'] = int((d == 0).sum())
    wp = stats.wilcoxon(d, zero_method='pratt', correction=True, method='approx'); out['p_wilcoxon_pratt'] = wp.pvalue
    # rank-biserial for matched pairs (Kerby): (sum ranks pos - sum ranks neg)/total ranks on non-zero diffs
    r = stats.rankdata(np.abs(nz)); out['rank_biserial'] = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    out['p_sign'] = stats.binomtest(out['n_pos'], len(nz), .5).pvalue if len(nz) else np.nan
    # sign-flip permutation test on mean difference (assumption-free under H0 of exchangeable signs)
    obs = abs(d.mean()); signs = RNG.choice([-1, 1], size=(NPERM, n)); perm = np.abs((signs * d).mean(axis=1))
    out['p_perm'] = (np.sum(perm >= obs - 1e-12) + 1) / (NPERM + 1)
    out['shapiro_W'], out['shapiro_p'] = stats.shapiro(d) if len(np.unique(d)) > 1 else (np.nan, np.nan)
    out['skew'] = stats.skew(d); out['kurt_excess'] = stats.kurtosis(d)
    if on_s is not None:  # McNemar on top-2-box (>=4)
        a = (np.asarray(on_s) >= 4); b = (np.asarray(off_s) >= 4)
        n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); out['top2_on'] = int(a.sum()); out['top2_off'] = int(b.sum()); out['mcnemar_b_c'] = (n10, n01)
        out['p_mcnemar_exact'] = stats.binomtest(n10, n10 + n01, .5).pvalue if (n10 + n01) else np.nan
    return out
def show(o, label):
    print(f'\n--- {label} ---')
    for k, v in o.items():
        if isinstance(v, tuple) and all(isinstance(x, float) for x in v): print(f'  {k}: [{v[0]:.4f}, {v[1]:.4f}]')
        elif isinstance(v, float): print(f'  {k}: {v:.6g}')
        else: print(f'  {k}: {v}')
res = {}
res['ALL'] = paired_block(pair.d, 'ALL', pair.on, pair.off); show(res['ALL'], 'PAIRED rag_on - rag_off, ALL (n=144 pairs) [PRIMARY]')
for L in ['en', 'ko']:
    g = pair[pair.lang == L]; res[L] = paired_block(g.d, L, g.on, g.off); show(res[L], f'PAIRED rag_on - rag_off, {L.upper()} (n={len(g)} pairs)')
print('\ndistribution of paired differences (on-off):\n', pd.crosstab(pair.lang, pair.d, margins=True).to_string())
hp = holm([res['en']['p_wilcoxon'], res['ko']['p_wilcoxon']])
print('\nHolm-adjusted Wilcoxon p, family {EN, KO}: EN=%.6g KO=%.6g' % tuple(hp))
hp2 = holm([res['en']['p_perm'], res['ko']['p_perm']]); print('Holm-adjusted permutation p, family {EN, KO}: EN=%.6g KO=%.6g' % tuple(hp2))
# language x condition interaction: compare paired differences between independent language groups
dE = pair[pair.lang == 'en'].d.values; dK = pair[pair.lang == 'ko'].d.values
mw = stats.mannwhitneyu(dE, dK, alternative='two-sided', method='asymptotic'); tw = stats.ttest_ind(dE, dK, equal_var=False)
print(f'\nInteraction (diff-in-diff EN-KO): mean dE={dE.mean():.4f} mean dK={dK.mean():.4f} diff={dE.mean()-dK.mean():.4f}; Welch t={tw.statistic:.3f} p={tw.pvalue:.6g}; MWU U={mw.statistic:.1f} p={mw.pvalue:.6g}; Cliff delta={cliffs(dE,dK):.3f}')
se = np.sqrt(dE.var(ddof=1)/len(dE)+dK.var(ddof=1)/len(dK)); print('  Welch 95%% CI of diff-in-diff: [%.4f, %.4f]' % ((dE.mean()-dK.mean())-1.96*se, (dE.mean()-dK.mean())+1.96*se))
# verbosity control
lp = pd.DataFrame({'on': on.response.str.len(), 'off': off.loc[on.index].response.str.len()}); ld = lp.on - lp.off
print('\nResponse length (chars): on mean=%.1f off mean=%.1f; mean diff=%.2f; Wilcoxon p=%.4g; Spearman(len_on-len_off, RR diff)=%.3f p=%.3g' % (lp.on.mean(), lp.off.mean(), ld.mean(), stats.wilcoxon(ld).pvalue, *stats.spearmanr(ld, pair.d)))

# ------------------------------------------------------------------ C. RETRIEVAL RELEVANCE
hdr('C. RETRIEVAL RELEVANCE (rag_on only; ONE score per response, i.e. per retrieved set of 3 passages)')
rr = on.retrieval_relevance.astype(int)
print('ALL', fmt(desc(rr)))
for L in ['en', 'ko']: print(L.upper(), fmt(desc(rr[on.language == L])))
freq = pd.DataFrame({'n': rr.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)}); freq['pct'] = 100 * freq.n / len(rr); print(freq.round(2).to_string())
print('by language (counts):\n', pd.crosstab(on.language, rr).reindex(columns=[1, 2, 3, 4, 5], fill_value=0).to_string())
print('by language (row %):\n', (100 * pd.crosstab(on.language, rr, normalize='index').reindex(columns=[1, 2, 3, 4, 5], fill_value=0)).round(1).to_string())
band = pd.cut(rr, [0, 2, 3, 5], labels=['poor(1-2)', 'partial(3)', 'relevant(4-5)'])
for nm, g in [('ALL', band), ('EN', band[on.language == 'en']), ('KO', band[on.language == 'ko'])]:
    vc = g.value_counts().reindex(['relevant(4-5)', 'partial(3)', 'poor(1-2)']); print(nm, [(i, int(v), round(100 * v / len(g), 1), tuple(np.round(100 * np.array(wilson(int(v), len(g))), 1))) for i, v in vc.items()])
e, k = rr[on.language == 'en'].values, rr[on.language == 'ko'].values
mw = stats.mannwhitneyu(e, k, alternative='two-sided', method='asymptotic'); tw = stats.ttest_ind(e, k, equal_var=False)
print(f'EN vs KO retrieval: MWU U={mw.statistic:.1f} p={mw.pvalue:.6g}; Welch t={tw.statistic:.3f} p={tw.pvalue:.6g}; mean diff={e.mean()-k.mean():.3f}; Cliff delta={cliffs(e,k):.3f} CI={tuple(np.round(boot_ci(cliffs,e,k,n=3000),3))}')
se = np.sqrt(e.var(ddof=1)/len(e)+k.var(ddof=1)/len(k)); print('  Welch 95%% CI mean diff: [%.3f, %.3f]' % ((e.mean()-k.mean())-1.96*se, (e.mean()-k.mean())+1.96*se))
rel = (rr >= 4).astype(int); tab = pd.crosstab(on.language, rel).reindex(columns=[0, 1], fill_value=0)
print('relevant(4-5) EN vs KO Fisher exact:', stats.fisher_exact(tab.values)[1], tab.values.tolist())
print('Chi-square 3-band x language:', stats.chi2_contingency(pd.crosstab(on.language, band).values)[:3])

# ------------------------------------------------------------------ D. FAITHFULNESS
hdr('D. FAITHFULNESS (rag_on only)')
f = on.faithfulness.value_counts().reindex(['supported', 'partially_supported', 'unsupported', 'no_factual_claims', 'not_applicable'], fill_value=0)
for lab, v in f.items(): print(f'{lab:22s} n={v:3d}  {100*v/len(on):5.1f}%')
nsup = int((on.faithfulness != 'no_factual_claims').sum()); lo, hi = wilson(nsup, len(on)); print(f'responses with identifiable claims: {nsup}/144 = {100*nsup/144:.1f}%  Wilson95% [{100*lo:.1f}, {100*hi:.1f}]')
for L in ['en', 'ko']:
    g = on[on.language == L]; kk = int((g.faithfulness == 'supported').sum()); lo, hi = wilson(kk, len(g)); print(f'  {L}: supported {kk}/{len(g)} = {100*kk/len(g):.1f}%  Wilson95% [{100*lo:.1f}, {100*hi:.1f}]')
tab = pd.crosstab(on.language, on.faithfulness == 'supported').reindex(columns=[False, True], fill_value=0); print('  EN vs KO (has claims) Fisher exact p =', stats.fisher_exact(tab.values)[1], '| OR =', stats.fisher_exact(tab.values)[0])
claims = []
for i, r in on.iterrows():
    for c in json.loads(r.claims): claims.append(dict(id=i, lang=r.language, retrieval=r.retrieval_relevance, status=c['status'], claim=c['claim'], evidence=c['evidence'], response=r.response, context=r.context))
cl = pd.DataFrame(claims)
print('\nclaim-level: total claims =', len(cl), 'in', cl.id.nunique(), 'responses | status counts:', cl.status.value_counts().to_dict(), '| by lang:', cl.lang.value_counts().to_dict())
print('claims per claim-bearing response:', fmt(desc(cl.groupby('id').size())))
on['n_claims'] = on.claims.map(lambda s: len(json.loads(s)))
print('label vs claims consistency:\n', pd.crosstab(on.faithfulness, on.n_claims > 0).to_string())
print('responses with mixed claim statuses:', int((cl.groupby('id').status.nunique() > 1).sum()))
print('label "supported" but any claim not supported:', int(cl[cl.id.isin(on[on.faithfulness == "supported"].index)].status.ne('supported').sum()))
norm = lambda s: re.sub(r'\s+', ' ', s.replace("\\'", "'")).strip().lower()
cl['ev_verbatim_in_ctx'] = [norm(e_) in norm(c_) for e_, c_ in zip(cl.evidence, cl.context)]
cl['claim_verbatim_in_resp'] = [norm(a) in norm(b) for a, b in zip(cl.claim, cl.response)]
cl['claim_verbatim_in_ctx'] = [norm(a) in norm(b) for a, b in zip(cl.claim, cl.context)]
print('evidence string appears verbatim in retrieved context: %d/%d' % (cl.ev_verbatim_in_ctx.sum(), len(cl)))
print('claim string appears verbatim in the response: %d/%d' % (cl.claim_verbatim_in_resp.sum(), len(cl)), ' ids:', cl[cl.claim_verbatim_in_resp].id.tolist())
print('claims written in English although response is Korean:', int(((cl.lang == 'ko')).sum()), '(all %d KO claims)' % (cl.lang == 'ko').sum())
def ngram(a, b, n=4):
    ta, tb = re.findall(r'\w+', norm(a)), re.findall(r'\w+', norm(b)); A = {tuple(ta[i:i+n]) for i in range(len(ta)-n+1)}; B = {tuple(tb[i:i+n]) for i in range(len(tb)-n+1)}
    return len(A & B) / max(1, len(A))
on['ov4'] = [ngram(r, c) for r, c in zip(on.response, on.context)]; on['ov3'] = [ngram(r, c, 3) for r, c in zip(on.response, on.context)]
print('response-vs-context 4-gram overlap: max=%.3f mean=%.3f ; 3-gram max=%.3f mean=%.3f' % (on.ov4.max(), on.ov4.mean(), on.ov3.max(), on.ov3.mean()))
print('\nRetrieval relevance x faithfulness label:\n', pd.crosstab(on.retrieval_relevance, on.faithfulness).to_string())
has = (on.faithfulness == 'supported').astype(int)
print('P(claims | retrieval=4) = %d/%d ; P(claims | retrieval<=3) = %d/%d' % (has[rr == 4].sum(), (rr == 4).sum(), has[rr <= 3].sum(), (rr <= 3).sum()))
print('Fisher exact (claims present vs retrieval=4):', stats.fisher_exact(pd.crosstab(rr == 4, has).reindex(index=[False, True], columns=[0, 1], fill_value=0).values))
print('Mann-Whitney retrieval score, supported vs no_factual_claims:', stats.mannwhitneyu(rr[has == 1], rr[has == 0], method='asymptotic'))
EN4 = (rr == 4) & (on.language == 'en'); KO4 = (rr == 4) & (on.language == 'ko')
print('Within retrieval=4: claims present EN %d/%d, KO %d/%d' % (has[EN4].sum(), EN4.sum(), has[KO4].sum(), KO4.sum()))

# ------------------------------------------------------------------ E. SAFETY
hdr('E. SAFETY')
for c in ['rag_on', 'rag_off']:
    g = df[df.condition == c]; k_ = int((g.safety != 0).sum()); n_ = len(g)
    print(f'{c}: flagged unsafe {k_}/{n_} = {100*k_/n_:.2f}% | exact one-sided 95% upper bound = {100*cp_upper(k_, n_):.2f}% | Wilson95% [{100*wilson(k_, n_)[0]:.2f}, {100*wilson(k_, n_)[1]:.2f}]')
    for L in ['en', 'ko']: gg = g[g.language == L]; print(f'   {L}: {int((gg.safety!=0).sum())}/{len(gg)}')
print('rule-of-three bound for 0/144: %.2f%% ; for 0/288: %.2f%%' % (100 * 3 / 144, 100 * 3 / 288))
print('safety_reason non-null rows:', df.safety_reason.notna().sum(), '| flagged rows:', (df.safety != 0).sum())

# ------------------------------------------------------------------ F. ADDITIONAL
hdr('F. ADDITIONAL ANALYSES (exploratory; correlational)')
on['rr_on'] = on.response_relevance; on['rr_off'] = off.loc[on.index].response_relevance; on['d'] = on.rr_on - on.rr_off
def spearman_ci(x, y):
    r = stats.spearmanr(x, y).statistic; ci = boot_ci(lambda a, b: stats.spearmanr(a, b).statistic, np.asarray(x), np.asarray(y), n=5000); return r, ci
fam = []
for nm, g in [('ALL', on), ('EN', on[on.language == 'en']), ('KO', on[on.language == 'ko'])]:
    for yname in ['rr_on', 'd']:
        r, ci = spearman_ci(g.retrieval_relevance.values, g[yname].values); p = stats.spearmanr(g.retrieval_relevance, g[yname]).pvalue; kt = stats.kendalltau(g.retrieval_relevance, g[yname])
        print(f'Spearman(retrieval, {yname:5s}) {nm}: n={len(g)} rho={r:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] p={p:.4g} | Kendall tau-b={kt.statistic:.3f} p={kt.pvalue:.4g}'); fam.append((nm, yname, p))
print('Holm over the 6 association tests:', dict(zip([f'{a}/{b}' for a, b, _ in fam], np.round(holm([p for *_, p in fam]), 4))))
print('\nMean response relevance (rag_on) and paired diff by retrieval score:')
t = on.groupby('retrieval_relevance').agg(n=('rr_on', 'size'), rr_on_mean=('rr_on', 'mean'), rr_off_mean=('rr_off', 'mean'), d_mean=('d', 'mean'), d_sd=('d', 'std')); print(t.round(3).to_string())
on['ret_band'] = pd.cut(on.retrieval_relevance, [0, 2, 3, 5], labels=['1-2', '3', '4-5'])
tb = on.groupby('ret_band', observed=True).agg(n=('d', 'size'), rr_on=('rr_on', 'mean'), rr_off=('rr_off', 'mean'), d=('d', 'mean'), sd=('d', 'std')); tb['se'] = tb.sd / np.sqrt(tb.n); tb['ci_lo'] = tb.d - 1.96 * tb.se; tb['ci_hi'] = tb.d + 1.96 * tb.se
print('\nPaired diff by retrieval band (normal-approx 95% CI):\n', tb.round(3).to_string())
print('Kruskal-Wallis of paired diff across bands:', stats.kruskal(*[g.d.values for _, g in on.groupby('ret_band', observed=True)]))
print('Kruskal-Wallis of rag_on RR across bands:', stats.kruskal(*[g.rr_on.values for _, g in on.groupby('ret_band', observed=True)]))
print('\nBy language x retrieval band:\n', on.groupby(['language', 'ret_band'], observed=True).agg(n=('d', 'size'), rr_on=('rr_on', 'mean'), rr_off=('rr_off', 'mean'), d=('d', 'mean')).round(3).to_string())
print('Retrieval score vs language-confounding: P(retr=4) EN=%.3f KO=%.3f' % ((on[on.language == "en"].retrieval_relevance == 4).mean(), (on[on.language == "ko"].retrieval_relevance == 4).mean()))
# Language comparisons, Holm family of 5
fam2 = []
for nm, a, b in [('RR rag_on', on[on.language == 'en'].rr_on, on[on.language == 'ko'].rr_on), ('RR rag_off', on[on.language == 'en'].rr_off, on[on.language == 'ko'].rr_off),
                 ('RR diff (on-off)', on[on.language == 'en'].d, on[on.language == 'ko'].d), ('Retrieval relevance', on[on.language == 'en'].retrieval_relevance, on[on.language == 'ko'].retrieval_relevance)]:
    r_ = stats.mannwhitneyu(a, b, method='asymptotic'); fam2.append((nm, a.mean(), b.mean(), r_.statistic, r_.pvalue, cliffs(a.values, b.values)))
ft = pd.crosstab(on.language, has).reindex(columns=[0, 1], fill_value=0).values; fam2.append(('Has claims (Fisher)', has[on.language == 'en'].mean(), has[on.language == 'ko'].mean(), np.nan, stats.fisher_exact(ft)[1], np.nan))
adj = holm([x[4] for x in fam2])
print('\nEN vs KO comparisons (independent groups, n=72 vs 72), Holm family m=5:')
for x, a_ in zip(fam2, adj): print(f'  {x[0]:22s} EN={x[1]:.3f} KO={x[2]:.3f} U={x[3]} p={x[4]:.4g} Holm-p={a_:.4g} Cliff={x[5]}')
print('\nKorean vs English message: mean rag_off RR / rag_on RR above. Context language: all retrieved contexts contain no Hangul ->', on.ctx_hangul.max() == 0)
# Retrieval 'hubness' / templating
print('\nContext contains "time-boxed" (the time-boxing passage): EN %d/72  KO %d/72' % tuple(on.groupby('language').context.apply(lambda s: s.str.contains('time-boxed').sum())[['en', 'ko']]))
for c, g in df.groupby('condition'):
    print(f'Responses containing "25분" ({c}): KO', int(g[g.language == 'ko'].response.str.contains('25분').sum()), '/72')
print('Responses containing "25분" and rr by condition (KO):'); print(df[(df.language == 'ko')].assign(p25=lambda x: x.response.str.contains('25분')).groupby(['condition', 'p25']).response_relevance.agg(['size', 'mean']).round(3).to_string())
print('Top repeated Korean rag_on context strings:', on[on.language == 'ko'].context.value_counts().head(3).tolist()[:3])
print('Max repeats of one rag_on context string:', on.context.value_counts().max())

# ------------------------------------------------------------------ figure data
fd = {}
fd['fig1_rr_by_condition_language'] = df.groupby(['language', 'condition']).response_relevance.agg(['count', 'mean', 'std']).reset_index()
fd['fig2_paired_diff_distribution'] = pd.crosstab(pair.lang, pair.d).reset_index()
fd['fig3_retrieval_distribution'] = pd.crosstab(on.language, rr).reindex(columns=[1, 2, 3, 4, 5], fill_value=0).reset_index()
fd['fig4_diff_by_retrieval'] = on.groupby(['language', 'retrieval_relevance']).agg(n=('d', 'size'), mean_diff=('d', 'mean'), mean_rr_on=('rr_on', 'mean')).reset_index()
fd['fig5_faith_by_retrieval'] = pd.crosstab(on.retrieval_relevance, on.faithfulness).reset_index()
import os; os.makedirs('/home/claude/figdata', exist_ok=True)
for k_, v in fd.items(): v.to_csv(f'/home/claude/figdata/{k_}.csv', index=False)
print('\nfigure-data tables written to /home/claude/figdata')
