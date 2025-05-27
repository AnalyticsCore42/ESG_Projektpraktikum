# Association Rule Mining Analysis


## 1. Program Association Rules


### Rules for support=0.1, confidence=0.6

Found 180 rules


Top 5 rules (by confidence):

- If ['RENEW', 'OP', 'IMPROV', 'AUDITS'] then ['ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'IMPROV', 'OP'] then ['ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'OP'] then ['AUDITS', 'ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'OP', 'ISSUES'] then ['AUDITS']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'OP', 'AUDITS'] then ['ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000


### Rules for support=0.3, confidence=0.8

Found 180 rules


Top 5 rules (by confidence):

- If ['IMPROV', 'OP'] then ['AUDITS']
  Support: 0.996, Confidence: 1.000, Lift: 1.000

- If ['IMPROV'] then ['AUDITS', 'ISSUES']
  Support: 1.000, Confidence: 1.000, Lift: 1.000

- If ['IMPROV', 'ISSUES'] then ['AUDITS']
  Support: 1.000, Confidence: 1.000, Lift: 1.000

- If ['IMPROV', 'AUDITS'] then ['ISSUES']
  Support: 1.000, Confidence: 1.000, Lift: 1.000

- If ['RENEW'] then ['ISSUES']
  Support: 0.997, Confidence: 1.000, Lift: 1.000


### Rules for support=0.2, confidence=0.7

Found 180 rules


Top 5 rules (by confidence):

- If ['RENEW', 'OP', 'IMPROV'] then ['AUDITS', 'ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'OP', 'IMPROV', 'ISSUES'] then ['AUDITS']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'OP', 'IMPROV', 'AUDITS'] then ['ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'IMPROV', 'OP'] then ['ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000

- If ['RENEW', 'OP'] then ['AUDITS', 'ISSUES']
  Support: 0.994, Confidence: 1.000, Lift: 1.000


## 2. Sequential Implementation Patterns


Top transitions by probability:

- ISSUES -> AUDITS: 0.208 probability (58440 occurrences)
- IMPROV -> ISSUES: 0.208 probability (58427 occurrences)
- AUDITS -> OP: 0.208 probability (58255 occurrences)
- RENEW -> IMPROV: 0.207 probability (58227 occurrences)
- OP -> RENEW: 0.167 probability (46987 occurrences)
- OP -> IMPROV: 0.001 probability (188 occurrences)
- AUDITS -> RENEW: 0.000 probability (138 occurrences)
- RENEW -> ISSUES: 0.000 probability (28 occurrences)
- ISSUES -> OP: 0.000 probability (14 occurrences)
- IMPROV -> AUDITS: 0.000 probability (9 occurrences)


Most effective transitions (by emission reduction):

- IMPROV -> AUDITS: -8.51% avg emission trend (9 occurrences)
- OP -> IMPROV: -5.52% avg emission trend (188 occurrences)
- ISSUES -> RENEW: -5.01% avg emission trend (1 occurrences)
- ISSUES -> OP: -4.45% avg emission trend (14 occurrences)
- OP -> ISSUES: -2.75% avg emission trend (2 occurrences)
- OP -> RENEW: -2.48% avg emission trend (46987 occurrences)
- AUDITS -> OP: -2.36% avg emission trend (58255 occurrences)
- IMPROV -> ISSUES: -2.28% avg emission trend (58427 occurrences)
- ISSUES -> AUDITS: -2.28% avg emission trend (58440 occurrences)
- RENEW -> IMPROV: -2.27% avg emission trend (58227 occurrences)

## 3. Program Co-occurrence Analysis


Top 10 most common program pairs:

- AUDITS & ISSUES: 11159 companies
- AUDITS & IMPROV: 11158 companies
- IMPROV & ISSUES: 11158 companies
- AUDITS & RENEW: 11131 companies
- ISSUES & RENEW: 11131 companies
- IMPROV & RENEW: 11130 companies
- AUDITS & OP: 11117 companies
- ISSUES & OP: 11117 companies
- IMPROV & OP: 11116 companies
- OP & RENEW: 11089 companies

## 4. Visualization


A network visualization of program transitions has been saved to: `analysis_results/figures/transition_network.png`

The thickness of each arrow represents the transition probability, and the labels show the percentage of transitions.