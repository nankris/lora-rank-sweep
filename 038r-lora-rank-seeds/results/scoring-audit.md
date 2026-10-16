# Why 038's 28.2% and 038c's 31.9% are not matched baselines

The full source diff is in scoring.diff; frozen sources are in reference/.

| Aspect | 038 sweep.py | 038c train.py |
|---|---|---|
| Prompt | `Classify the news article into one of: World, Sports, Business, Sci/Tech.` then `Article: …` and `Answer:` | `Classify the topic of this news article as World, Sports, Business, or Sci/Tech.` then `Article: …` and `Topic:` |
| Article truncation | strip whitespace, first 600 characters, then truncate total prompt to MAXLEN-8=184 tokens (can remove Answer suffix) | tokenize full article, truncate article to 192 minus head/tail/longest label; always retain Topic suffix |
| Candidate score | sum log probabilities of space-prefixed label **and EOS** | sum log probabilities of space-prefixed label only |
| Padding | right padding, implicit position IDs | left padding, explicit cumulative attention-mask position IDs |
| Logits | all positions | last MAXLAB+1 positions; correct causal shift for candidate labels |
| Loss summary | total gold NLL / total gold token count | average per-example gold per-token NLL |
| Data | shuffle seed 1234, 2,000 test items | same requested dataset, shuffle seed, count; historical dataset revision not pinned |

Both maximize unnormalized sums, not length-normalized candidate scores. But 038 adds a label-dependent log P(EOS | prompt,label), which can change the winning label even with identical prompts. Input wording and truncation also change the conditional probabilities. Padding with appropriate attention masks is intended to preserve real-token positions, so the mere padding-side change is not evidence of a bug; 038c explicitly supplies positions.

Thus the 3.7 percentage point gap (74/2,000 items in aggregate, not necessarily 74 discordant predictions) is confounded by different prompts, text budgets and candidate definitions. It is not a measured seed effect or a matched replication discrepancy. A code diff establishes these protocol differences, but cannot apportion the numerical gap among them. No historical item predictions are available here for a paired attribution test. In addition, training differs (038 dropout .05 and Python shuffle vs 038c dropout 0 and torch.randperm, schedule warmups differ), so 038's training recipe must not be silently substituted for this rerun.
