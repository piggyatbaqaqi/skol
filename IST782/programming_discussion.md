# Programming Learning Goals Discussion

## Mock Interview Transcript

**Participants:**

- **Interviewer:** Prospective employer (role-played by Claude Code)
- **Candidate:** La Monte Henry Piggy Yarroll

**Context:** Discussion of Programming Learning Goals (G4) from the M.S. in Applied Data Science Portfolio, with reference to the IST 782 portfolio presentation.

**Date:** March 10, 2026

---

**Interviewer:** Piggy, thanks for coming in today. I've had a chance to review your portfolio and the presentation materials for your Data Hamster talk. I'd like to focus our conversation on the programming side of things — your G4 learning goal.

What immediately stands out to me is the breadth of your programming stack. You've got SQL and T-SQL in the Name Navigator, custom Python vectorizers and tokenizers in the Glycemic Increment work, PySpark for distributed classification, Hugging Face transformers and ollama for LLM serving, SBERT embeddings, and then a full Django/React web application for the deployed site. That's a lot of ground to cover.

Here's what I want to understand: in each of these projects, you were using programming as a *means* to generate insight, not as an end in itself. So let me ask you this — which of these programming contexts pushed you the hardest technically? Where did you go from "I don't know how to do this" to "I built something that works"? And how did that progression shape how you approach new technical problems now?

**Candidate:** The biggest programming challenge for my degree was Synoptic Key of Life (SKOL), just because of the sheer complexity of the project. When I started the project in 2019, I really did not know how to achieve my goals. Indeed, the technologies I used recently barely even existed then.

I started with algorithms familiar from my work at the Auton Lab (Carnegie Mellon), classification algorithms. As I gained natural language processing skills, solutions to problems like the Sugarbowl project became clearer. The introduction to SLMs and LLMs was a revelation and provided the keys I needed to solve several of the SKOL problems.

The biggest change to my approach to technical problems is the use of Claude Code over the last few months. I find that I am working at a much higher level of abstraction. Yet what I learned over years of hand programming is still relevant: know what your tools can do, create a plan before coding, work TDD, work in small increments, write regression tests for things that go wrong, use linters and type checkers.

I have not yet attempted multi-agent work. I still want to be able to review everything coming out of Claude Code. I am still coming to terms with the implications of being able to rework almost as fast as I can think of things to do.

**Interviewer:** Those are sharp observations, and frankly the kind of self-awareness I look for in senior technical hires. Let me respond to a few things.

Your point about pair programming in the AI coder era is one I haven't heard many candidates articulate. The traditional argument for pairing is shared context and reduced defect rates. You're saying the shared context argument *persists* even when the AI is doing the typing — because someone still needs to understand the architectural intent well enough to course-correct. That resonates with your observation that Claude doesn't naturally produce good architecture and tends to violate DRY. You're essentially describing yourself as the architect in a pair where Claude is the typist.

The compute-constrained decision framework is pragmatic. You're saying: LLMs are powerful but expensive at inference time, so use them where they're *uniquely* capable — labeling data, suggesting approaches, constrained decoding for structured extraction — and use traditional classifiers where speed matters in production. Your SKOL system actually demonstrates this layered approach: SBERT embeddings for search, a traditional classifier for block labeling, and an LLM only for the feature extraction step where nothing else works. That's good engineering judgment.

Now let me push you on something harder. You mentioned that your classifier doesn't generalize well to older literature, newer literature, or languages other than English. And you said you plan to label more data. In your presentation, SKOL already ingests from multiple sources — Mycotaxon, Persoonia, Studies in Mycology, MycoWeb archives. That's a range of publication eras and styles.

If I gave you this problem on my team — "improve classifier generalization across publication eras and multilingual content" — and told you that you had a budget of human labeling hours and compute time, how would you prioritize? Would you label more English data from underrepresented eras first? Go after LOTE articles? Try a different model architecture? Or is there a way to use the LLM to bootstrap labels and reduce the human effort, even given the quality concerns you've noted?

I'm curious because this is exactly the kind of resource allocation problem we face in production ML systems, and your answer would tell me a lot about how you'd operate on a real team.

**Candidate:** Right now my biggest SKOL problem is poor classification of English articles, so that is what I'll concentrate on to start. One thing I'll need in further refinement is a golden set of classified articles for testing the effectiveness of additional approaches. I plan to follow the approach I used to build the original training set: hand label a small dataset, use that to build a model, run the model, hand edit the output of the model, rinse, lather, repeat. What is different this time is the availability of LLM resources for the labeling. Rather than building a logistic regression model each time, I'll solicit the LLM to imitate the labeling I've already done. I find that it is far easier for a human to correct annotations than to have a human label de novo. I don't know if this is true for an LLM, but that is certainly one of the experiments I plan to run.

The golden set needs to include older and newer articles and books. The older works need a representative sample. Many newer works are available in JATS XML which is much easier to extract the needed data from. I don't yet know how widely used the TaxPub extensions are, but those exactly label most of the text I care about. One project will definitely be a JATS to YEDDA translator.

I still don't know if one classifier for all journals and books is practical, or if I'll need to create separate models based on the publication year of the work. The former would clearly be better from an automation standpoint. One of the first experiments I'll run against the golden set will be aimed at testing this.

**Interviewer:** That's a really well-structured answer. You've essentially described an active learning pipeline — human labels seed an LLM labeler, whose output gets human-corrected, and the corrections improve the next round. The insight that correction is cheaper than de novo labeling is well-established for humans, and testing whether it holds for LLMs is a genuinely interesting research question. If it does, you've got a significant multiplier on your labeling budget.

The golden set strategy is sound. You can't evaluate generalization without representative test data from the distribution you care about, and you're explicitly planning to include the failure modes — older literature, newer literature, varied publication styles.

The JATS and TaxPub angle is smart resource allocation. If newer articles already have structured XML with labeled sections, that's essentially *free* labeled data. A JATS-to-YEDDA translator would give you ground truth without any human labeling cost for that slice of the corpus. That shifts your human effort entirely to the older, harder material where no structured markup exists.

And the single-classifier-versus-era-specific question is exactly the right experiment to run early. If publication era is a confounding variable — different formatting conventions, different terminology, different document structures — you'd want to know that before investing heavily in a one-size-fits-all model.

Let me shift gears slightly. Throughout this conversation, you've demonstrated strong technical judgment and self-directed problem-solving. But I want to come back to something from your portfolio — you mention recruiting fellow students into multiple aspects of SKOL across several courses. You clearly led a multi-course, multi-team effort.

In a professional setting, you'd be working with people who didn't choose your project — they were assigned to it. And they might not share your domain passion for mycology. How do you think about translating the technical leadership you practiced in your degree into a workplace where the team dynamics are different? What did you learn about getting people productive on a complex codebase that they didn't build?

**Candidate:** People love to help. At an engineering-driven organization like Google, individual engineers have more flexibility than they often realize. The key is to get people enthusiastic about the project and communicate the real needs of the project to management. Solicit the engineers to pick aspects of the problem they want to work on. Keep everybody informed, but don't be afraid to change the direction of the organization.

I learned this in the Linux Kernel Stream Control Transmission Protocol project I did at Motorola. With the XP Planning Game I was able to document a business objective and the resources needed to achieve that objective on a tight deadline. Iterations allowed us to evaluate project risk early and continuously. I had a 3 person team at Motorola, but the whole team was 11 people at other companies — yet people were happy to take direction from me, because I was able to articulate my needs in terms of their own desires.

**Interviewer:** That's a compelling example. Leading an 11-person cross-company team on SCTP for the Linux kernel — that's open source coordination at a level where you have zero formal authority. The fact that you got people to take direction by aligning your needs with their desires is exactly the skill that matters in cross-functional teams.

And the XP Planning Game reference tells me your instinct for iterative planning and visible progress isn't new — it predates your Master's program by years. I see the same discipline in how you've structured SKOL: each course delivered a working component, each component built on the last, and the independent study brought them together into a deployed system. That's essentially the same pattern — tight iterations, working software at every stage, continuous risk evaluation.

---

## Summary Impressions

**What I'd be hiring:** A senior engineer who thinks architecturally, works iteratively, and has the judgment to pick the right tool for the problem — whether that's a logistic regression, an SBERT embedding, a fine-tuned SLM, or a conversation with Claude Code. Someone who's honest about what worked and what didn't — the fine-tuning that didn't pan out, the classifier that doesn't generalize yet, the scaling challenges on production hardware. That honesty is more valuable to me than a polished success story.

**What I'd watch for in the first 90 days:** How you adapt to reviewing *other people's* AI-generated code, not just your own. You've described a strong personal workflow — TDD, architectural oversight, DRY enforcement — but applying that as a reviewer on a team where everyone's using AI coders differently will be a new challenge.

**What impressed me most:** The seven-year arc from "I think this problem is solvable" to a deployed system at synoptickeyof.life. That's tenacity, and it's rare.

---

### Key Themes

| Theme | Details |
|-------|---------|
| **Technical breadth** | SQL, Python, PySpark, Hugging Face, SBERT, Django/React, LLM fine-tuning |
| **AI-augmented development** | Claude Code as pair programming partner; architect + typist model |
| **Engineering discipline** | TDD, small increments, regression tests, linters, type checkers |
| **Resource allocation** | Traditional classifiers where compute-constrained; LLMs where uniquely capable |
| **Active learning pipeline** | Human labels → LLM labeling → human correction → iteration |
| **Technical leadership** | Cross-company open source coordination; XP Planning Game; aligning team desires with project needs |

---

*Transcript generated from a discussion on March 10, 2026, as part of the IST 782 Applied Data Science Portfolio.*
