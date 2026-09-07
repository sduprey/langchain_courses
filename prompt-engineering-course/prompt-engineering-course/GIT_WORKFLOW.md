# Git workflow for this course

Every lab and the capstone are handed in through Git. This is the exact flow
the slides refer to.

## Once, at the start
```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```
Fork this repo on GitHub, then clone YOUR fork:
```bash
git clone https://github.com/<you>/prompt-engineering-course.git
cd prompt-engineering-course
cp .env.example .env          # then edit .env — it is gitignored, never commit it
```

## Every lab
```bash
git switch -c s3-eval-harness  # a branch per lab/exercise
# ... do the work ...
git add .
git commit -m "Add eval harness and score baseline prompt (62% -> 74%)"
git push -u origin s3-eval-harness
```
Then open a Pull Request on GitHub. The PR description is where you report your
numbers ("accuracy 62% -> 74% after adding two few-shot examples").

## Resolving a merge conflict
```bash
git merge main
# Git marks conflicts in the file between <<<<<<< and >>>>>>>.
# Edit to the correct result, remove the markers, then:
git add <file>
git commit
```

## Capstone (teams)
- One shared repo; each member works on their own branch.
- Integrate through reviewed PRs — never by emailing files.
- Tag your submission:  `git tag v1.0 && git push --tags`

## Golden rules
- `.env` and secrets never get committed (they're in `.gitignore`).
- Small, present-tense commits that say *what changed and why*.
- `main` always runs.
