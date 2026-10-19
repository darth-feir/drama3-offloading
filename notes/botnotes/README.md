# botnotes — the long version of short comments

Code in `bin/` carries **short** comments: enough to stop you "simplifying"
a line into a bug, and no more. When the full story is longer than that, the
comment ends in a pointer and the story lives here:

```python
# Pad with the most-negative FINITE value, not -inf: the graph's own
# finiteness guard scans the full-width tensor. see notes/botnotes/lm-head-inf-sentinel.md
pad = torch.finfo(head.weight.dtype).min
```

The rule being applied: **a comment earns its place inline if someone about
to edit that line needs it right then.** Everything else — how the bug was
found, what was tried first, the measurements behind a constant — is
reference material and belongs in a file.

## Why not just leave it all inline

266 lines of this repo's comments were in blocks of six lines or more,
several of them half a page. That is a real cost: the code stops being
skimmable, and a wall of prose gets skipped exactly as reliably as no prose
at all.

## Why not just delete it

Because most of it is a record of something that cost hours and looked like
something else at the time. Three separate bugs here produced **plausible
output rather than an error** — a duplicated function definition that made
every "fix" patch a copy that never ran, a cache that was allocated and
never consulted, and a sentinel value that tripped a guard. Deleting the
account of those leaves the next person to rediscover them.

## The rot problem, and the lint

Pointers go stale in a way inline comments do not. So `bin/check_botnotes.py`
runs over the tree and fails if:

- a `notes/botnotes/<slug>.md` reference does not resolve, or
- a botnote exists that nothing references.

Run it before committing. It is cheap and it is the only thing standing
between this scheme and a pile of dead links.

## Writing one

Keep the house rule: every number marked measured, inferred, or someone
else's, and corrections left **in place** rather than deleted. Several
confident conclusions in `notes/` turned out to be wrong and the record of
how is usually the useful part.

```markdown
# <what the thing is>

**Where:** `bin/file.py`, `<function>`

<what the code does and why that is not the obvious choice>

## What breaks if you change it back

<the concrete failure, with the symptom someone would actually see>
```

That last section is the point of the whole directory.
