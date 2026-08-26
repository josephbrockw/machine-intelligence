# machine-intelligence

## Reusuable Functions 

```markdown
src/mi/
    __init__.py
    gpu.py               # cross-cutting: no single domain owns device selection
    lm/
        __init__.py
        model_loader.py
        sampling.py      # <- top-p goes here
        generate.py
        extraction.py
        eval.py
        data.py          # math500 loader
    bayes/               # when you get there
        __init__.py
```

**Three rules that keep this working as it grows**
1. Name domains after the subject, not the book. lm/, not reasoning_model/. Your resource directories are numbered by when you started them, and a second LM book should add to lm/ rather than spawn lm2/. Domains outlive resources; the numbering is a reading log, not an architecture.
2. Not everything graduates into mi/. Your CLAUDE.md already says single-resource helpers sit next to their notebooks — that rule is what stops mi/ becoming a junk drawer. A function moves into mi/<domain>/ when a second consumer appears, or when you specifically want it as durable reference. Until then it stays in the resource directory where you wrote it.
3. Create a domain folder when you have code for it, not before. An empty bayes/ is a promise you might not keep. 01-ThinkBayes2 currently uses the book's own utils.py next to its notebooks, which is exactly right — nothing needs to move.