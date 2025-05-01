the current module system as of commit `commit 5f111f2567917d23de13f41bb31ccd3870a748ae (HEAD -> propper-modules, main)` is bad.
for example we declare variable `a` in module `M` as the string `M.a` in the default environment.
This is not correct.

We must keep an environment for each module, and when we look up variables in the type or environment scope,
we always use a module "path" meaning "M.N.a" is the path `{"M", "N", "a"}`, and we ought to search for module `M`
in the root environment, and if found, search for module `N` in `M`'s environment, and so forth.

Ignore the work needed to handle functors and module types for now.

Design a refactor to make the module handling more robust.
Design small steps and tests we can run along the way.
