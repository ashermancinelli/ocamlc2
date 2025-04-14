# Modules

Create another AST structure or to hold each module. host module contains mapping of module names to modules.

`Open M` brings whole module into host module.
Declaring a module creates a new module, starts up the typechecker for the submodule and returns it, host module adds it to module map.
