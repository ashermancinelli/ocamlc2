#!/usr/bin/perl 

# Type inference implementation
# This is heavily based on the appendix from the Cardelli paper.

package ast;

# constructors for Abstract syntax trees

sub lambda {
    my ($class) = shift;
    my ($self) = {};
    $self->{type} = "lambda";
    $self->{var} = shift;
    $self->{body} = shift;
    bless $self, $class;
    return $self;
};

sub ident {
    my ($class) = shift;
    my ($self) = {};
    bless $self, $class;
    $self->{type} = "ident";
    $self->{name} = shift;
    return $self;
};

sub funapp {
    my ($class) = shift;
    my ($self) = {};
    bless $self, $class;
    $self->{type} = "funapp";
    $self->{fn} = shift;
    $self->{arg} = shift;
    return $self;
};

sub let {
    my ($class) = shift;
    my ($self) = {};
    bless $self, $class;
    $self->{type} = "let";
    $self->{var} = shift;
    $self->{defn} = shift;
    $self->{body} = shift;
    return $self;
}

sub letrec {
    my ($self) = &let;  # call let with the same arguments
    $self->{type} = "letrec";
    return $self;
}


# a recursive print procedure

sub print { 
    my ($ast) = shift;

    # identifiers aren't parenthesized
    if ($ast->{type} eq "ident") {
        print $ast->{name};
        return;
    } 

    print "(";

    if ($ast->{type} eq "lambda") {
        print "fn $ast->{var} => ";
        $ast->{body}->print;
    } elsif ($ast->{type} eq "funapp") {
        $ast->{fn}->print;
        print " ";
        $ast->{arg}->print;
    } elsif ($ast->{type} =~ /^let(rec)?$/) {
        print "$ast->{type} $ast->{var} = ";
        $ast->{defn}->print;
        print " in ";
        $ast->{body}->print;
    } else {
        die;
    }
    print ")";
};

package type;

# constructors for types

sub new_fun {
            my($class) = shift;
            bless { type => "oper",
                    opname => "->",
                    args => [ shift, shift ]
                }, $class;
    } ;

sub new_var {
# $varname is a global variable that
# ensures unique identifier allocation
    $varname++ if defined $varname;
    $varname = "a" unless defined $varname;
    return bless { type => "var", varname => $varname }, shift;
};

sub oper {
    my ($class) = shift;
    bless { type => "oper",
        opname => shift,
        args => shift
    }, $class;
};


# once again, the recursive print procedure

sub print {
    my ($self) = shift;
    my ($paren) = shift;
    if ($self->{type} eq "var") {
        # find the unified definition of this variable
        if ($self->{instance}) {
    #        print "[$self->{varname}=";
            $self->{instance}->print();
    #        print "]";
        } else {
            print $self->{varname};
        }
    } else {
        if ($#{$self->{args}} == 1) {
            # for binary operators, print them as "arg1 op arg2"
            print "(";
            $self->{args}[0]->print();
            print $self->{opname};
            $self->{args}[1]->print();
            print ")";
        } else {
            # for all others, print them as "op arg1 arg2 ..."
            print $self->{opname};
            foreach $arg (@{$self->{args}}) {
                print " ";
                $arg->print();
            }
        }
    }
};


package main;

# try to infer the type of expression
sub tryexp {
    my ($ast) = $_[0];
    $ast->print();
    print " : ";
# we need to eval this block because it might "die"
# if the type is non-inferrable
    eval { (&analyze($ast, $myenv, []))->print(); };
    print "\n";
}

# generate a fresh (renamed) variable
# keeps an environment of variables that
# have already been renamed, so that 
# (t->t) gets renamed to (q->q), not (q->r)
sub freshvar {
    my ($type, $env) = @_;
    $env->{$type} = type->new_var() unless $env->{$type};
    return $env->{$type};
}

# generate a fresh type, copying the generic
# variables
sub fresh {
# maintains an environment of variables that
# have already been copied/renamed, which
# it passes down recursively
    my ($type, $env, $nongen) = @_;
    $type = &prune($type);
    if ($type->{type} eq "var") {
        if (scalar grep { &occursintype($type, $_); } @$nongen) {
            # if non-generic
            return $type;
        } else {
            return &freshvar($type, $env);
        }
    } elsif ($type->{type} eq "oper") {
        # recursively operate on the arguments
        return type->oper($type->{opname}, 
            [ map { &fresh($_, $env, $nongen) }  @{$type->{args}} ]);
    } else {
        die;
    }
}

# retrieves the type of an ident from the environment
# generating a fresh copy of the generic variables
sub gettype {
    my ($name, $env, $nongen) = @_;
    my($type) = $env->{$name};
    # all int literals are of type int
    $type = type->oper("int", [])
        if ($name =~ /^(\d+)$/);
    unless (defined $type) {
        print "undefined symbol: $name";
        die;
    }
#   print "gettype $name = "; $type->print(); print "\n";
    $type = &fresh($type, {}, $nongen);
#   print "gettype fresh $name = "; $type->print(); print "\n";
    return $type;
}

# figure out if variable $type1 occurs anywhere in type $type2
sub occursintype {
    my ($type1, $type2) = @_;

    if ($type2->{type} eq "var") {
        return $type1 == $type2;
    } elsif ($type2->{type} eq "oper") {
        return scalar grep { &occursintype($type1, $_); } 
            @{$type2->{args}};
    } else {
        die;
    }
}

# returns the currently defining instance of $type
# as a side effect, collapses the list of type instances
sub prune { 
    my ($type) = $_[0];

    if ($type->{type} eq "var") {
        if (defined $type->{instance}) {
            $type->{instance} = &prune($type->{instance});
            return $type->{instance};
        }
    }
    return $type;
}

sub unify { 
    my ($type1, $type2, $nongen) = @_;
# reduce to types we currently care about
    $type1= &prune($type1);
    $type2= &prune($type2);

# do NOT unify a non-generic variable with a generic one:
# we have to "taint" the generic variable during unification
# so that we no longer get fresh copies of it
    if ($type1->{type} eq "var" && $type2->{type} eq "var" && 
       (scalar grep { &occursintype($type1, $_) } @$nongen) &&
       !(scalar grep { &occursintype($type2, $_) } @$nongen)) {
        return &unify($type2, $type1, $nongen);
    }

# debugging stuff
#    print "unify ";
#    $type1->print();
#    print " ";
#    $type2->print();
#    print "\n";

    if ($type1->{type} eq "var") {
        do { print "recursive unification"; die; } 
            if &occursintype($type1, $type2) &&
                    $type1 != $type2;
        # define a type instance
        $type1->{instance} = $type2;
    } elsif ($type1->{type} eq "oper") {
        if ($type2->{type} eq "var") {
            return &unify($type2, $type1, $nongen);
        }
        if ($type1->{opname} eq $type2->{opname}) {
            if ($#{$type1->{args}} != $#{$type2->{args}}) {
                print "type mismatch";
                die;
            }
            # unify arguments
            for (my($i) = 0; $i <= $#{$type1->{args}}; $i++) {
                unify($type1->{args}[$i], $type2->{args}[$i], $nongen);
            }
        } else {
            print "Type mismatch: $type1->{opname} ne $type2->{opname}";
            die;
        }
    } else {
        die;
    }
};
            
# recursively infer the type of an $ast expression
sub analyze {
    my($ast, $env, $nongen) = @_;
    my($res);

#    print "analyze ";
#    $ast->print;
#    print "\n";

    # $res and goto done are used to 
    # be able to print out intermediate results when debugging

    if ($ast->{type} eq "ident") {
        $res = &gettype($ast->{name}, $env, $nongen);
        goto done;
    } elsif ($ast->{type} eq "funapp") {
        my ($funtype) = &analyze($ast->{fn}, $env, $nongen);
        my ($argtype) = &analyze($ast->{arg}, $env, $nongen);
        my ($resulttype) = type->new_var();
        # unify (argtype -> alpha) funtype
        &unify(type->new_fun($argtype, $resulttype), $funtype, $nongen);
        $res = $resulttype;
        goto done;
    } elsif ($ast->{type} eq "lambda") {
        # create a new non-generic variable for the binder
        my ($newenv) = { %$env };
        my ($argtype) = type->new_var();
        $newenv->{$ast->{var}} = $argtype;
        my ($newnongen) = [ @$nongen ];
        push @$newnongen, $argtype;
        # analyze the expression with this new variable
        my ($result) = &analyze($ast->{body}, $newenv, $newnongen);
        # create the appopriate return type
        $res = type->new_fun ($argtype, $result);
        goto done;
    } elsif ($ast->{type} eq "let") {
        # analyze the definition
        my ($defntype) = &analyze($ast->{defn}, $env, $nongen);
        # this is then the type of the binder (which is now generic)
        my ($newenv) = { %$env };
        $newenv->{$ast->{var}} = $defntype;
        # analyze the body
        $res = &analyze($ast->{body}, $newenv, $nongen);
        goto done;
    } elsif ($ast->{type} eq "letrec") {
        # generate a new non-generic type for the binder
        my ($newenv) = { %$env };
        my ($newnongen) = [ @$nongen ];
        my ($newtype) = type->new_var();
        $newenv->{$ast->{var}} = $newtype;
        push @$newnongen, $newtype;
        # analyze the type of the definition, with the binder type
        # being non-generic (as if we were using the fixed point combinator)
        my ($defntype) = &analyze($ast->{defn}, $newenv, $newnongen);
        # unify to obtain the proper type of binder
        &unify($newtype, $defntype, $nongen);
        # which now becomes generic while analyzing the body
        $res = &analyze($ast->{body}, $newenv, $nongen);
        goto done;
    } else {
        die "unknown AST type";
    }
done:
# debugging output
#    $ast->print();
#    print " : ";
#    $res->print();
#    print "\n";
    return $res;
} 

# setup the environment
# pair: a -> (b -> (a X b)
$var1 = type->new_var;
$var2 = type->new_var;
$pairtype = type->oper(" X ", [$var1, $var2]);
$myenv->{pair} = type->new_fun($var1, type->new_fun($var2, $pairtype));
$myenv->{"true"} = type->oper("bool", []);

# cond: bool -> (a -> (a -> a)
$var1 = type->new_var;
$myenv->{"cond"} = type->new_fun(type->oper("bool", []),
    type->new_fun($var1, type->new_fun($var1, $var1)));
# zero: int->bool
$myenv->{"zero"} = type->new_fun(type->oper("int", []), type->oper("bool", []));
# pred: int->int
$myenv->{"pred"} = type->new_fun(type->oper("int", []), type->oper("int", []));
# times: int->(int->int)
$myenv->{"times"} = type->new_fun(type->oper("int", []),
    type->new_fun(type->oper("int", []), type->oper("int", [])));
      

# some test expressions

# letrec factorial = fn n => cond (zero n) 1 (times n factorial(pred(n)))
# in factorial(5)
$ast = ast->letrec("factorial", # letrec factorial = 
            ast->lambda("n",    # fn n => 
                ast->funapp(
                    ast->funapp(   # cond (zero n) 1 
                        ast->funapp(ast->ident("cond"),     # cond (zero n)
                            ast->funapp(ast->ident("zero"), ast->ident("n"))),
                        ast->ident("1")),
                    ast->funapp(    # times n
                        ast->funapp(ast->ident("times"), ast->ident("n")),
                        ast->funapp(ast->ident("factorial"),
                            ast->funapp(ast->ident("pred"), ast->ident("n")))
                    )
                )
            ),      # in 
            ast->funapp(ast->ident("factorial"), ast->ident("5"))

        ); 
&tryexp($ast);

# fn x => (pair(x(3)) (x(true))
$ast = ast->lambda("x", 
        ast->funapp(
            ast->funapp(ast->ident("pair"), 
                ast->funapp(ast->ident("x"),ast->ident("3"))),
            ast->funapp(ast->ident("x"), ast->ident("true"))));

&tryexp($ast);

# pair(f(3), f(true))

$pair =         ast->funapp(
            ast->funapp(ast->ident("pair"), 
                ast->funapp(ast->ident("f"),ast->ident("4"))),
            ast->funapp(ast->ident("f"), ast->ident("true")));

&tryexp($pair);

# let f = fn x => x in 
#    pair (f(3), f(true))

$ast = ast->letrec("f",    # let f 
          ast->lambda("x", ast->ident("x")),    # = fn x=>x
          $pair); # pair(f(3), f(true))

&tryexp($ast);

# fn f => f f
$ast = ast->lambda("f", ast->funapp(ast->ident("f"), ast->ident("f")));
# this will fail
&tryexp($ast);

# let g = fn f => 5 in g g
$ast = ast->let("g", 
        ast->lambda("f", ast->ident("5")),
        ast->funapp(ast->ident("g"), ast->ident("g")));
&tryexp($ast);

# example from the slides that demonstrates
# generic and non-generic variables:
# fn g => let f = fn x => g in pair (f 3, f true)
$ast = ast->lambda("g",
        ast->let("f",
            ast->lambda("x", ast->ident("g")),
            ast->funapp(
                ast->funapp(ast->ident("pair"),
                    ast->funapp(ast->ident("f"), ast->ident("3")),
                ),
                ast->funapp(ast->ident("f"), ast->ident("true"))
            )
        ));
&tryexp($ast);
exit 0;
