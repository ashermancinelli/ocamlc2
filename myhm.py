"""
http://lucacardelli.name/Papers/BasicTypechecking.pdf
https://web.archive.org/web/20050420002559/http://www.cs.berkeley.edu/~nikitab/courses/cs263/hm.pl
https://raw.githubusercontent.com/rob-smallshire/hindley-milner-python/refs/heads/master/inference.py
"""

import functools

DEBUG = False


def log(*a, **kw):
    pass


if DEBUG:

    def log(*a, **kw):
        print(*a, **kw)


def logwrap(func):
    """
    A decorator that logs the function name, arguments, and return value.
    Only active when DEBUG is True.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG:
            return func(*args, **kwargs)

        args_strs = [str(a) for a in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        signature = ", ".join(args_strs + kwargs_repr)

        print(f"-- {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} -> {result}")

        return result

    return wrapper


class AST:
    pass

class Apply(AST): ...

class Identifier(AST):
    def __init__(self, name):
        self.name = name

    def __call__(self, *args: AST):
        app = Apply(self, args[0])
        for arg in args[1:]:
            app = Apply(app, arg)
        return app

    def __str__(self):
        return self.name


class IntLit(AST):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class BoolLit(AST):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class BinOp(AST):
    def __init__(self, op, lhs, rhs):
        self.op, self.lhs, self.rhs = op, lhs, rhs

    def __str__(self):
        return f"{self.lhs} {self.op} {self.rhs}"


class Let(AST):
    def __init__(self, name, value, body):
        self.name, self.value, self.body = name, value, body

    def __str__(self):
        return f"let {self.name} = {self.value} in {self.body}"


class Lambda(AST):
    def __init__(self, param, body):
        self.param, self.body = param, body

    def __str__(self):
        return f"(λ{self.param} . {self.body})"

class Apply(AST):
    def __init__(self, fn, arg):
        self.fn, self.arg = fn, arg

    def __str__(self):
        return f"({self.fn} {self.arg})"


class TypeOperator: ...


class TypeVariable: ...


TypeExpr = TypeOperator | TypeVariable


class TypeVariable:
    _id = 0

    def __init__(self):
        self._id = TypeVariable._id
        TypeVariable._id += 1
        self.instance = None
        self._name = None

    def id(self) -> int:
        return self._id

    def instantiated(self) -> bool:
        return self.instance is not None

    @logwrap
    def assign(self, type_expr: TypeExpr) -> None:
        self.instance = type_expr

    def name(self):
        if self._name is None:
            self._name = f"T{self.id()}"
        return self._name

    def __str__(self):
        if self.instance is not None:
            return str(self.instance)
        return self.name()

    def __repr__(self):
        return str(self)


class TypeOperator:
    def __init__(self, name, *types: list[TypeExpr]):
        self.name = name
        self.types = types

    def __str__(self):
        match self.types:
            case 0:
                return self.name()
            case [l, r]:
                return f"({l} {self.name} {r})"
            case _:
                return f"{self.name}{' '.join(self.types)}"

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        return self.types[index]


class Function(TypeOperator):
    def __init__(self, from_type, to_type):
        super().__init__("->", from_type, to_type)


IntType = TypeOperator("int")
BoolType = TypeOperator("bool")


class TypeCheck:
    def __init__(self):
        # self.environment: dict[str, TypeExpr] = environment
        pass

    def __str__(self):
        return "TypeCheck()"

    @logwrap
    def infer(self, ast, env, concrete_types: list[TypeExpr]) -> TypeVariable:
        match ast:
            case Identifier(name=name):
                if name.isnumeric():
                    return self.infer(IntLit(int(name)), env, concrete_types)
                return self.typeof(name, env, concrete_types)
            case IntLit():
                return IntType
            case BoolLit():
                return BoolType
            case Apply(fn=fn, arg=arg):
                fun_type = self.infer(fn, env, concrete_types)
                arg_type = self.infer(arg, env, concrete_types)
                result_type = TypeVariable()
                self.unify_type_expressions(Function(arg_type, result_type), fun_type)
                return result_type
            case Lambda(param=param, body=body):
                arg_type = TypeVariable()
                result_type = self.infer(body, env.copy() | {param: arg_type}, concrete_types.copy() | {arg_type})
                return Function(arg_type, result_type)
            case Let(name=name, value=value, body=body):
                value_type = self.infer(value, env, concrete_types)
                body_type = self.infer(body, env.copy() | {name: value_type}, concrete_types.copy())
                return body_type
            case _:
                assert False

    @logwrap
    def typeof(self, name: str, env, concrete_types: set[TypeExpr]) -> TypeVariable:
        if val := env.get(name, None):
            return self.deep_copy_with_new_generic_typevars(
                val, concrete_types
            )
        raise RuntimeError(f"Undefined symbol: {name}")

    @logwrap
    def deep_copy_with_new_generic_typevars(
        self, type_variable: TypeExpr, concrete_type_vars: list[TypeExpr]
    ) -> TypeExpr:
        mapping: dict[TypeExpr] = {}

        def recursive_copy(type_expr: TypeExpr) -> TypeExpr:
            type_expr = self.prune(type_expr)
            match type_expr:
                case TypeVariable():
                    if self.is_generic(type_expr, concrete_type_vars):
                        if type_expr not in mapping:
                            mapping[type_expr] = TypeVariable()
                        return mapping[type_expr]
                    else:
                        return type_expr
                case TypeOperator():
                    copied_subtypes = [recursive_copy(t) for t in type_expr.types]
                    return TypeOperator(type_expr.name, *copied_subtypes)

        return recursive_copy(type_variable)

    @logwrap
    def prune_type_expression(self, type_expression: TypeExpr) -> TypeExpr:
        type_expression.instance = self.prune(type_expression.instance)
        return type_expression.instance

    @logwrap
    def prune(self, type_expression: TypeExpr) -> TypeExpr:
        """
                The function Prune is used whenever a type expression has to be inspected: it will always
        return a type expression which is either an uninstantiated type variable or a type operator; i.e. it
        will skip instantiated variables, and will actually prune them from expressions to remove long
        chains of instantiated variables.
        """
        match type_expression:
            case TypeVariable():
                if type_expression.instantiated():
                    return self.prune_type_expression(type_expression)
                return type_expression
            case TypeOperator():
                return type_expression

    @logwrap
    def is_concrete(self, expr: TypeExpr, concrete_type_exprs: list[TypeExpr]) -> bool:
        return self.is_sub_type_expression_of_any(expr, concrete_type_exprs)

    @logwrap
    def is_generic(self, expr: TypeExpr, concrete_type_exprs: list[TypeExpr]) -> bool:
        return not self.is_sub_type_expression_of_any(expr, concrete_type_exprs)

    @logwrap
    def is_sub_type_expression_of_any(
        self, maybe_subexpr: TypeExpr, expr_iterable: list[TypeExpr]
    ) -> bool:
        return any(
            self.is_sub_type_expression_of(maybe_subexpr, member_type)
            for member_type in expr_iterable
        )

    @logwrap
    def is_sub_type_expression_of(
        self, maybe_subexpr: TypeExpr, expr: TypeExpr
    ) -> bool:
        expr = self.prune(expr)
        match expr:
            case TypeVariable():
                return expr == maybe_subexpr
            case TypeOperator():
                return self.is_sub_type_expression_of_any(maybe_subexpr, expr.types)

    @logwrap
    def unify_type_expressions(self, expr1: TypeExpr, expr2: TypeExpr) -> None:
        expr1, expr2 = self.prune(expr1), self.prune(expr2)
        match expr1:
            case TypeVariable():
                if expr1 != expr2:
                    if self.is_sub_type_expression_of(expr1, expr2):
                        raise RuntimeError(
                            f"Recursive unification when trying to unify these types: {expr1} {expr2}"
                        )
                    expr1.assign(expr2)

            case TypeOperator(name=n1, types=ts1):
                match expr2:
                    case TypeVariable():
                        self.unify_type_expressions(expr2, expr1)
                    case TypeOperator(name=n2, types=ts2):
                        log(n1, ts1)
                        log(n2, ts2)
                        if n1 != n2 or len(ts1) != len(ts2):
                            raise RuntimeError(f"Type mismatch: {expr1} {expr2}")
                        for subtype1, subtype2 in zip(ts1, ts2):
                            self.unify_type_expressions(subtype1, subtype2)
            case _:
                print(expr1, expr2)
                assert False


def typevars(how_many: int):
    return [TypeVariable() for _ in range(how_many)]


def test(base_env, ast: AST | list[AST]):
    match ast:
        case list():
            for a in ast:
                test(base_env, a)
        case _:
            tc = TypeCheck()
            print(ast)
            try:
                result = tc.infer(ast, base_env, set())
                print(result)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    pair_type = TypeOperator("*", *typevars(2))
    environment = {
        "true": BoolType,
        "false": BoolType,
        "*": Function(IntType, Function(IntType, IntType)),
        "tuple2": Function(pair_type[0], Function(pair_type[1], pair_type)),
        "pair": Function(pair_type[0], Function(pair_type[1], pair_type)),
    }

    x = Identifier("x")
    pair = Identifier("pair")
    lit5 = IntLit(5)
    lit3 = IntLit(3)
    true = BoolLit(True)
    false = BoolLit(False)

    tests = [
        # Should fail:
        # fn x => (pair ((x 3) ((x true)))
        Lambda(x.name, pair(x(lit5), x(true))),
        # Should pass
        Lambda(x.name, pair(x(lit5), x(lit3))),
        Lambda(x.name, pair(x(true), x(false))),
    ]
    test(environment, tests)
