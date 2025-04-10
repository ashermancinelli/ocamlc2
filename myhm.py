"""
http://lucacardelli.name/Papers/BasicTypechecking.pdf
https://web.archive.org/web/20050420002559/http://www.cs.berkeley.edu/~nikitab/courses/cs263/hm.pl
https://raw.githubusercontent.com/rob-smallshire/hindley-milner-python/refs/heads/master/inference.py
"""


class AST:
    pass


class Identifier(AST):
    def __init__(self, name):
        self.name = name
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
        return f'{self.lhs} {self.op} {self.rhs}'


class Let(AST):
    def __init__(self, name, value, body):
        self.name, self.value, self.body = name, value, body
    def __str__(self):
        return f'let {self.name} = {self.value} in {self.body}'


class Lambda(AST):
    def __init__(self, param, body):
        self.param, self.body = param, body

    def __str__(self):
        return f"(\\{self.param} -> {self.body})"


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


class TypeOperator:
    def __init__(self, name, *types: list[TypeExpr]):
        self.name = name
        self.types = types

    def __str__(self):
        match self.types:
            case 0:
                return self.name()
            case [l, r]:
                return f'({l} {self.name} {r})'
            case _:
                return f'{self.name} {' '.join(self.types)}'

    def __getitem__(self, index):
        return self.types[index]


class Function(TypeOperator):
    def __init__(self, from_type, to_type):
        super().__init__('->', from_type, to_type)

IntType = TypeOperator("int")
BoolType = TypeOperator("bool")


class TypeCheck:
    def __init__(self, environment):
        self.environment: dict[str, TypeExpr] = environment

    def infer(self, ast, concrete_types: list[TypeExpr] = None) -> TypeVariable:
        if concrete_types is None:
            concrete_types = set()

        match ast:
            case Identifier(name=name):
                return self.typeof(name, concrete_types)
            case IntLit():
                return IntType
            case BoolLit():
                return BoolType
            case Apply(fn=fn, arg=arg):
                fun_type = self.infer(fn, concrete_types)
                arg_type = self.infer(arg, concrete_types)
                result_type = TypeVariable()
                self.unify_type_expressions(Function(arg_type, result_type), fun_type)
                return result_type
            case Lambda(param=param, body=body):
                arg_type = TypeVariable()
                env = self.environment.copy()
                concrete = concrete_types.copy()
                self.environment[param] = arg_type
                result_type = self.infer(body, concrete)
                self.environment = env
                concrete_types = concrete
                return Function(arg_type, result_type)
            case Let(name=name, value=value, body=body):
                value_type = self.infer(value, concrete)
                env = self.environment.copy()
                concrete = concrete_types.copy()
                self.environment[name] = value_type
                body_type = self.infer(body, concrete)
                self.environment = env
                concrete_types = concrete
                return body_type
            case _:
                assert False

    def typeof(self, name: str, concrete_types: set[TypeExpr]) -> TypeVariable:
        if name in self.environment:
            return self.deep_copy_with_new_generic_typevars(
                self.environment[name], concrete_types
            )
        raise RuntimeError(f"Undefined symbol: {name}")

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

    def prune_type_expression(self, type_expression: TypeExpr) -> TypeExpr:
        type_expression.instance = self.prune(type_expression.instance)
        return type_expression.instance

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

    def is_concrete(self, expr: TypeExpr, concrete_type_exprs: list[TypeExpr]) -> bool:
        return self.is_sub_type_expression_of_any(expr, concrete_type_exprs)

    def is_generic(self, expr: TypeExpr, concrete_type_exprs: list[TypeExpr]) -> bool:
        return not self.is_sub_type_expression_of_any(expr, concrete_type_exprs)

    def is_sub_type_expression_of_any(
        self, maybe_subexpr: TypeExpr, expr_iterable: list[TypeExpr]
    ) -> bool:
        return any(
            self.is_sub_type_expression_of(maybe_subexpr, member_type)
            for member_type in expr_iterable
        )

    def is_sub_type_expression_of(
        self, maybe_subexpr: TypeExpr, expr: TypeExpr
    ) -> bool:
        expr = self.prune(expr)
        match expr:
            case TypeVariable():
                return expr == maybe_subexpr
            case TypeOperator():
                return self.is_sub_type_expression_of_any(maybe_subexpr, expr.types)

    def unify_type_expressions(self, expr1: TypeExpr, expr2: TypeExpr) -> None:
        expr1, expr2 = self.prune(expr1), self.prune(expr2)
        match expr1:
            case TypeVariable():
                if self.is_sub_type_expression_of(expr1, expr2) and expr1 != expr2:
                    raise RuntimeError(
                        f"Recursive unification when trying to unify these types: {expr1} {expr2}"
                    )
                expr1.assign(expr2)

            case TypeOperator():
                match expr2:
                    case TypeVariable():
                        self.unify_type_expressions(expr2, expr1)
                    case TypeOperator():
                        if expr1.name != expr2.name or len(expr1.types) != len(
                            expr2.types
                        ):
                            raise RuntimeError(f"Type mismatch: {expr1} {expr2}")
                        for subtype1, subtype2 in zip(expr1.types, expr2.types):
                            self.unify_type_expressions(subtype1, subtype2)
            case _:
                print(expr1, expr2)
                assert False


def typevars(how_many: int):
    return [TypeVariable() for _ in range(how_many)]


def test(base_env, ast: AST):
    tc = TypeCheck(base_env)
    print(ast)
    try:
        result = tc.infer(ast)
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
    }

    ast = Lambda(
        "x",
        Apply(
            Apply(Identifier("tuple2"), Apply(Identifier("x"), IntLit(3))),
            Apply(Identifier("x"), BoolLit(True)),
        ),
    )
    test(environment, ast)

    ast = Lambda(
        "x",
        Apply(
            Apply(Identifier("tuple2"), Apply(Identifier("x"), IntLit(3))),
            Apply(Identifier("x"), IntLit(5)),
        ),
    )
    test(environment, ast)
