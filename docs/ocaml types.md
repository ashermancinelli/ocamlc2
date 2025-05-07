8 Type and exception definitions
8.1 Type definitions
8.2 Exception definitions
8.1 Type definitions
Type definitions bind type constructors to data types: either variant types, record types, type abbreviations, or abstract data types. They also bind the value constructors and record fields associated with the definition.


type-definition	::=	type [nonrec] typedef { and typedef }
 
typedef	::=	[type-params] typeconstr-name type-information
 
type-information	::=	[type-equation] [type-representation] { type-constraint }
 
type-equation	::=	= typexpr
 
type-representation	::=	= [|] constr-decl { | constr-decl }
 	∣	 = record-decl
 	∣	 = |
 
type-params	::=	type-param
 	∣	 ( type-param { , type-param } )
 
type-param	::=	[ext-variance] ' ident
 
ext-variance	::=	variance [injectivity]
 	∣	 injectivity [variance]
 
variance	::=	+
 	∣	 -
 
injectivity	::=	 !
 
record-decl	::=	{ field-decl { ; field-decl } [;] }
 
constr-decl	::=	(constr-name ∣ [] ∣ (::)) [ of constr-args ]
 
constr-args	::=	typexpr { * typexpr }
 
field-decl	::=	[mutable] field-name : poly-typexpr
 
type-constraint	::=	constraint typexpr = typexpr
See also the following language extensions: private types, generalized algebraic datatypes, attributes, extension nodes, extensible variant types and inline records.

Type definitions are introduced by the type keyword, and consist in one or several simple definitions, possibly mutually recursive, separated by the and keyword. Each simple definition defines one type constructor.

A simple definition consists in a lowercase identifier, possibly preceded by one or several type parameters, and followed by an optional type equation, then an optional type representation, and then a constraint clause. The identifier is the name of the type constructor being defined.

type colour =
  | Red | Green | Blue | Yellow | Black | White
  | RGB of {r : int; g : int; b : int}

type 'a tree = Lf | Br of 'a * 'a tree * 'a;;

type t = {decoration : string; substance : t'}
and t' = Int of int | List of t list
In the right-hand side of type definitions, references to one of the type constructor name being defined are considered as recursive, unless type is followed by nonrec. The nonrec keyword was introduced in OCaml 4.02.2.

The optional type parameters are either one type variable ' ident, for type constructors with one parameter, or a list of type variables ('ident1,…,'identn), for type constructors with several parameters. Each type parameter may be prefixed by a variance constraint + (resp. -) indicating that the parameter is covariant (resp. contravariant), and an injectivity annotation ! indicating that the parameter can be deduced from the whole type. These type parameters can appear in the type expressions of the right-hand side of the definition, optionally restricted by a variance constraint ; i.e. a covariant parameter may only appear on the right side of a functional arrow (more precisely, follow the left branch of an even number of arrows), and a contravariant parameter only the left side (left branch of an odd number of arrows). If the type has a representation or an equation, and the parameter is free (i.e. not bound via a type constraint to a constructed type), its variance constraint is checked but subtyping etc. will use the inferred variance of the parameter, which may be less restrictive; otherwise (i.e. for abstract types or non-free parameters), the variance must be given explicitly, and the parameter is invariant if no variance is given.

The optional type equation = typexpr makes the defined type equivalent to the type expression typexpr: one can be substituted for the other during typing. If no type equation is given, a new type is generated: the defined type is incompatible with any other type.

The optional type representation describes the data structure representing the defined type, by giving the list of associated constructors (if it is a variant type) or associated fields (if it is a record type). If no type representation is given, nothing is assumed on the structure of the type besides what is stated in the optional type equation.

The type representation = [|] constr-decl { | constr-decl } describes a variant type. The constructor declarations constr-decl1, …, constr-decln describe the constructors associated to this variant type. The constructor declaration constr-name of typexpr1 * … * typexprn declares the name constr-name as a non-constant constructor, whose arguments have types typexpr1 …typexprn. The constructor declaration constr-name declares the name constr-name as a constant constructor. Constructor names must be capitalized.

The type representation = { field-decl { ; field-decl } [;] } describes a record type. The field declarations field-decl1, …, field-decln describe the fields associated to this record type. The field declaration field-name : poly-typexpr declares field-name as a field whose argument has type poly-typexpr. The field declaration mutable field-name : poly-typexpr behaves similarly; in addition, it allows physical modification of this field. Immutable fields are covariant, mutable fields are non-variant. Both mutable and immutable fields may have explicitly polymorphic types. The polymorphism of the contents is statically checked whenever a record value is created or modified. Extracted values may have their types instantiated.

The two components of a type definition, the optional equation and the optional representation, can be combined independently, giving rise to four typical situations:

Abstract type: no equation, no representation.
 ‍
When appearing in a module signature, this definition specifies nothing on the type constructor, besides its number of parameters: its representation is hidden and it is assumed incompatible with any other type.
Type abbreviation: an equation, no representation.
 ‍
This defines the type constructor as an abbreviation for the type expression on the right of the = sign.
New variant type or record type: no equation, a representation.
 ‍
This generates a new type constructor and defines associated constructors or fields, through which values of that type can be directly built or inspected.
Re-exported variant type or record type: an equation, a representation.
 ‍
In this case, the type constructor is defined as an abbreviation for the type expression given in the equation, but in addition the constructors or fields given in the representation remain attached to the defined type constructor. The type expression in the equation part must agree with the representation: it must be of the same kind (record or variant) and have exactly the same constructors or fields, in the same order, with the same arguments. Moreover, the new type constructor must have the same arity and the same type constraints as the original type constructor.
The type variables appearing as type parameters can optionally be prefixed by + or - to indicate that the type constructor is covariant or contravariant with respect to this parameter. This variance information is used to decide subtyping relations when checking the validity of :> coercions (see section 11.7.7).

For instance, type +'a t declares t as an abstract type that is covariant in its parameter; this means that if the type τ is a subtype of the type σ, then τ t is a subtype of σ t. Similarly, type -'a t declares that the abstract type t is contravariant in its parameter: if τ is a subtype of σ, then σ t is a subtype of τ t. If no + or - variance annotation is given, the type constructor is assumed non-variant in the corresponding parameter. For instance, the abstract type declaration type 'a t means that τ t is neither a subtype nor a supertype of σ t if τ is subtype of σ.

The variance indicated by the + and - annotations on parameters is enforced only for abstract and private types, or when there are type constraints. Otherwise, for abbreviations, variant and record types without type constraints, the variance properties of the type constructor are inferred from its definition, and the variance annotations are only checked for conformance with the definition.

Injectivity annotations are only necessary for abstract types and private row types, since they can otherwise be deduced from the type declaration: all parameters are injective for record and variant type declarations (including extensible types); for type abbreviations a parameter is injective if it has an injective occurrence in its defining equation (be it private or not). For constrained type parameters in type abbreviations, they are injective if either they appear at an injective position in the body, or if all their type variables are injective; in particular, if a constrained type parameter contains a variable that doesn’t appear in the body, it cannot be injective.

The construct constraint ' ident = typexpr allows the specification of type parameters. Any actual type argument corresponding to the type parameter ident has to be an instance of typexpr (more precisely, ident and typexpr are unified). Type variables of typexpr can appear in the type equation and the type declaration.

8.2 Exception definitions

exception-definition	::=	exception constr-decl
 	∣	 exception constr-name = constr
Exception definitions add new constructors to the built-in variant type exn of exception values. The constructors are declared as for a definition of a variant type.

# exception E of int * string;;
exception E of int * string
The form exception constr-decl generates a new exception, distinct from all other exceptions in the system. The form exception constr-name = constr gives an alternate name to an existing exception.

# exception E of int * string

  exception F = E

  let eq =
     E (1, "one") = F (1, "one");;
exception E of int * string
exception F of int * string
val eq : bool = true
« Expressions
