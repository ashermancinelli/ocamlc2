(*
RUN: p3 -d -f %s | FileCheck %s.ref
XFAIL: *
*)
module type OrderedType = sig
  type t
  val compare : t -> t -> int
end
module type S = sig
  type key
end

module Make (Ord : OrderedType) : S with type key = Ord.t

module M = struct
  type t = int
  let compare a b = a - b
end

module N = Make(M)

(*
module_type_definition: ... 0:0-3:3 module type OrderedType = sig type t = 't8 val compare : (t -> t -> int) end
| module: module 0:0-0:6
| type: type 0:7-0:11
| module_type_name: OrderedType 0:12-0:23 module type OrderedType = sig type t = 't8 val compare : (t -> t -> int) end
| =: = 0:24-0:25
| body: 
| signature: ... 0:26-3:3
| | sig: sig 0:26-0:29
| | type_definition: type t 1:2-1:8 t
| | | type: type 1:2-1:6
| | | type_binding: t 1:7-1:8 t
| | | | name: 
| | | | type_constructor: t 1:7-1:8
| | value_specification: val compare : t -> t -> int 2:2-2:29 (t -> t -> int)
| | | val: val 2:2-2:5
| | | value_name: compare 2:6-2:13 (t -> t -> int)
| | | :: : 2:14-2:15
| | | type: 
| | | function_type: t -> t -> int 2:16-2:29 (t -> t -> int)
| | | | domain: 
| | | | type_constructor_path: t 2:16-2:17 t
| | | | | type_constructor: t 2:16-2:17
| | | | ->: -> 2:18-2:20
| | | | codomain: 
| | | | function_type: t -> int 2:21-2:29
| | | | | domain: 
| | | | | type_constructor_path: t 2:21-2:22 t
| | | | | | type_constructor: t 2:21-2:22
| | | | | ->: -> 2:23-2:25
| | | | | codomain: 
| | | | | type_constructor_path: int 2:26-2:29 int
| | | | | | type_constructor: int 2:26-2:29
| | end: end 3:0-3:3
module_type_definition: ... 4:0-6:3
| module: module 4:0-4:6
| type: type 4:7-4:11
| module_type_name: S 4:12-4:13
| =: = 4:14-4:15
| body: 
| signature: ... 4:16-6:3
| | sig: sig 4:16-4:19
| | type_definition: type key 5:2-5:10
| | | type: type 5:2-5:6
| | | type_binding: key 5:7-5:10
| | | | name: 
| | | | type_constructor: key 5:7-5:10
| | end: end 6:0-6:3
module_definition: module Make (Ord : OrderedType) : S with type key = Ord.t 8:0-8:57
| module: module 8:0-8:6
| module_binding: Make (Ord : OrderedType) : S with type key = Ord.t 8:7-8:57
| | module_name: Make 8:7-8:11
| | module_parameter: (Ord : OrderedType) 8:12-8:31
| | | (: ( 8:12-8:13
| | | module_name: Ord 8:13-8:16
| | | :: : 8:17-8:18
| | | module_type: 
| | | module_type_path: OrderedType 8:19-8:30
| | | | module_type_name: OrderedType 8:19-8:30
| | | ): ) 8:30-8:31
| | :: : 8:32-8:33
| | module_type: 
| | module_type_constraint: S with type key = Ord.t 8:34-8:57
| | | module_type: 
| | | module_type_path: S 8:34-8:35
| | | | module_type_name: S 8:34-8:35
| | | with: with 8:36-8:40
| | | constrain_type: type key = Ord.t 8:41-8:57
| | | | type: type 8:41-8:45
| | | | type_constructor_path: key 8:46-8:49
| | | | | type_constructor: key 8:46-8:49
| | | | =: = 8:50-8:51
| | | | equation: 
| | | | type_constructor_path: Ord.t 8:52-8:57
| | | | | extended_module_path: Ord 8:52-8:55
| | | | | | module_name: Ord 8:52-8:55
| | | | | .: . 8:55-8:56
| | | | | type_constructor: t 8:56-8:57
comment: ... 10:0-11:2
*)
