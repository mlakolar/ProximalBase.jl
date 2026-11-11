# Reviewable declaration-level milestone-0 policy; no package code is evaluated.
const STORAGE_FILES = Set(["src/sparse_iterate.jl", "src/block_iterate.jl"])
const CORE_TYPES = Set(["AbstractFunction", "SeparableFunction", "ScalarLoss", "Squared",
  "ZeroFunction", "LinearFunction", "SqrNormL2", "QuadraticFunction", "LeastSquaresLoss", "NormL1", "Scaled"])
const LATE_SCALE_NAMES = Set(["_hard_scale", "_steps_hard", "_has_infinite_level", "_hard_function", "_hard"])
const SECOND_ORDER_NAMES = Set(["hessian_vec!", "hessian_vec", "hashessian", "quadratic_model", "quadratic_model!",
  "coordinate_lipschitz!", "coordinate_lipschitz", "gradient_mapping!", "gradient_mapping"])
const SYMMETRIC_ARCHIVE_NAMES = Set(["_check_entry_array_blocks",
  "_stored_box_value", "_builtin_kernels", "_check_value_shapes", "_check_value_shape_blocks",
  "_check_write_array_blocks", "_OwnerLipschitz", "_owner_role_error", "_owner_role_supported",
  "_owner_role_preflight", "_owner_role_preflight_blocks", "_hessian_vec_unchecked!", "_quadratic_model_unchecked!",
  "_quadratic_model_unchecked", "_coordinate_lipschitz_unchecked!", "_check_second_order_arrays",
  "_check_lipschitz_arrays", "_check_hessian_shape", "_throw_product_shape", "_check_product_dims",
  "_check_hessian_support", "_check_hessian_support_parts", "_check_hessian_parts", "_check_hessian_part",
  "_check_hessian_blocks", "_sum_checked_values", "_in_checked_sets", "_l0_point", "_accumulate!",
  "_value_dot_normal", "_value_scaled_dot", "_checked_scaled_dot"])
const ALL_TYPES = Set(r.name for r in rows if r.kind in ("abstract", "struct", "primitive") && !isempty(r.name))
const ALL_FUNCTION_TYPES = Set(r.name for r in rows if r.kind in ("abstract", "struct") &&
  ("AbstractFunction" in r.refs || "SeparableFunction" in r.refs || "ScalarLoss" in r.refs))
const ALLOWED_INTERNAL = Set(["Scratch", "PrecisionError", "_SplitAcc", "_ExactZero", "_ExpSum", "_ShiftableIndex",
  "_RidgeOrdinary", "_RidgeScaled", "_OrdinaryRidge", "_ProjectionScale", "_ScaledArray", "_ExpScaledArray",
  "_NormalScale", "_ShiftWindow", "_LevelProbe", "_StepInput", "_AllPairs", "_StoredPairs", "_StoredValuePairs", "_CoordinatePair", "_SymmetricGradient", "_CheckedParameter"])
const ALLOWED_TYPES = union(CORE_TYPES, ALLOWED_INTERNAL,
  Set(r.name for r in rows if r.path in STORAGE_FILES && r.kind in ("struct", "abstract", "primitive")))
const DEFERRED_TYPES = union(setdiff(ALL_TYPES, ALLOWED_TYPES), Set(["_ScalarPrecompose", "_ArrayPrecompose", "_PlainIndicator"]))
const SHARED_NAMES = Set(["_exactly_zero_at", "_data_type", "entry_parameters", "_inner_functions",
  "_stored_value", "_value_and_prox_unchecked!", "prox", "_value_dense"])

function seed(r)
  p, l, n = r.path, r.group_start, r.name
  p in STORAGE_FILES && return true
  p == "src/ProximalBase.jl" && return n == "ldexp"
  p == "src/lapack.jl" && return n == "_is_device_array"
  p == "src/utils.jl" && return n == "opnorm2"
  p == "src/constrained.jl" && return n == "_checkwrite" && l in (211, 213, 221)
  p == "src/interface.jl" && return l <= 610 && !(n in SECOND_ORDER_NAMES) && n != "_excluded_coordinates" &&
    !(n in ("kkt_violation", "_kkt_violation") && "BlockIterate" in r.typed)
  p == "src/functions/smooth.jl" && return l < 975 || (p == "src/functions/smooth.jl" && n == "lipschitz" && !isempty(intersect(CORE_TYPES, r.typed)))
  p == "src/functions/leastsquares.jl" && return n != "_xty_cache!"
  p == "src/functions/glm.jl" && return l <= 81
  p == "src/functions/separable.jl" && return (l <= 390 && !(n in ("_sep_value_parts", "_sep_terms", "_term_add", "_weighted_term_add", "_rparts", "_sep_parts_type", "_parts_eltype"))) || (p == "src/functions/separable.jl" &&
    (n == "_HostArray" || n == "_l1_value_sum" || (n in ("_l1_value", "_value_dense") && "NormL1" in r.typed)))
  p == "src/calculus.jl" && return (l <= 821 && !("_ShiftedArray" in r.refs) && n != "args") ||
    (p == "src/calculus.jl" && ((n in LATE_SCALE_NAMES && 2260 <= l <= 2328) || (n == "lipschitz" && "Scaled" in r.typed)))
  p == "src/model_data.jl" && return n in ("_data_type", "_data_float", "_data_of") &&
    (isempty(intersect(r.typed, ALL_FUNCTION_TYPES)) || !isempty(intersect(r.typed, CORE_TYPES)))
  p == "src/metric.jl" && return l <= 39 || (p == "src/metric.jl" && 225 <= l <= 333)
  p == "src/symmetric_parameters.jl" && return !(n in SYMMETRIC_ARCHIVE_NAMES) && !(n in SECOND_ORDER_NAMES) &&
    !("Tuple" in r.typed && isempty(intersect(r.typed, CORE_TYPES)) && !(n in ("_check_own_arrays", "_check_inner_arrays", "_stored_value_pairs", "_has_symmetric")))
  p == "src/block_prox.jl" && return n == "prox_block" && "AbstractFunction" in r.typed
  false
end

function eligible(r)
  bad = intersect(Set(r.typed), DEFERRED_TYPES)
  isempty(bad) && return true
  # Only these explicitly documented shared unions may be narrowed to surviving arms.
  r.name in SHARED_NAMES && !isempty(intersect(Set(r.typed), CORE_TYPES))
end

const SELECTED = Set(r.id for r in rows if seed(r) && eligible(r))
# Shared low-level helper closure. Overloaded helper names select all eligible same-name helpers.
let changed = true
  while changed
    changed = false
    needed = Set{String}()
    for r in rows
      r.id in SELECTED && union!(needed, r.refs)
    end
    for r in rows
      r.path in ("src/utils.jl", "src/compat.jl") || continue
      eligible(r) || continue
      r.name in DEFERRED_TYPES && continue
      (r.name in needed || last(split(r.name, '.')) in needed) || continue
      if !(r.id in SELECTED)
        push!(SELECTED, r.id)
        changed = true
      end
    end
  end
end

function disposition(r)
  r.path in STORAGE_FILES && return ("retain", "Complete post-848 storage implementation/products; numerical source unchanged")
  if r.id in SELECTED
    bad = intersect(Set(r.refs), DEFERRED_TYPES)
    !isempty(bad) && return ("extract", "Retain core arms; remove deferred typed/body references: " * join(sort!(collect(bad)), ','))
    return ("extract", "Retained core declaration or transitive low-level dependency; preserve formula/body")
  end
  ("archive", "Outside exact retained boundary or exclusively deferred dispatch/helper")
end
