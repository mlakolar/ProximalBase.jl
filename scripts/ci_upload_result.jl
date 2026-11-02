# The "Conformance report upload result" step of the CI test job. An upload counts only when
# its step succeeded and returned an artifact id. The Julia lts, Linux artifact, from which the
# coverage map is regenerated, is required: without a confirmed upload the step exits 1. On the
# other jobs a missing upload is a warning, and the job keeps the result of its tests. The
# outcome goes to the job summary in every case, before the step fails.
#
#   FIRST=<outcome> FIRST_ID=<id> RETRY=<outcome> RETRY_ID=<id> REQUIRED=true|false \
#   TESTS=<test step outcome> JOB=<name> julia scripts/ci_upload_result.jl
#
# test/ci_gate.jl tests the decisions. No packages are needed.

module CIUploadResult

confirmed(outcome, id) = outcome == "success" && !isempty(id)

function describe(outcome, id)
  (isempty(outcome) || outcome == "skipped") && return "not run"
  outcome == "success" || return outcome
  return isempty(id) ? "succeeded without an artifact id" : "artifact $id"
end

"""
    upload_result(; required, tests, first, first_id, retry, retry_id, job) -> (level, line)

`first` and `retry` are the outcomes of the two upload steps (`""` when a step did not run),
`first_id` and `retry_id` the artifact ids they returned and `tests` the outcome of the test
step. `level` is `:ok`, `:notice`, `:warning` or `:error`; only `:error` fails the step.
"""
function upload_result(; required::Bool, tests, first, first_id, retry, retry_id, job)
  passed = tests == "success"
  failed_tests = "the test step failed ($(isempty(tests) ? "no outcome" : tests)), which fails the job"
  what = passed ? "Conformance reports" : "Diagnostic conformance reports"
  tail = passed ? "" : " $(uppercasefirst(failed_tests))."
  confirmed(first, first_id) && return (:ok, "$what uploaded (artifact $first_id).$tail")
  confirmed(retry, retry_id) &&
    return (:notice, "$what uploaded on the retry (artifact $retry_id); first attempt: $(describe(first, first_id)).$tail")
  line = "GitHub did not confirm the upload of the conformance reports of $job " *
         "(first attempt: $(describe(first, first_id)); retry: $(describe(retry, retry_id)))."
  required && return (:error, "$line This artifact is required, so the job fails" *
                              (passed ? "." : "; $failed_tests too."))
  passed && return (:warning, "$line The tests passed; the artifact is missing, which does not fail this job.")
  return (:warning, "$line $(uppercasefirst(failed_tests)).")
end

function main(env::AbstractDict = ENV)
  required = get(env, "REQUIRED", "")
  required in ("true", "false") || error("REQUIRED must be true or false, not $(repr(required))")
  level, line = upload_result(; required = required == "true", tests = get(env, "TESTS", ""),
                              first = get(env, "FIRST", ""), first_id = get(env, "FIRST_ID", ""),
                              retry = get(env, "RETRY", ""), retry_id = get(env, "RETRY_ID", ""),
                              job = get(env, "JOB", "this job"))
  println(level == :ok ? "" : "::$level::", line)
  summary = get(env, "GITHUB_STEP_SUMMARY", "")
  isempty(summary) || open(io -> print(io, "\n**Conformance reports**: ", line, "\n"), summary, "a")
  return level == :error ? 1 : 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
  exit(CIUploadResult.main())
end
