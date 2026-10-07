# Legacy material

Files kept from the original research repository
(`github.com/YadhHafsi/execution`, branch `yadh-branch`, a fork of
`jpmorganchase/abides-jpmc-public`) for reference. Nothing in this directory is
used by the `experiments/` package or maintained.

* `version_testing/`: upstream ABIDES regression utilities that run an RMSC-3
  configuration at the current and at a past commit and compare the logs. They
  expect to sit directly under the repository root, so from `legacy/` they need
  path adjustments.
* `profiling/profile_rmsc.py`: upstream script that profiles an RMSC-3 run.
* `install.sh`, `setup-dev.sh`: the original install scripts, superseded by the
  instructions in the top-level README and by `make install`.
