# Vendored copy of exPyrimenter

This folder is a vendored (copied-in) snapshot of Jakob Bossek's
[expyrimenter](https://github.com/jakobbossek/expyrimenter) package.

- Source: https://github.com/jakobbossek/expyrimenter
- Vendored commit: `47499f37c335d8648d037bdb82fc0fbd460aabca`
- Vendored on: 2026-08-20
- Reason: not yet published on PyPI; no LICENSE file present in the
  upstream repository at the time of vendoring, so this copy is used with
  the author's explicit permission (he is the Bachelorarbeit supervisor
  and asked whether/how his students use the package).

## Preferred alternative: convert to a git submodule

A vendored copy does not track upstream updates. If/when you can run a
`git submodule add` locally (needs network access, which this automated
import did not have), replace this folder with a real submodule instead:

```bash
git rm -r external/expyrimenter
git submodule add https://github.com/jakobbossek/expyrimenter.git external/expyrimenter
git commit -m "Track expyrimenter as a git submodule instead of a vendored copy"
```

A ready-to-use git bundle (`.expyrimenter.bundle.tmp` in the repo root, if
still present) already contains the full history of the vendored commit
and can be used offline instead of the GitHub URL above:

```bash
git submodule add ./.expyrimenter.bundle.tmp external/expyrimenter
# then fix the remote so future `git submodule update --remote` reaches upstream:
git config -f .gitmodules submodule.external/expyrimenter.url https://github.com/jakobbossek/expyrimenter.git
cd external/expyrimenter && git remote set-url origin https://github.com/jakobbossek/expyrimenter.git && cd ../..
git submodule sync
rm .expyrimenter.bundle.tmp
```

## Known issue found while integrating this package

`Registry.show_logs()` / `Job.log()` only print `"NoneType: None"` for
jobs that fail under `JoblibRunnerBackend`, instead of the real traceback.
`Job.log(e)` calls `traceback.print_exc()`, which needs an *active*
exception context -- but by the time `Registry.run()` calls
`job.set_failed().log(value)`, the exception has already been caught in
`runner_wrapper` and passed around as a plain value, so there is no active
context left. Worth reporting on the issue tracker.
