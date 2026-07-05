# pharos B — Complete (CLI multi-input + variable substitution + permissions)

**Status:** B done. **248 tests pass (2 integration skipped), ruff clean, end-to-end CLI supports `--input-extra`, `--var`, and `--grant`.**

---

## A — CLI multi-port input + variable substitution

### `--input-extra port=value` (repeatable)

Seed multiple `__in__.<port>` edges with one flag per input. Previously the CLI only seeded `__in__.prompt`, blocking graphs with multiple inputs (e.g. `text` + `msg`).

```bash
# my-python-agent.yaml has both __in__.text and __in__.msg
uv run pharos run my-python-agent.yaml \
    --input-extra "text=hello world" \
    --input-extra "msg=greetings"
# → word_count: 2  /  prefixed_msg: 🎯 greetings
```

### `--var key=value` (repeatable)

Replace `${NAME}` placeholders inside double-quoted YAML strings
before loading. Comments and unquoted values are left alone
(so a `# uses ${comment_var}` line stays as-is).

```yaml
# template-demo.yaml
nodes:
  - id: agent
    type: faux
    model: "${model_name}"
    system: "${system_prompt}"
```

```bash
uv run pharos run template-demo.yaml \
    --var model_name=faux-fast \
    --var "system_prompt=You are concise." \
    --input "hi"
```

Unknown vars fail fast with a clear error message:
```
ValueError: graph references ${system_prompt} but --var system_prompt=... was not provided
```

---

## B — Permissions (RBAC)

Each Entity can declare `required_permissions` as a class attribute:

```python
@entity
class ShellEntity(Entity):
    required_permissions = {"shell:execute"}
    # ...
```

The Director checks `entity.required_permissions ⊆ ctx.granted_permissions`
**before** calling `setup()`. If any required permission is missing,
the run fails with `PermissionError` and a clear message:

```
PermissionError: entity 'runner' (ShellEntity) requires ['shell:execute']
but run only grants no permissions
```

### CLI: `--grant permission` (repeatable)

```bash
# Without grant → permission error
uv run pharos run graphs/02_llm_then_shell.yaml --input "ls"
# → PermissionError: requires ['shell:execute']

# With grant → runs
uv run pharos run graphs/02_llm_then_shell.yaml --input "ls" --grant shell:execute
# → converged=True
```

### Design choices

- **Class-level attribute, not constructor** — `required_permissions`
  is part of the entity's contract, like `ins`/`outs`. Subclasses
  can override.
- **Set membership, not granular** — A permission is either granted
  or not. No "read-only vs read-write" granularity yet. (P7+
  if needed.)
- **Check happens once per fire** — Even though a node fires
  multiple times in SDF, the permission check only matters at
  the first setup. `_safe_fire` does the check on every fire for
  simplicity; cost is negligible.

---

## Verification

```bash
# A.1 multi-port seed
uv run pharos run my-python-agent.yaml --input-extra "text=hello world" --input-extra "msg=greetings"
# word_count: 2   prefixed_msg: 🎯 greetings

# A.2 variable substitution
uv run pharos run template-demo.yaml --var model_name=faux-fast --var "system_prompt=You are concise." --input "hi"
# model="${model_name}" → model="faux-fast", system="You are concise."

# B.1 without permission
uv run pharos run graphs/02_llm_then_shell.yaml --input "ls"
# → PermissionError: entity 'runner' (ShellEntity) requires ['shell:execute']

# B.2 with permission
uv run pharos run graphs/02_llm_then_shell.yaml --input "ls" --grant shell:execute
# → converged=True
```

---

## Files touched

| Path | Change |
| --- | --- |
| `pharos/cli.py` | `--input-extra`, `--var`, `--grant` flags; `_seed_inputs`, `_substitute_vars` |
| `pharos/ir/__init__.py` | `load_graph_from_text`, `load_graph_from_dict` |
| `pharos/core/entity.py` | `required_permissions: ClassVar[set[str]]` |
| `pharos/core/entity.py` | docs explaining permission contract |
| `pharos/entities/shell.py` | `required_permissions = {"shell:execute"}` |
| `pharos/directors/base.py` | `RunContext.granted_permissions: set[str]` |
| `pharos/directors/fn.py` | `_safe_fire` permission check |
| `tests/cli/test_cli_seed_and_vars.py` | NEW: 11 tests for A |
| `tests/auth/test_permissions.py` | NEW: 6 tests for B |
| `tests/entities/test_entities.py` | Updated 3 ShellEntity tests |

---

## Limitations

- **Permission is all-or-nothing per entity** — you can't say
  "this Shell can only run `ls`". A future version could parse
  `required_permissions = {"shell:execute:ls"}` for allow-listing.
- **No revocation** — once granted, permissions stay granted for
  the run. A long-running SDF loop with a malicious reviewer
  can't escalate, because permissions are checked at fire time
  but the granted set is static.
- **No audit log** — who granted what, when, isn't recorded.
  Could integrate with trace in P7+.

---

## Reproduction

```bash
cd /Users/heshuwen/pharos
uv run pytest                       # 248 passed, 2 skipped
uv run ruff check .                 # 0 errors
uv run python bench/hello_world.py  # P95 ≈ 12ms (unchanged)

# A: multi-port seed
uv run pharos run my-python-agent.yaml --input-extra "text=hello" --input-extra "msg=world"

# A: variable substitution
uv run pharos run template-demo.yaml --var model_name=faux-fast --var "system_prompt=Be brief."

# B: permission denied
uv run pharos run graphs/02_llm_then_shell.yaml --input "ls"

# B: permission granted
uv run pharos run graphs/02_llm_then_shell.yaml --input "ls" --grant shell:execute
```