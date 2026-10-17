# 02 — Configuration, logging and a typed CLI (step S0.2)

## 1. Why one `Settings` class
In v1, model names and paths were hard-coded in several files (PLAN §5 #16). Changing a model meant a search through the code, and nothing said which value was really in use. v2 has one class, `geekbot.config.Settings`, as the only place that reads the environment. Every other module receives a `Settings` or a value taken from it. Then "which model is running?" has exactly one answer.

## 2. 12-factor configuration
The idea (from the "twelve-factor app" guide): **the same code runs everywhere; only the environment differs.** Settings are not edited in the source. They come from environment variables, for example `GEEKBOT_PROFILE=target`.
- A `.env` file is a convenient place to keep those variables on your own PC. It's git-ignored, because it is yours (and may hold secrets).
- `.env.example` is committed. It lists every option, so a new machine knows what exists.
- Priority in pydantic-settings, highest first: arguments passed in code → real environment variables → `.env` → the defaults in the class.

## 3. pydantic-settings
`class Settings(BaseSettings)` declares each setting with a type and a default. At startup pydantic reads `GEEKBOT_<NAME>` for each field, converts the text to the type (`"DEBUG"` to a log level, `"data"` to a `Path`) and **fails immediately** with a clear error if a value is wrong. Failing at startup is better than failing in the middle of an ingest.
- `Literal["proto", "target"]` makes any other profile name an error.
- `frozen=True`: settings can't change after creation, so no function can change them for everyone else by accident.
- `extra="ignore"`: unknown variables in `.env` are ignored. Trade-off: a typo like `GEEKBOT_PROFLE` is silently ignored, so `geekbot config show` is how you check what is really in effect.

## 4. Profiles: how Geekbot picks models
`proto` = `gemma3:4b` + `qwen2.5-coder:3b` (fast, for building). `target` = `gemma4:12b` + `qwen2.5-coder:7b` (the only source of reported numbers). **Default: `proto`** (D-043). An override variable beats the profile. Every reported number must name its model. This is how a default that favors speed doesn't cost us honest results.

## 5. Secrets and masking
`SecretStr` is a string that hides itself: `repr` shows `SecretStr('**********')`. The real text is only available through `.get_secret_value()`. Masking is a second defense for output we write ourselves (`config show`): even when the secret is set, the output says only `"********"`. Two more rules: Typer's pretty tracebacks are set not to show local variables (a crash must not print a secret), and a test checks the secret never appears in the output.

## 6. Logging
Python's standard `logging` is enough. We configure only the `geekbot` logger, never the root logger, so third-party libraries keep their own behavior. Modules use `logging.getLogger(__name__)`, which is a child of `geekbot` and inherits its level. Logs go to stderr, so stdout holds only a command's real output (e.g. JSON you may pipe elsewhere).

## 7. Typer
Typer builds a command-line tool from ordinary typed functions: the parameters become arguments and options, the docstring becomes the help text. Sub-commands (`geekbot config show`, `geekbot csv ask`) are sub-apps. Commands whose steps aren't built yet are stubs that exit with code 2: **a non-zero exit code means "this did not succeed"**, which scripts and CI rely on. 2 is the conventional code for "the command can't be used as invoked".

## Interview-style questions
1. **Why not read `os.environ` wherever a value is needed?** Scattered reads mean no types, no validation, no single list of what the program needs, and hard-to-test code. One typed class validates once at startup.
2. **What is 12-factor config?** Config lives in the environment, not in code, so one build runs in dev, CI and production with only the environment changing.
3. **Why does a stub exit 2 and not 0?** Exit code 0 means success; a script that calls an unbuilt command must see a failure.
4. **A secret is a `SecretStr`. Why also mask it in `config show`?** `SecretStr` protects against accidental printing of the object; our own output code could still call `.get_secret_value()`. The test pins the behavior down.
5. **Why is the default profile `proto`, and what protects the honest results?** It speeds up daily work; the protection is that every result records the resolved model, and reported numbers must come from `target` explicitly.
