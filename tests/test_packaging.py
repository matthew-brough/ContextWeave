import importlib.resources as resources


def test_schema_sql_exists() -> None:
    """Verify schema.sql is packaged."""
    files = resources.files("contextweave.index")
    assert (files / "schema.sql").is_file()


def test_lang_queries_exist() -> None:
    """Verify language query files (.scm) are packaged."""
    for query_pkg in ("core", "defs", "edges"):
        files = resources.files(f"contextweave.lang.queries.{query_pkg}")
        assert any(f.name.endswith(".scm") for f in files.iterdir()), query_pkg


def test_prompts_exist() -> None:
    """Verify prompt TOML files are packaged."""
    files = resources.files("contextweave.models.prompts")
    assert (files / "annotate.toml").is_file()


def test_eval_files_exist() -> None:
    """Verify evaluation suite and baseline files are packaged."""
    eval_files = resources.files("contextweave.eval")

    suites_dir = eval_files / "suites"
    assert (suites_dir / "self.toml").is_file()

    baselines_dir = eval_files / "baselines"
    assert (baselines_dir / "self.nomodel.json").is_file()


def test_agents_md_exists() -> None:
    """Verify agent markdown files are packaged."""
    files = resources.files("contextweave.agents")
    assert (files / "routing.md").is_file()
