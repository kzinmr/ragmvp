# RAG MVP

- [LanceDB](https://github.com/lancedb/lancedb)



# `api`

## Run RAG

```python
python src/main.py
```


# Notes

- `src/ragmvp` structure:
  - It's due to [hatch's default file selection rules](https://hatch.pypa.io/latest/plugins/builder/wheel/#default-file-selection).
  - `hatchling` is used uv's default build-system backend.

- `uv.lock` is ignored in `.gitignore`:
  - > However, in case of collaboration, if having platform-specific dependencies or dependencies having no cross-platform support, pipenv may install dependencies that don't work, or not install all needed dependencies.

- `pydantic.mypy` doesn't work with `uv`:
  - `plugins = ["pydantic.mypy"]` in `pyproject.toml` raises error `No module named 'pydantic'`.
  - pre-commit only works. (See. [issues/3953](https://github.com/pydantic/pydantic/issues/3953#issuecomment-1399178132))
