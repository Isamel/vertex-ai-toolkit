# Release Process

This document describes how to create a new release of the Vertex AI Toolkit.

## Overview

Releases are fully automated via GitHub Actions. Pushing a `v*` tag triggers the
build workflow which:

1. Builds standalone binaries for **Linux (amd64)** and **Windows (amd64)** using PyInstaller
2. Verifies each binary runs (`vaig --version`)
3. Creates a **GitHub Release** with auto-generated release notes
4. Attaches both binaries as release assets

## Prerequisites

- Push access to the `main` branch (or a release branch)
- The CI workflow (`ci.yml`) must be passing — all tests, ruff, and mypy green
- `CHANGELOG.md` updated with the new version's changes

## Creating a Release

### 1. Update the version

Update the version in `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

### 2. Update the changelog

Add a new section to `CHANGELOG.md` following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- ...

### Fixed
- ...

### Changed
- ...
```

Add the comparison link at the bottom:

```markdown
[X.Y.Z]: https://github.com/Isamel/vertex-ai-toolkit/compare/vPREVIOUS...vX.Y.Z
```

### 3. Commit and tag

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

### 4. Monitor the build

The build workflow will start automatically. Monitor it at:

```
https://github.com/Isamel/vertex-ai-toolkit/actions/workflows/build.yml
```

The workflow runs two parallel build jobs (Linux + Windows), then a release job
that creates the GitHub Release once both builds succeed.

### 5. Verify the release

Once complete, check the release at:

```
https://github.com/Isamel/vertex-ai-toolkit/releases/tag/vX.Y.Z
```

Verify that:
- Both `vaig` (Linux) and `vaig.exe` (Windows) are attached
- Release notes are auto-generated from commits since the last tag
- The prerelease flag is correct (only set for `-rc`, `-beta`, `-alpha` tags)

## Versioning Strategy

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

| Version Part | When to Bump | Example |
|---|---|---|
| **MAJOR** (X) | Breaking changes to CLI interface, config format, or public API | `1.0.0` → `2.0.0` |
| **MINOR** (Y) | New features, tools, skills, or agents (backward-compatible) | `0.1.0` → `0.2.0` |
| **PATCH** (Z) | Bug fixes, performance improvements, doc updates | `0.1.0` → `0.1.1` |

### Pre-release Tags

For release candidates or beta versions, append a suffix:

```bash
git tag v0.2.0-rc.1   # Release candidate
git tag v0.2.0-beta.1  # Beta release
git tag v0.2.0-alpha.1 # Alpha release
```

The build workflow automatically marks these as **prerelease** on GitHub.

## Build Artifacts

| Artifact | Platform | Location |
|---|---|---|
| `vaig` | Linux amd64 | `artifacts/vaig-linux-amd64/vaig` |
| `vaig.exe` | Windows amd64 | `artifacts/vaig-windows-amd64/vaig.exe` |

Artifacts are retained for 30 days on every build (including `workflow_dispatch`).
Release assets are permanent — attached to the GitHub Release.

## Workflow Architecture

```
build.yml
├── build (matrix: ubuntu-latest, windows-latest)
│   ├── Checkout code
│   ├── Setup Python 3.12
│   ├── Install dependencies ([live])
│   ├── PyInstaller --onefile build
│   ├── Verify binary (--version)
│   └── Upload artifact (30-day retention)
│
└── release (needs: build, only on v* tags)
    ├── Download all artifacts
    └── Create GitHub Release
        ├── Auto-generated release notes
        ├── Prerelease detection (-rc, -beta, -alpha)
        └── Attach vaig + vaig.exe
```

## Manual Builds

You can trigger a build without creating a release using the **workflow_dispatch**
trigger. Go to Actions → Build Standalone Binaries → Run workflow. This builds
and uploads artifacts but does not create a release.

## Troubleshooting

### Build fails on PyInstaller

PyInstaller requires explicit `--hidden-import` and `--collect-all` for packages
that use dynamic imports (google-genai, google-auth, kubernetes, pydantic).
If a new dependency is added, it may need to be listed in the build step.

### Release not created

The release job only runs when `github.ref` starts with `refs/tags/v`. Verify:

```bash
git tag -l          # List local tags
git ls-remote --tags origin  # List remote tags
```

### Binary crashes on startup

Check that `config/default.yaml` is included via `--add-data`. The binary needs
this file at runtime. Platform-specific path separators are handled by the
matrix (`":"` for Linux, `";"` for Windows).
