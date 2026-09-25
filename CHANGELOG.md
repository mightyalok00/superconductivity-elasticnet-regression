# Changelog

All notable changes to this project are documented here.

## 1.3.0 — ML Engineering Hardening

- Removed hard-coded Windows user paths from runtime configuration.
- Added environment-based settings with `pydantic-settings`.
- Added optional API-key and rate-limit controls, disabled by default.
- Added `pyproject.toml` for Python project/tooling configuration.
- Added production Dockerfile, non-root runtime user, and container health check.
- Added lean `requirements-api.txt` for production images.
- Added `.dockerignore` and `.env.example`.
- Expanded API and configuration tests.
- Added coverage reporting and Ruff checks to CI.
- Added Docker image build verification to CI.
- Added feature standard deviations to model metadata.
- Added lightweight input-drift and API load-test utilities.
- Removed an unused modeling import caught by CI.


## 1.2.0 — Interactive ML Dashboard

- Added live single-material prediction on the homepage.
- Added top-10 model contribution explanations for each prediction.
- Added browser-session prediction history.
- Added two-material comparison mode.
- Added ranked batch CSV prediction with optional minimum-temperature filtering.
- Added grouped feature reference with descriptions and training ranges.
- Added out-of-distribution risk scoring.
- Added request IDs, latency reporting, structured logging, and live service metrics.
- Added dark/light theme support, favicon, robots.txt, and custom 404 handling.
- Added versioned `/api/v1` endpoints.
- Added model/data cards and architecture documentation.
- Added automated API tests and model-artifact CI.

## 1.1.0 — Production API Upgrade

- Added saved ElasticNet deployment artifact and model metadata.
- Added batch CSV predictions.
- Added training-range warnings.
- Added model schema and metadata endpoints.
- Added FastAPI CI workflow.

## 1.0.0 — Initial Deployment

- Added FastAPI service.
- Added Railway deployment configuration.
- Added interactive landing page.
- Added `/health`, `/docs`, `/model-info`, `/features`, `/sample`, and `/predict`.
- Added live Railway deployment.

## Analysis Milestone

- Completed all 16 structured analytical questions.
- Compared Linear Regression, Ridge, Lasso, and ElasticNet.
- Added 5-fold cross-validation, residual analysis, composition experiment, and scientific interpretation.
- Added README visualizations and MIT license.
