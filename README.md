# infrawhisperer.github.io

Personal engineering blog of **Raghav Potluri** (handle: InfraWhisperer). Field notes
on the infrastructure under LLM traffic — Dynamo, llm-d, vLLM, SGLang, prompt caching,
and gateway telemetry.

Built with Jekyll, served by GitHub Pages at <https://infrawhisperer.github.io>.

## Local preview

System Ruby on macOS is usually too old for current Jekyll. Easiest path is a container:

```sh
docker run --rm -p 4000:4000 -v "$PWD":/site -w /site ruby:3.3 \
  bash -c "gem install bundler -N && bundle install && bundle exec jekyll serve --host 0.0.0.0"
```

Then open <http://localhost:4000>.

Native (if you have Ruby 3.x):

```sh
bundle install
bundle exec jekyll serve
```

## Writing a post

Drop a Markdown file in `_posts/` named `YYYY-MM-DD-slug.md` with front matter:

```yaml
---
title: "Your title"
date: 2026-07-14
topic: dynamo        # optional, shown in the byline
reading_time: 8 min  # optional
---
```

## Structure

- `_layouts/` — `default`, `home` (hero + trace), `post`
- `_includes/` — `head` (SEO + Person JSON-LD), `header`, `footer`, `trace-hero` (the signature)
- `assets/css/main.css` — the whole theme (light + dark, one accent, trace-viz palette)
- `index.html` — deep-dive cards + the gateway series
- `about.md`, `writing.html`
