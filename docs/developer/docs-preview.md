# Local documentation preview

The public site is <https://devaiexp.github.io/ddo-examples/>.
The `docs-pages.yml` workflow builds with MkDocs 1.6.1 in strict mode and
publishes through GitHub Pages after documentation changes reach `main`.
Maintainers can also dispatch it manually. Only the generated static site is
uploaded; model packages, weights, environments and private records are not
needed. GitHub Pages must use GitHub Actions as its publishing source.

The commands below remain a local preview and do not publish anything.

From the examples repository root, run this in PowerShell or Bash:

```text
uvx --from mkdocs==1.6.1 mkdocs serve --config-file mkdocs.yml --dev-addr 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/> in Chrome or Edge. Use the browser's page-translation
menu to read in Portuguese or another language. Browser translation may send page
text to the translation provider; it does not modify the English Markdown sources.
If automatic translation is not offered, select English as the page's source
language in the browser translation menu. Code snippets should still be copied
from the original English page or source Markdown.

uvx installs/runs the pinned MkDocs tool in its own cache, not the inference venv;
the first invocation needs internet access to obtain the tool and dependencies.
No Torch, CUDA, model, custom-block clone or pipeline execution is needed.
Saving documentation reloads the local preview. Stop with Ctrl+C. If port 8000 is
occupied, use `--dev-addr 127.0.0.1:8001` instead. Do not bind to `0.0.0.0` for
private local review.

The preview includes only `docs/` and the explicit public README and agent guidance.
Its small MkDocs hook preserves repository-relative
links using virtual files; original files are neither copied nor rewritten.
`.internal/`, environments, outputs, weights and ignored custom-block clones are
not served. HTML output is generated under ignored `.internal/docs-preview/site/`.
Nothing is deployed or uploaded; this does not enable GitHub Pages.

To verify rendering without starting a server:

```text
uvx --from mkdocs==1.6.1 mkdocs build --config-file mkdocs.yml --strict
```

MkDocs reference: <https://www.mkdocs.org/getting-started/>.

## Architecture diagrams

Mermaid fences render in the browser using the pinned Mermaid 12.2.0 module
from jsDelivr. Diagram rendering needs access to that CDN; if it is unavailable,
the source remains readable. No inference dependency is installed. The browser
fetches JavaScript; the preview does not upload Markdown to a diagram service.
