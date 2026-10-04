# 🎓 AI Academic Tutor

An interactive Streamlit academic tutor that accepts a problem as text or an
image and uses an OpenAI-compatible vision API to explain the solution
step by step.

## Features

- Upload a clear photo of a problem.
- Type a problem without an image.
- Supports mathematics, physics, chemistry, and other academic subjects.
- Uses image understanding for equations, diagrams, graphs, and labels.
- Gives structured explanations when appropriate:
  1. Problem
  2. Key concepts
  3. Step-by-step solution
  4. Final answer
  5. Check
- Reviews a student's own work and identifies the first incorrect step.
- Supports follow-up questions in the same conversation.
- Renders LaTeX mathematics in Streamlit.
- Uses an OpenAI-compatible API interface.
- xAI Grok is the default provider/model configuration.

## Project files

```text
.
├── app.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- A valid API key for an OpenAI-compatible vision model
- Internet access from the machine running Streamlit

## Run locally

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Set your API key

Linux/macOS:

```bash
export LLM_API_KEY="your-api-key"
```

Windows PowerShell:

```powershell
$env:LLM_API_KEY="your-api-key"
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## xAI configuration

The default configuration is:

```text
API base URL: https://api.x.ai/v1
Model: grok-4.7
```

You can override either value with environment variables:

```bash
export LLM_BASE_URL="https://api.x.ai/v1"
export LLM_MODEL="grok-4.7"
```

The same values can also be entered in the Streamlit sidebar.

## Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Create a Streamlit Community Cloud app and select the repository,
   branch, and `app.py`.
4. Open the app's **Settings → Secrets**.
5. Add:

```toml
LLM_API_KEY = "your-api-key"
```

Optional:

```toml
LLM_BASE_URL = "https://api.x.ai/v1"
LLM_MODEL = "grok-4.7"
```

Do **not** put an API key directly inside `app.py` or commit it to GitHub.

## Using another OpenAI-compatible provider

The application keeps the provider settings configurable. If another provider
supports OpenAI-style Chat Completions with image input, enter its compatible
base URL and vision-capable model in the sidebar.

The exact message/image format and supported parameters are provider-specific,
so a provider that is not fully OpenAI-compatible may require additional
changes.

## Image quality

For the best academic results:

- Use a sharp, well-lit image.
- Include the complete question.
- Keep equations and labels visible.
- Avoid glare and heavy shadows.
- If a diagram is important, include the entire diagram.

The application intentionally instructs the model not to guess unreadable
information.

## Security

- Never commit API keys to GitHub.
- Prefer environment variables or Streamlit Secrets.
- Treat uploaded academic images as potentially sensitive user data.
- Use an API provider whose data-handling policy is appropriate for your use.

## Troubleshooting

### "The request could not be completed"

Open the **Technical details** section shown by the application and check:

1. API key is valid.
2. API base URL is correct.
3. Model name is correct.
4. The selected model supports image input when an image is uploaded.
5. The account has sufficient API access/credits.
6. The uploaded image is a supported type and is not excessively large.

### The model says it cannot read the image

Use a clearer image with better lighting and higher resolution. Make sure all
equations, symbols, labels, and diagram features are visible.

### Text-only questions work but images fail

The selected model or provider may not support image input, or it may use a
different image-message format. Select a documented vision-capable model or
adapt `to_api_messages()` for that provider.
