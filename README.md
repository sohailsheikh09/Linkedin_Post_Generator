# LinkedIn Post Generator — AI + Human-in-the-Loop

An AI-powered Streamlit app that drafts LinkedIn posts using **Groq LLM**, lets you review and revise them with a **Human-in-the-Loop (HITL)** workflow powered by **LangGraph**, and then publishes the approved post directly to **LinkedIn** via the official REST API.

---

## What It Does

```
START → Generate Draft → [HUMAN REVIEW] ──── Approved ──→ Post to LinkedIn → END
                               ↑                    │
                               └──── Revise ←───────┘
```

1. You fill in: topic, key points, tone, and target audience
2. The AI generates a LinkedIn post draft
3. You review it — either give feedback for a revision or approve it
4. On approval it posts directly to your LinkedIn profile

---

## Screenshots

| Input Form | Review Draft | Published |
|------------|-------------|-----------|
| Fill topic & details | Read AI draft, approve or revise | Success + Post ID |

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| pip | latest |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sohailsheikh09/Linkedin_Post_Generator.git
cd Linkedin_Post_Generator
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
LINKEDIN_ACCESS_TOKEN=your_linkedin_access_token_here
LINKEDIN_AUTHOR_URN=urn:li:person:your_person_id_here
```

---

## How to Get Each API Key / Token

### GROQ_API_KEY (Free)

Groq provides a fast, free LLM API for models like LLaMA and Mistral.

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up or log in with your Google / GitHub account
3. Click **"API Keys"** in the left sidebar
4. Click **"Create API Key"**
5. Give it a name (e.g. `linkedin-bot`) and click **Create**
6. Copy the key — it starts with `gsk_...`
7. Paste it in your `.env` as `GROQ_API_KEY=gsk_...`

> Free tier includes generous token limits — no credit card required.

---

### LINKEDIN_ACCESS_TOKEN

LinkedIn uses OAuth 2.0. Follow these steps carefully:

#### Step 1 — Create a LinkedIn App

1. Go to [https://www.linkedin.com/developers/apps](https://www.linkedin.com/developers/apps)
2. Click **"Create app"**
3. Fill in:
   - **App name**: e.g. `LinkedIn Post Generator`
   - **LinkedIn Page**: Select your personal/company page (or create one — it's free)
   - **App logo**: Upload any image
4. Check the **Legal Agreement** box and click **Create app**

#### Step 2 — Enable Required Permissions (Products)

1. On your app page, go to the **"Products"** tab
2. Request access to **"Share on LinkedIn"** — click **"Select"**
3. Also request **"Sign In with LinkedIn using OpenID Connect"**
4. These are usually approved instantly

#### Step 3 — Get Your OAuth 2.0 Credentials

1. Go to the **"Auth"** tab of your app
2. Note down:
   - **Client ID**
   - **Client Secret**
3. Under **"OAuth 2.0 settings"**, add a redirect URL:
   ```
   https://oauth.pstmn.io/v1/callback
   ```
   (This is Postman's OAuth redirect — easiest way for beginners)

#### Step 4 — Generate Access Token via Postman (Easiest Method)

1. Download and open [Postman](https://www.postman.com/downloads/)
2. Create a new **GET** request (URL doesn't matter)
3. Go to the **Authorization** tab
4. Set **Type** to `OAuth 2.0`
5. Click **"Get New Access Token"** and fill in:

   | Field | Value |
   |-------|-------|
   | Token Name | `linkedin-token` |
   | Grant Type | `Authorization Code` |
   | Callback URL | `https://oauth.pstmn.io/v1/callback` |
   | Auth URL | `https://www.linkedin.com/oauth/v2/authorization` |
   | Access Token URL | `https://www.linkedin.com/oauth/v2/accessToken` |
   | Client ID | *(your app's Client ID)* |
   | Client Secret | *(your app's Client Secret)* |
   | Scope | `w_member_social openid profile email` |
   | State | `random123` |

6. Click **"Request Token"** — your browser opens and you log in to LinkedIn
7. After login, Postman shows the token — copy the **Access Token** value
8. Paste it in your `.env` as `LINKEDIN_ACCESS_TOKEN=AQV...`

> **Token expiry**: LinkedIn tokens expire in **60 days**. Repeat this step to refresh it.

#### Alternative — LinkedIn Token Generator Tool

LinkedIn provides an official token generator for testing:
1. Go to [https://www.linkedin.com/developers/tools/oauth/token-generator](https://www.linkedin.com/developers/tools/oauth/token-generator)
2. Select your app
3. Check `w_member_social` scope
4. Click **"Request access token"**
5. Authorize and copy the token

---

### LINKEDIN_AUTHOR_URN

This is your LinkedIn **person ID** — it identifies who the post is from.

#### Method 1 — From LinkedIn API (Recommended)

After getting your access token, run this in your terminal:

```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "LinkedIn-Version: 202511" \
     https://api.linkedin.com/v2/userinfo
```

Look for the `sub` field in the response — that is your person ID.

Your URN will be: `urn:li:person:YOUR_SUB_VALUE`

#### Method 2 — From Postman

1. Create a GET request to: `https://api.linkedin.com/v2/userinfo`
2. Add header: `Authorization: Bearer YOUR_ACCESS_TOKEN`
3. Add header: `LinkedIn-Version: 202511`
4. Send — copy the `sub` field from the JSON response
5. Your `LINKEDIN_AUTHOR_URN` = `urn:li:person:<sub_value>`

**Example `.env`:**

```env
GROQ_API_KEY=gsk_abc123...
LINKEDIN_ACCESS_TOKEN=AQVtYWFi...
LINKEDIN_AUTHOR_URN=urn:li:person:yyotj6wNOM
```

---

## Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Project Structure

```
Linkedin_Post_Generator/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # Your API keys (NOT committed to git)
├── .gitignore          # Ignores .env, __pycache__, etc.
└── README.md           # This file
```

---

## How the Code Works

### LangGraph Workflow (HITL Pattern)

```python
# Graph is built once and stored in Streamlit session_state
graph = build_graph()   # InMemorySaver enables pause/resume

# First invocation — runs until interrupt
graph.invoke(initial_state, config=config)

# Resume with human answer
graph.invoke(Command(resume={interrupt_id: feedback}), config=config)
```

The `interrupt()` call inside `ask_for_feedback` **pauses** the graph and saves state via `InMemorySaver`. When the user submits their review in the browser, the graph **resumes** from exactly where it paused.

### Key Nodes

| Node | Purpose |
|------|---------|
| `generate_draft` | Calls Groq LLM to write initial post |
| `ask_for_feedback` | Pauses with `interrupt()` — waits for human |
| `decide_next` | Routes to `revise_draft` or `post_to_linkedin` |
| `revise_draft` | Rewrites draft based on human feedback |
| `post_to_linkedin_real` | Calls LinkedIn REST API to publish |

### Streamlit Session State Flow

```
phase = "input"    →  render input form
phase = "reviewing" →  show draft, collect feedback
phase = "done"      →  show result (success / error)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY not found` | Ensure `.env` is in the same folder as `app.py` |
| `Status 401` from LinkedIn | Token expired — regenerate it (valid 60 days) |
| `Status 403` from LinkedIn | App missing `w_member_social` scope — re-check Products tab |
| `Missing LINKEDIN_AUTHOR_URN` | Run the `/v2/userinfo` API call above to get your sub |
| Draft not generating | Check Groq API key and internet connection |
| App crashes on rerun | Clear browser cache or restart: `streamlit run app.py` |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `langgraph` | Agentic HITL workflow |
| `langchain-openai` | LLM client (Groq via OpenAI-compatible API) |
| `langchain-core` | Message types |
| `python-dotenv` | Load `.env` file |
| `requests` | LinkedIn API HTTP calls |

---

## Security Notes

- **Never commit your `.env` file** — it's in `.gitignore`
- LinkedIn tokens expire in 60 days — refresh them regularly
- Do not share your `LINKEDIN_ACCESS_TOKEN` publicly

---

## License

MIT License — free to use and modify.

---

## Author

**Sohail Sheikh** — [github.com/sohailsheikh09](https://github.com/sohailsheikh09)
