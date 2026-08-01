"""
AI Invoice & Payment Assistant
--------------------------------
A Streamlit web app that lets freelancers/SMEs upload PDF invoices,
uses Google Gemini (gemini-1.5-flash) to extract structured invoice data,
stores it in SQLite, tracks paid/unpaid/overdue status on a dashboard, and
drafts polite payment-reminder emails with one click.

Author: IIT Patna Generative AI Capstone Sprint
"""

import os
import io
import json
import sqlite3
from datetime import datetime, date

import pandas as pd
import streamlit as st
import pdfplumber
import google.generativeai as genai


# =========================================================
# CONFIG
# =========================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invoices.db")
MODEL_NAME = "gemini-1.5-flash"

st.set_page_config(
    page_title="AI Invoice & Payment Assistant",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DATABASE LAYER
# =========================================================
def get_connection():
    """Return a SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the invoices table if it does not already exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT,
            client_name TEXT,
            invoice_number TEXT,
            invoice_date TEXT,
            due_date TEXT,
            total_amount REAL,
            currency TEXT,
            items TEXT,
            status TEXT DEFAULT 'Unpaid',
            source_file TEXT,
            last_email_draft TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_invoice_to_db(data: dict, source_file: str):
    """Insert a newly extracted invoice record into the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO invoices
            (vendor_name, client_name, invoice_number, invoice_date, due_date,
             total_amount, currency, items, status, source_file, last_email_draft, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("vendor_name", "Unknown Vendor"),
            data.get("client_name", "Unknown Client"),
            data.get("invoice_number", "N/A"),
            data.get("invoice_date", ""),
            data.get("due_date", ""),
            float(data.get("total_amount") or 0),
            data.get("currency", "INR"),
            json.dumps(data.get("items", [])),
            "Unpaid",
            source_file,
            "",
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def get_all_invoices() -> pd.DataFrame:
    """Return all invoices as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM invoices ORDER BY id DESC", conn)
    conn.close()
    return df


def update_invoice_status(invoice_id: int, new_status: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE invoices SET status = ? WHERE id = ?", (new_status, invoice_id))
    conn.commit()
    conn.close()


def update_email_draft(invoice_id: int, draft_text: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE invoices SET last_email_draft = ? WHERE id = ?", (draft_text, invoice_id))
    conn.commit()
    conn.close()


def delete_invoice(invoice_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM invoices WHERE id = ?", (invoice_id,))
    conn.commit()
    conn.close()


# =========================================================
# GEMINI CLIENT
# =========================================================
def get_gemini_api_key() -> str:
    """Resolve the Gemini API key from session-state, environment, or Streamlit secrets."""
    api_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        raise ValueError("No Google Gemini API key found. Please enter it in the sidebar.")
    return api_key


def get_gemini_model(system_instruction: str) -> "genai.GenerativeModel":
    """Configure the google-generativeai SDK and return a ready-to-use GenerativeModel."""
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
    )


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file using pdfplumber."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


# =========================================================
# AI: STRUCTURED INVOICE EXTRACTION (JSON MODE)
# =========================================================
EXTRACTION_SYSTEM_PROMPT = """You are an expert accounting assistant that extracts structured
data from raw invoice text. You always respond with STRICT, VALID JSON only, matching this
exact schema, and nothing else (no markdown, no commentary):

{
  "vendor_name": string,        // the company/person issuing the invoice (who is being paid)
  "client_name": string,        // the company/person being billed (who owes money)
  "invoice_number": string,     // invoice/reference number, "N/A" if not found
  "invoice_date": string,       // format YYYY-MM-DD, best guess if ambiguous
  "due_date": string,           // format YYYY-MM-DD. If not explicitly stated, estimate as
                                 // invoice_date + 30 days
  "total_amount": number,       // the final total/grand total as a plain number, no currency symbols
  "currency": string,           // 3-letter currency code, e.g. INR, USD, EUR. Default "INR" if unclear
  "items": [
    {
      "description": string,
      "quantity": number,
      "unit_price": number,
      "amount": number
    }
  ]
}

Rules:
- Dates MUST be in YYYY-MM-DD format.
- If a field truly cannot be found, use a sensible default ("N/A" for strings, 0 for numbers, [] for items).
- Never invent monetary values that are not implied by the text; only estimate due_date if missing.
- Output JSON only.
"""


def extract_invoice_data(pdf_text: str) -> dict:
    """Call Gemini in JSON mode to turn raw PDF text into structured invoice fields."""
    model = get_gemini_model(system_instruction=EXTRACTION_SYSTEM_PROMPT)
    response = model.generate_content(
        f"Extract invoice details from this text:\n\n{pdf_text[:12000]}",
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    raw = response.text
    data = json.loads(raw)
    return data


# =========================================================
# AI: REMINDER EMAIL GENERATION
# =========================================================
EMAIL_SYSTEM_PROMPT = """You are a professional, courteous billing assistant who writes short,
polite, firm-but-friendly payment reminder emails on behalf of a vendor to their client for an
overdue or upcoming invoice. Keep the tone respectful and professional, never aggressive.
Always include: a clear subject line, invoice number, amount due, due date, and a polite call
to action. Sign off as "The [vendor_name] Billing Team". Keep the body under 160 words.
Respond in this exact JSON schema only:
{
  "subject": string,
  "body": string
}
"""


def generate_reminder_email(invoice: dict) -> dict:
    """Call Gemini to draft a payment reminder email for a given invoice record."""
    days_overdue = ""
    try:
        due = datetime.strptime(invoice["due_date"], "%Y-%m-%d").date()
        delta = (date.today() - due).days
        if delta > 0:
            days_overdue = f"It is currently {delta} day(s) overdue."
        elif delta == 0:
            days_overdue = "It is due today."
        else:
            days_overdue = f"It is due in {abs(delta)} day(s)."
    except Exception:
        days_overdue = ""

    user_prompt = f"""
    Vendor (sender): {invoice['vendor_name']}
    Client (recipient): {invoice['client_name']}
    Invoice Number: {invoice['invoice_number']}
    Invoice Date: {invoice['invoice_date']}
    Due Date: {invoice['due_date']}
    Total Amount: {invoice['currency']} {invoice['total_amount']}
    Status: {invoice['status']}
    {days_overdue}

    Write a polite payment reminder email from the vendor to the client about this invoice.
    """

    model = get_gemini_model(system_instruction=EMAIL_SYSTEM_PROMPT)
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    raw = response.text
    return json.loads(raw)


# =========================================================
# HELPERS
# =========================================================
def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_billed": 0, "unpaid_amount": 0, "overdue_count": 0, "total_invoices": 0}

    total_billed = df["total_amount"].sum()
    unpaid_df = df[df["status"] == "Unpaid"]
    unpaid_amount = unpaid_df["total_amount"].sum()

    today = date.today()

    def is_overdue(row):
        if row["status"] == "Paid":
            return False
        try:
            due = datetime.strptime(row["due_date"], "%Y-%m-%d").date()
            return due < today
        except Exception:
            return False

    overdue_count = df.apply(is_overdue, axis=1).sum() if not df.empty else 0

    return {
        "total_billed": total_billed,
        "unpaid_amount": unpaid_amount,
        "overdue_count": int(overdue_count),
        "total_invoices": len(df),
    }


def status_badge(status: str, due_date_str: str) -> str:
    today = date.today()
    try:
        due = datetime.strptime(due_date_str, "%Y-%m-%d").date()
    except Exception:
        due = None

    if status == "Paid":
        return "🟢 Paid"
    if due is not None and due < today:
        return "🔴 Overdue"
    if status == "Pending":
        return "🟡 Pending"
    return "🟠 Unpaid"


CUSTOM_CSS = """
<style>
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
}
.invoice-card {
    background-color: #ffffff10;
    border: 1px solid #4b4b4b33;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.small-label {
    font-size: 0.75rem;
    color: #9a9a9a;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
</style>
"""


# =========================================================
# APP INIT
# =========================================================
init_db()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "gemini_api_key" not in st.session_state:
    default_key = os.environ.get("GEMINI_API_KEY", "")
    if not default_key:
        try:
            default_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            default_key = ""
    st.session_state["gemini_api_key"] = default_key


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧾 InvoicePilot AI")
    st.caption("AI Invoice & Payment Assistant")
    st.divider()

    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state["gemini_api_key"],
        type="password",
        help=(
            "Your key is kept only in this browser session. Get a free key at "
            "aistudio.google.com/app/apikey"
        ),
    )
    st.session_state["gemini_api_key"] = api_key_input

    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "⬆️ Upload Invoice", "📁 All Invoices", "🔌 Automation Guide"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Built for the IIT Patna Generative AI Capstone Sprint")
    st.caption(f"Model: `{MODEL_NAME}`")


# =========================================================
# PAGE: DASHBOARD
# =========================================================
if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.caption("A real-time overview of everything you're owed.")

    df = get_all_invoices()
    metrics = compute_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Billed", f"₹{metrics['total_billed']:,.2f}")
    col2.metric("Unpaid Amount", f"₹{metrics['unpaid_amount']:,.2f}")
    col3.metric("Overdue Invoices", metrics["overdue_count"])
    col4.metric("Total Invoices", metrics["total_invoices"])

    st.divider()

    if df.empty:
        st.info("No invoices yet. Head to **Upload Invoice** to add your first one.")
    else:
        st.subheader("Recent Invoices")
        display_df = df.copy()
        display_df["Status"] = display_df.apply(
            lambda r: status_badge(r["status"], r["due_date"]), axis=1
        )
        display_df = display_df[
            ["id", "vendor_name", "client_name", "invoice_number", "invoice_date",
             "due_date", "total_amount", "currency", "Status"]
        ].rename(columns={
            "id": "ID",
            "vendor_name": "Vendor",
            "client_name": "Client",
            "invoice_number": "Invoice #",
            "invoice_date": "Invoice Date",
            "due_date": "Due Date",
            "total_amount": "Amount",
            "currency": "Currency",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Unpaid Amount by Client")
        unpaid_only = df[df["status"] != "Paid"]
        if not unpaid_only.empty:
            chart_data = unpaid_only.groupby("client_name")["total_amount"].sum().sort_values(ascending=False)
            st.bar_chart(chart_data)
        else:
            st.success("🎉 No unpaid invoices right now!")


# =========================================================
# PAGE: UPLOAD INVOICE
# =========================================================
elif page == "⬆️ Upload Invoice":
    st.title("⬆️ Upload Invoice")
    st.caption("Upload a PDF invoice — AI will extract the details automatically.")

    uploaded_file = st.file_uploader("Choose a PDF invoice", type=["pdf"])

    if uploaded_file is not None:
        st.success(f"File uploaded: **{uploaded_file.name}**")

        if st.button("🚀 Extract & Save Invoice", type="primary", use_container_width=True):
            if not st.session_state["gemini_api_key"]:
                st.error("Please enter your Google Gemini API key in the sidebar first.")
            else:
                try:
                    with st.spinner("Reading PDF..."):
                        raw_text = extract_text_from_pdf(uploaded_file)

                    if not raw_text:
                        st.error(
                            "Couldn't extract any text from this PDF. It may be a scanned "
                            "image without a text layer. Try an OCR'd PDF instead."
                        )
                    else:
                        with st.spinner("Extracting structured data with AI..."):
                            extracted = extract_invoice_data(raw_text)

                        st.subheader("✅ Extracted Details")
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"**Vendor:** {extracted.get('vendor_name')}")
                        c1.markdown(f"**Client:** {extracted.get('client_name')}")
                        c2.markdown(f"**Invoice #:** {extracted.get('invoice_number')}")
                        c2.markdown(f"**Invoice Date:** {extracted.get('invoice_date')}")
                        c3.markdown(f"**Due Date:** {extracted.get('due_date')}")
                        c3.markdown(
                            f"**Total:** {extracted.get('currency')} "
                            f"{extracted.get('total_amount')}"
                        )

                        items = extracted.get("items", [])
                        if items:
                            st.markdown("**Line Items**")
                            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

                        save_invoice_to_db(extracted, uploaded_file.name)
                        st.success("Invoice saved to database! Go to **All Invoices** to view and manage it.")
                        st.balloons()

                except ValueError as ve:
                    st.error(str(ve))
                except json.JSONDecodeError:
                    st.error("AI returned an unexpected format. Please try uploading again.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    st.divider()
    with st.expander("💡 Tips for best extraction results"):
        st.markdown(
            """
            - Use PDFs that contain selectable text (not pure scanned images).
            - Standard invoice formats (vendor, client, line items, total, due date) work best.
            - If a due date isn't explicitly present, the AI will estimate 30 days from the invoice date.
            - You can always edit the payment status later from the **All Invoices** page.
            """
        )


# =========================================================
# PAGE: ALL INVOICES
# =========================================================
elif page == "📁 All Invoices":
    st.title("📁 All Invoices")
    st.caption("Manage payment status and generate reminder emails.")

    df = get_all_invoices()

    if df.empty:
        st.info("No invoices yet. Head to **Upload Invoice** to add your first one.")
    else:
        f1, f2 = st.columns([2, 1])
        search_term = f1.text_input("🔍 Search by client or vendor", "")
        status_filter = f2.selectbox("Filter by status", ["All", "Unpaid", "Pending", "Paid"])

        filtered_df = df.copy()
        if search_term:
            mask = (
                filtered_df["client_name"].str.contains(search_term, case=False, na=False)
                | filtered_df["vendor_name"].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]

        st.caption(f"Showing {len(filtered_df)} of {len(df)} invoices")
        st.divider()

        for _, row in filtered_df.iterrows():
            badge = status_badge(row["status"], row["due_date"])
            with st.container(border=True):
                top_l, top_r = st.columns([3, 1])
                with top_l:
                    st.markdown(f"### {row['vendor_name']} → {row['client_name']}")
                    st.caption(f"Invoice #{row['invoice_number']}  •  Source: {row['source_file']}")
                with top_r:
                    st.markdown(f"#### {badge}")

                d1, d2, d3, d4 = st.columns(4)
                d1.markdown(f"**Invoice Date**\n\n{row['invoice_date']}")
                d2.markdown(f"**Due Date**\n\n{row['due_date']}")
                d3.markdown(f"**Amount**\n\n{row['currency']} {row['total_amount']:,.2f}")

                new_status = d4.selectbox(
                    "Status",
                    ["Unpaid", "Pending", "Paid"],
                    index=["Unpaid", "Pending", "Paid"].index(row["status"])
                    if row["status"] in ["Unpaid", "Pending", "Paid"] else 0,
                    key=f"status_{row['id']}",
                )
                if new_status != row["status"]:
                    update_invoice_status(row["id"], new_status)
                    st.rerun()

                try:
                    items = json.loads(row["items"]) if row["items"] else []
                except Exception:
                    items = []
                if items:
                    with st.expander("View line items"):
                        st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if row["status"] != "Paid":
                        generate_clicked = st.button(
                            "✉️ Generate Reminder Email",
                            key=f"email_btn_{row['id']}",
                            use_container_width=True,
                        )
                    else:
                        generate_clicked = False
                        st.success("Fully paid — no reminder needed ✅")
                with btn_col2:
                    if st.button("🗑️ Delete Invoice", key=f"delete_btn_{row['id']}", use_container_width=True):
                        delete_invoice(row["id"])
                        st.rerun()

                if generate_clicked:
                    if not st.session_state["gemini_api_key"]:
                        st.error("Please enter your Google Gemini API key in the sidebar first.")
                    else:
                        try:
                            with st.spinner("Drafting a polite reminder email..."):
                                email = generate_reminder_email(row)
                            draft_text = f"Subject: {email['subject']}\n\n{email['body']}"
                            update_email_draft(row["id"], draft_text)
                            st.session_state[f"draft_{row['id']}"] = draft_text
                        except ValueError as ve:
                            st.error(str(ve))
                        except Exception as e:
                            st.error(f"Something went wrong generating the email: {e}")

                draft_key = f"draft_{row['id']}"
                existing_draft = st.session_state.get(draft_key) or row["last_email_draft"]
                if existing_draft:
                    st.text_area(
                        "📧 Reminder Email Draft (copy and send manually, or wire up via SMTP/Make.com)",
                        value=existing_draft,
                        height=180,
                        key=f"draft_area_{row['id']}",
                    )


# =========================================================
# PAGE: AUTOMATION GUIDE
# =========================================================
elif page == "🔌 Automation Guide":
    st.title("🔌 Automation Guide")
    st.caption("Turn this app into a fully automated reminder pipeline using Make.com and webhooks.")

    st.markdown(
        """
### Why automate?
Right now, generating a reminder email is one click — but *sending* it is manual.
You can close that gap with a no-code automation layer like **Make.com** (or Zapier),
triggered by a webhook this app calls whenever an invoice becomes overdue.

### High-level flow
1. This Streamlit app (or a scheduled script) checks the SQLite database daily for
   invoices where `status = 'Unpaid'` and `due_date < today`.
2. For each overdue invoice, the app calls `generate_reminder_email()` to get a
   subject + body.
3. The app sends an HTTP POST request to a **Make.com Webhook URL** with the
   invoice + email payload.
4. Make.com's scenario receives the webhook, then sends the actual email via
   Gmail/Outlook/SMTP modules, and can also log it to Google Sheets or Slack.

### Example webhook payload
```json
{
  "invoice_id": 12,
  "vendor_name": "Acme Design Studio",
  "client_name": "Bright Retail Pvt Ltd",
  "client_email": "accounts@brightretail.com",
  "amount_due": 45000,
  "currency": "INR",
  "due_date": "2026-07-15",
  "subject": "Friendly Reminder: Invoice #INV-1042 Payment Due",
  "body": "Dear Bright Retail team, this is a gentle reminder that Invoice #INV-1042..."
}
```

### Example Python snippet to trigger the webhook
```python
import requests

MAKE_WEBHOOK_URL = "https://hook.make.com/your-unique-webhook-id"

def send_to_make(payload: dict):
    response = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.status_code
```

### Setting it up in Make.com
1. Create a free Make.com account.
2. Create a new Scenario → add a **Webhooks → Custom Webhook** trigger module.
   Copy the generated URL into `MAKE_WEBHOOK_URL` above.
3. Add a **Gmail/Outlook → Send an Email** module after the webhook trigger.
   Map `subject`, `body`, and `client_email` fields from the webhook payload.
4. (Optional) Add a **Google Sheets → Add a Row** module to log every reminder sent.
5. Turn the scenario ON. Every time this app posts to the webhook, an email goes out
   automatically and gets logged.

### Scheduling the daily check
- Locally: use `cron` (Linux/Mac) or Task Scheduler (Windows) to run a small Python
  script daily that queries overdue invoices and posts each one to the webhook.
- On Streamlit Community Cloud: use an external scheduler (e.g. GitHub Actions
  `schedule` cron job, or a free cron service like cron-job.org) to hit a small
  FastAPI/Flask endpoint you deploy alongside the app, which runs the same check.
        """
    )
