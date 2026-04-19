"""SEC EDGAR 8-K filing fetcher.

8-K filings are "material event" disclosures — the most important signal
for the LLM agent. They cover:
  - Earnings results (Item 2.02)
  - Bankruptcy/financial distress (Item 1.03)
  - Legal proceedings (Item 8.01)
  - Executive departures (Item 5.02)
  - FDA decisions, regulatory actions (Item 8.01)
  - Mergers & acquisitions (Item 1.01)
  - Guidance changes (Item 7.01 / 8.01)

No API key required — SEC EDGAR is free and public.
Rate limit: 10 requests/second (we stay well under that).

Usage:
    from llm_agent.fetchers.edgar import fetch_8k_filings
    filings = fetch_8k_filings("AAPL", days_back=14)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

EDGAR_BASE    = "https://data.sec.gov/submissions"
TICKER_MAP    = "https://www.sec.gov/files/company_tickers.json"
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN  = 0.15   # stay under 10 req/sec SEC limit

# SEC requires a descriptive User-Agent
HEADERS = {
    "User-Agent": "quant-research-bot contact@example.com",
    "Accept"    : "application/json",
}

# 8-K item codes that are most relevant for trading decisions
HIGH_IMPACT_ITEMS = {
    "1.01": "Material agreement",
    "1.02": "Agreement termination",
    "1.03": "Bankruptcy / receivership",
    "2.02": "Results of operations (earnings)",
    "2.04": "Mine safety — triggering event",
    "3.01": "Delisting notice",
    "4.01": "Auditor change",
    "4.02": "Accounting restatement",
    "5.02": "Executive departure / appointment",
    "7.01": "Regulation FD — guidance",
    "8.01": "Other material events",
    "9.01": "Financial statements / exhibits",
}


class EdgarError(Exception):
    pass


def _get_cik(symbol: str) -> Optional[str]:
    """Look up SEC CIK number for a ticker symbol."""
    try:
        time.sleep(SLEEP_BETWEEN)
        resp = requests.get(TICKER_MAP, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        symbol_upper = symbol.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == symbol_upper:
                # CIK must be zero-padded to 10 digits
                return str(entry["cik_str"]).zfill(10)
        return None
    except requests.RequestException as e:
        raise EdgarError(f"Failed to look up CIK for {symbol}: {e}") from e


def fetch_8k_filings(
    symbol: str,
    days_back: int = 14,
    max_filings: int = 5,
) -> list[dict]:
    """Fetch recent 8-K filings for a ticker from SEC EDGAR.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL"
    days_back : int
        How many calendar days back to search (default 14)
    max_filings : int
        Maximum number of 8-K filings to return

    Returns
    -------
    list[dict]
        Each dict contains:
          symbol       : str
          filed_date   : str (YYYY-MM-DD)
          form_type    : str (always "8-K" or "8-K/A")
          description  : str (filing description if available)
          items        : list[str] (item codes mentioned)
          impact_label : str (human-readable impact summary)
          filing_url   : str (direct link to filing)
    """
    cik = _get_cik(symbol)
    if not cik:
        return []   # symbol not found in EDGAR (foreign ADRs, ETFs, etc.)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).date()

    try:
        time.sleep(SLEEP_BETWEEN)
        url  = f"{EDGAR_BASE}/CIK{cik}.json"
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        sub  = resp.json()
    except requests.RequestException as e:
        raise EdgarError(f"Failed to fetch EDGAR submissions for {symbol}: {e}") from e

    recent = sub.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    filed_dates= recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    descriptions = recent.get("primaryDocument", [])

    results = []
    for form, filed, accession, desc in zip(forms, filed_dates, accessions, descriptions):
        if form not in ("8-K", "8-K/A"):
            continue
        try:
            filed_date = datetime.strptime(filed, "%Y-%m-%d").date()
        except ValueError:
            continue
        if filed_date < cutoff:
            break   # filings are in reverse chronological order, done

        # Build filing URL
        acc_clean   = accession.replace("-", "")
        filing_url  = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_clean}/{desc}"
        )

        results.append({
            "symbol"      : symbol.upper(),
            "filed_date"  : filed,
            "form_type"   : form,
            "accession"   : accession,
            "description" : desc,
            "items"       : [],          # populated below if item data available
            "impact_label": "8-K filing",
            "filing_url"  : filing_url,
        })

        if len(results) >= max_filings:
            break

    return results


def format_for_prompt(filings: list[dict]) -> str:
    """Format 8-K filings into a clean string for Claude prompts."""
    if not filings:
        return "No recent SEC 8-K filings found."

    lines = []
    for f in filings:
        lines.append(
            f"• [{f['filed_date']}] {f['form_type']} — {f['description']}"
        )
    return "\n".join(lines)


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    days   = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"Fetching 8-K filings for {symbol} (last {days} days) …")
    try:
        filings = fetch_8k_filings(symbol, days_back=days)
        print(f"Found {len(filings)} 8-K filings\n")
        print(format_for_prompt(filings))
        if filings:
            print(f"\nFirst filing URL: {filings[0]['filing_url']}")
    except EdgarError as e:
        print(f"Error: {e}")
