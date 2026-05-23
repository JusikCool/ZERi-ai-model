TICKERS_BY_SECTOR: dict[str, list[str]] = {
    "semiconductor": ["NVDA", "AMD", "INTC", "QCOM", "MU", "AVGO"],
    "bigtech": ["AAPL", "GOOGL", "META", "AMZN", "MSFT", "NFLX"],
    "software": ["ORCL", "CRM", "ADBE", "NOW", "SNOW", "UBER"],
    "financial": ["JPM", "GS", "BAC", "MS", "V", "MA"],
    "healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH", "CVS"],
    "biotech": ["AMGN", "GILD", "BIIB", "REGN", "MRNA", "VRTX"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
    "consumer": ["WMT", "TGT", "COST", "HD", "NKE", "SBUX"],
}

SECTOR_IDS: list[str] = list(TICKERS_BY_SECTOR.keys())

SECTOR_MAP: dict[str, str] = {
    ticker: sector
    for sector, tickers in TICKERS_BY_SECTOR.items()
    for ticker in tickers
}


def get_sector(ticker: str) -> str:
    if ticker not in SECTOR_MAP:
        raise KeyError(f"Unknown ticker: {ticker}")
    return SECTOR_MAP[ticker]


def get_tickers(sector_id: str) -> list[str]:
    if sector_id not in TICKERS_BY_SECTOR:
        raise KeyError(f"Unknown sector: {sector_id}")
    return list(TICKERS_BY_SECTOR[sector_id])
