import sqlite3
import os
from datetime import datetime

DB_PATH = os.environ.get(
    "TRADER_REGISTRY_DB_PATH",
    "/home/maxim/.openclaw/workspace/projects/dings-trader/data/trader.sqlite",
)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Models table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY,
        version TEXT,
        start_time TIMESTAMP,
        initial_capital REAL,
        current_equity REAL,
        status TEXT
    )
    """)
    
    # Trades table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id TEXT,
        symbol TEXT,
        side TEXT,
        entry_price REAL,
        exit_price REAL,
        size REAL,
        entry_time TIMESTAMP,
        exit_time TIMESTAMP,
        profit_pct REAL,
        status TEXT,
        FOREIGN KEY(model_id) REFERENCES models(id)
    )
    """)

    # Model packages registry (Forecast + PPO artifacts)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_packages (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        forecast_rel_path TEXT NOT NULL,
        ppo_rel_path TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        warmup_required INTEGER NOT NULL DEFAULT 1,
        warmup_status TEXT NOT NULL DEFAULT 'PENDING',
        warmup_completed_at TIMESTAMP,
        warmup_error TEXT,
        feature_mask TEXT
    )
    """)

    # Migration: Add warmup_error if missing
    try:
        cursor.execute("ALTER TABLE model_packages ADD COLUMN warmup_error TEXT")
    except sqlite3.OperationalError:
        pass

    # Migration: Add feature_mask if missing
    try:
        cursor.execute("ALTER TABLE model_packages ADD COLUMN feature_mask TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

def get_conn():
    return sqlite3.connect(DB_PATH)


def ensure_model(model_id: str, version: str | None = None, initial_capital: float = 1100.0):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO models (id, version, start_time, initial_capital, current_equity, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (model_id, version, datetime.utcnow().isoformat(), initial_capital, initial_capital, "LIVE"),
    )
    cursor.execute(
        """
        UPDATE models
        SET version = COALESCE(version, ?),
            initial_capital = COALESCE(initial_capital, ?),
            current_equity = COALESCE(current_equity, ?),
            status = COALESCE(status, ?)
        WHERE id = ?
        """,
        (version, initial_capital, initial_capital, "LIVE", model_id),
    )
    conn.commit()
    conn.close()


def get_model(model_id: str):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, version, start_time, initial_capital, current_equity, status FROM models WHERE id = ?",
        (model_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["id", "version", "start_time", "initial_capital", "current_equity", "status"]
    return dict(zip(keys, row))


def update_equity(model_id: str, equity: float):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE models SET current_equity = ? WHERE id = ?", (float(equity), model_id))
    conn.commit()
    conn.close()


def set_model_status(model_id: str, status: str):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE models SET status = ? WHERE id = ?", (status, model_id))
    conn.commit()
    conn.close()


def check_kill_switch(model_id: str, threshold: float = 200.0):
    model = get_model(model_id)
    if not model:
        return False
    if model.get("status") == "FAILED":
        return True
    equity = model.get("current_equity")
    if equity is None:
        equity = model.get("initial_capital")
    if equity is None:
        return False
    if float(equity) < threshold:
        set_model_status(model_id, "FAILED")
        return True
    return False


def get_open_position(model_id: str):
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM trades 
        WHERE model_id = ? AND status = 'OPEN'
        ORDER BY entry_time DESC LIMIT 1
        """,
        (model_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def open_position(model_id: str, symbol: str, side: str, price: float, size: float):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO trades (model_id, symbol, side, entry_price, size, entry_time, status)
        VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
        """,
        (model_id, symbol, side, float(price), float(size), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def close_position(trade_id: int, exit_price: float, profit_pct: float):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE trades 
        SET exit_price = ?, exit_time = ?, profit_pct = ?, status = 'CLOSED'
        WHERE id = ?
        """,
        (float(exit_price), datetime.utcnow().isoformat(), float(profit_pct), trade_id),
    )
    conn.commit()
    conn.close()


def get_trades(model_id: str):
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT symbol, side, entry_time, exit_time, profit_pct, status 
        FROM trades 
        WHERE model_id = ? 
        ORDER BY exit_time DESC, entry_time DESC
        """,
        (model_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        # Format for UI
        ts = r["exit_time"] if r["exit_time"] else r["entry_time"]
        try:
            # Try to format timestamp if it's ISO
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = str(ts)
            
        profit_val = r["profit_pct"]
        if profit_val is None:
            profit_str = "0.00%"
        else:
            profit_str = f"{profit_val:+.1f}%"

        results.append({
            "time": time_str,
            "side": r["side"],
            "profit": profit_str,
            "status": r["status"]
        })
    return results


def get_all_models():
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, version, status, current_equity FROM models ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]


# --- Model packages registry helpers ---

def ensure_default_model_package(
    package_id: str,
    name: str,
    forecast_rel_path: str,
    ppo_rel_path: str,
    status: str = "READY",
    warmup_required: bool = True,
):
    """Seed a known model package if it doesn't exist."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO model_packages
        (id, name, forecast_rel_path, ppo_rel_path, status, created_at, warmup_required, warmup_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            package_id,
            name,
            forecast_rel_path,
            ppo_rel_path,
            status,
            datetime.utcnow().isoformat(),
            1 if warmup_required else 0,
            "PENDING" if warmup_required else "DONE",
        ),
    )
    conn.commit()
    conn.close()


def create_model_package(
    package_id: str,
    name: str,
    forecast_rel_path: str,
    ppo_rel_path: str,
    status: str = "UPLOADED",
    warmup_required: bool = True,
):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO model_packages
        (id, name, forecast_rel_path, ppo_rel_path, status, created_at, warmup_required, warmup_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            package_id,
            name,
            forecast_rel_path,
            ppo_rel_path,
            status,
            datetime.utcnow().isoformat(),
            1 if warmup_required else 0,
            "PENDING" if warmup_required else "DONE",
        ),
    )
    conn.commit()
    conn.close()


def get_model_package(package_id: str):
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_packages WHERE id = ?", (package_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_model_packages():
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, status, created_at, warmup_required, warmup_status, warmup_completed_at
        FROM model_packages
        ORDER BY created_at DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_model_package_warmup_status(package_id: str, warmup_status: str, completed: bool = False, error_msg: str | None = None):
    conn = get_conn()
    cursor = conn.cursor()
    warmup_completed_at = datetime.utcnow().isoformat() if completed else None
    
    # Reset completed_at when restarting warmup (PENDING/RUNNING after FAILED)
    if not completed and warmup_status in ("PENDING", "RUNNING"):
        cursor.execute(
            """
            UPDATE model_packages
            SET warmup_status = ?,
                warmup_completed_at = NULL,
                warmup_error = ?
            WHERE id = ?
            """,
            (warmup_status, error_msg, package_id),
        )
    else:
        cursor.execute(
            """
            UPDATE model_packages
            SET warmup_status = ?,
                warmup_completed_at = COALESCE(?, warmup_completed_at),
                warmup_error = ?
            WHERE id = ?
            """,
            (warmup_status, warmup_completed_at, error_msg, package_id),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
