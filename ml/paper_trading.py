"""Paper Trading Engine für dings-trader.

Simuliertes Trading mit virtuellem Konto:
- Startkapital: 10.000 USDT
- Max 5 Positionen gleichzeitig
- Hebel: 7x-10x konfigurierbar
- 5% Profit-Ziel pro Trade
- 48h Zeitlimit pro Trade
"""
import sqlite3
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json

DB_PATH = os.environ.get("TRADER_DB_PATH", "/home/maxim/.openclaw/workspace/projects/dings-trader/data/paper_trading.sqlite")

# Trading fees: 0.1% per trade side (entry and exit)
FEE_RATE = 0.001  # 0.1%

@dataclass
class PaperPosition:
    id: int
    model_id: str
    symbol: str
    side: str  # "Long" oder "Short"
    entry_price: float
    size_usdt: float  # Position Size in USDT
    leverage: float  # z.B. 7.0 für 7x
    take_profit_pct: float  # z.B. 5.0 für 5%
    stop_loss_pct: float  # z.B. -3.0 für -3%
    open_time: datetime
    close_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    realized_pnl_usdt: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, EXPIRED
    close_reason: Optional[str] = None  # TP, SL, EXPIRED, MANUAL, SIGNAL_FLIP

@dataclass
class PaperAccount:
    model_id: str
    model_package_id: str
    balance_usdt: float  # Verfügbares Geld
    total_equity: float  # Balance + offene Positionen
    initial_balance: float
    created_at: datetime
    updated_at: datetime
    reset_at: datetime  # When the account was last reset
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_positions: int = 5
    default_leverage: float = 7.0
    profit_target_pct: float = 5.0
    time_limit_hours: float = 48.0
    total_fees_paid: float = 0.0  # Total trading fees paid


class PaperTradingEngine:
    """Paper Trading Engine mit SQLite Backend."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialisiert die Paper-Trading Datenbank."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Paper Accounts Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_accounts (
                model_id TEXT PRIMARY KEY,
                model_package_id TEXT,
                balance_usdt REAL DEFAULT 10000.0,
                total_equity REAL DEFAULT 10000.0,
                initial_balance REAL DEFAULT 10000.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                max_positions INTEGER DEFAULT 5,
                default_leverage REAL DEFAULT 7.0,
                profit_target_pct REAL DEFAULT 5.0,
                time_limit_hours REAL DEFAULT 48.0
            )
        """)
        
        # Add model_package_id column if it doesn't exist (migration)
        try:
            cursor.execute("SELECT model_package_id FROM paper_accounts LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE paper_accounts ADD COLUMN model_package_id TEXT")
        
        # Add reset_at column if it doesn't exist (migration)
        try:
            cursor.execute("SELECT reset_at FROM paper_accounts LIMIT 1")
        except sqlite3.OperationalError:
            # SQLite doesn't support DEFAULT with CURRENT_TIMESTAMP in ALTER TABLE
            cursor.execute("ALTER TABLE paper_accounts ADD COLUMN reset_at TIMESTAMP")
            # Set default value for existing rows
            cursor.execute("UPDATE paper_accounts SET reset_at = created_at WHERE reset_at IS NULL")
        
        # Add total_fees_paid column if it doesn't exist (migration)
        try:
            cursor.execute("SELECT total_fees_paid FROM paper_accounts LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE paper_accounts ADD COLUMN total_fees_paid REAL DEFAULT 0.0")
            # Set default value for existing rows
            cursor.execute("UPDATE paper_accounts SET total_fees_paid = 0.0 WHERE total_fees_paid IS NULL")
        
        # Paper Positions Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                size_usdt REAL,
                leverage REAL,
                take_profit_pct REAL DEFAULT 5.0,
                stop_loss_pct REAL DEFAULT -3.0,
                open_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                close_time TIMESTAMP,
                exit_price REAL,
                realized_pnl_pct REAL,
                realized_pnl_usdt REAL,
                status TEXT DEFAULT 'OPEN',
                close_reason TEXT,
                FOREIGN KEY(model_id) REFERENCES paper_accounts(model_id)
            )
        """)
        
        # Trade History für detaillierte Analyse
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER,
                model_id TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                size_usdt REAL,
                leverage REAL,
                pnl_pct REAL,
                pnl_usdt REAL,
                open_time TIMESTAMP,
                close_time TIMESTAMP,
                duration_hours REAL,
                close_reason TEXT,
                FOREIGN KEY(position_id) REFERENCES paper_positions(id)
            )
        """)
        
        # Daily P&L Tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                date TEXT,
                start_equity REAL,
                end_equity REAL,
                pnl_pct REAL,
                pnl_usdt REAL,
                trades_count INTEGER DEFAULT 0,
                UNIQUE(model_id, date)
            )
        """)
        
        # Equity History Tracking (for chart)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                equity REAL,
                balance REAL,
                open_positions_count INTEGER DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_equity_history_model_time 
            ON paper_equity_history(model_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def create_account(
        self, 
        model_id: str, 
        model_package_id: Optional[str] = None,
        initial_balance: float = 10000.0,
        max_positions: int = 5,
        default_leverage: float = 7.0,
        profit_target_pct: float = 5.0,
        time_limit_hours: float = 48.0
    ) -> PaperAccount:
        """Erstellt ein neues Paper-Trading Konto."""
        # Fallback if package id not provided: slugify from model_id or use ppo_v1
        if model_package_id is None:
            if model_id.startswith("paper_"):
                model_package_id = model_id[6:]
            else:
                model_package_id = "ppo_v1"

        conn = self._get_conn()
        cursor = conn.cursor()
        
        now = datetime.utcnow()
        cursor.execute("""
            INSERT OR REPLACE INTO paper_accounts 
            (model_id, model_package_id, balance_usdt, total_equity, initial_balance, created_at, updated_at, reset_at,
             max_positions, default_leverage, profit_target_pct, time_limit_hours, total_fees_paid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, model_package_id, initial_balance, initial_balance, initial_balance, 
            now, now, now, max_positions, default_leverage, profit_target_pct, time_limit_hours, 0.0
        ))
        
        conn.commit()
        conn.close()
        
        return self.get_account(model_id)
    
    def get_account(self, model_id: str) -> Optional[PaperAccount]:
        """Holt ein Paper-Trading Konto."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_accounts WHERE model_id = ?
        """, (model_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Handle reset_at - use created_at as fallback for backward compatibility
        reset_at = row["created_at"]
        try:
            if row["reset_at"]:
                reset_at = row["reset_at"]
        except (KeyError, IndexError):
            pass
        
        # Handle total_fees_paid - default to 0.0 for backward compatibility
        total_fees_paid = 0.0
        try:
            if row["total_fees_paid"] is not None:
                total_fees_paid = row["total_fees_paid"]
        except (KeyError, IndexError):
            pass
        
        return PaperAccount(
            model_id=row["model_id"],
            model_package_id=row["model_package_id"] or "ppo_v1",
            balance_usdt=row["balance_usdt"],
            total_equity=row["total_equity"],
            initial_balance=row["initial_balance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            reset_at=datetime.fromisoformat(reset_at),
            total_trades=row["total_trades"],
            winning_trades=row["winning_trades"],
            losing_trades=row["losing_trades"],
            max_positions=row["max_positions"],
            default_leverage=row["default_leverage"],
            profit_target_pct=row["profit_target_pct"],
            time_limit_hours=row["time_limit_hours"],
            total_fees_paid=total_fees_paid
        )
    
    def get_or_create_account(self, model_id: str, **kwargs) -> PaperAccount:
        """Holt oder erstellt ein Paper-Trading Konto."""
        account = self.get_account(model_id)
        if account is None:
            account = self.create_account(model_id, **kwargs)
        return account
    
    def get_open_positions(self, model_id: str) -> List[PaperPosition]:
        """Holt alle offenen Positionen eines Modells."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_positions 
            WHERE model_id = ? AND status = 'OPEN'
            ORDER BY open_time DESC
        """, (model_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_position(row) for row in rows]
    
    def get_position_count(self, model_id: str) -> int:
        """Zählt offene Positionen."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as count FROM paper_positions 
            WHERE model_id = ? AND status = 'OPEN'
        """, (model_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return row["count"] if row else 0
    
    def can_open_position(self, model_id: str) -> bool:
        """Prüft ob eine neue Position geöffnet werden kann."""
        account = self.get_account(model_id)
        if not account:
            return False
        
        current_count = self.get_position_count(model_id)
        return current_count < account.max_positions
    
    def open_position(
        self,
        model_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        size_usdt: float,
        leverage: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None
    ) -> Optional[PaperPosition]:
        """Öffnet eine neue Paper-Position."""
        
        # Prüfe ob Position geöffnet werden kann
        if not self.can_open_position(model_id):
            print(f"[PaperTrading] Max positions reached for {model_id}")
            return None
        
        account = self.get_account(model_id)
        if not account:
            print(f"[PaperTrading] Account not found for {model_id}")
            return None
        
        # Verwende Default-Werte wenn nicht angegeben
        leverage = leverage or account.default_leverage
        take_profit_pct = take_profit_pct or account.profit_target_pct
        stop_loss_pct = stop_loss_pct or -3.0  # Default -3% SL
        
        # Prüfe ob genug Balance vorhanden (inkl. Margin + Entry Fee)
        margin_required = size_usdt / leverage
        entry_fee = size_usdt * FEE_RATE
        total_required = margin_required + entry_fee
        
        if account.balance_usdt < total_required:
            print(f"[PaperTrading] Insufficient balance: {account.balance_usdt} < {total_required} (margin: {margin_required}, fee: {entry_fee})")
            return None
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        now = datetime.utcnow()
        cursor.execute("""
            INSERT INTO paper_positions 
            (model_id, symbol, side, entry_price, size_usdt, leverage, 
             take_profit_pct, stop_loss_pct, open_time, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (model_id, symbol, side, entry_price, size_usdt, leverage,
              take_profit_pct, stop_loss_pct, now))
        
        position_id = cursor.lastrowid
        
        # Update Balance (Margin wird reserviert + Entry Fee wird abgezogen)
        new_balance = account.balance_usdt - total_required
        new_total_fees = account.total_fees_paid + entry_fee
        cursor.execute("""
            UPDATE paper_accounts 
            SET balance_usdt = ?, total_fees_paid = ?, updated_at = ?
            WHERE model_id = ?
        """, (new_balance, new_total_fees, now, model_id))
        
        print(f"[PaperTrading] Position opened: {symbol} {side}, Size: {size_usdt} USDT, Entry Fee: {entry_fee:.4f} USDT")
        
        conn.commit()
        conn.close()
        
        return self.get_position(position_id)
    
    def get_position(self, position_id: int) -> Optional[PaperPosition]:
        """Holt eine einzelne Position."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_positions WHERE id = ?
        """, (position_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_position(row)
    
    def _row_to_position(self, row: sqlite3.Row) -> PaperPosition:
        """Konvertiert DB-Row zu PaperPosition."""
        return PaperPosition(
            id=row["id"],
            model_id=row["model_id"],
            symbol=row["symbol"],
            side=row["side"],
            entry_price=row["entry_price"],
            size_usdt=row["size_usdt"],
            leverage=row["leverage"],
            take_profit_pct=row["take_profit_pct"],
            stop_loss_pct=row["stop_loss_pct"],
            open_time=datetime.fromisoformat(row["open_time"]),
            close_time=datetime.fromisoformat(row["close_time"]) if row["close_time"] else None,
            exit_price=row["exit_price"],
            realized_pnl_pct=row["realized_pnl_pct"],
            realized_pnl_usdt=row["realized_pnl_usdt"],
            status=row["status"],
            close_reason=row["close_reason"]
        )
    
    def calculate_unrealized_pnl(self, position: PaperPosition, current_price: float) -> tuple:
        """Berechnet unrealisierten P&L für eine Position."""
        if position.side == "Long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100 * position.leverage
        else:  # Short
            pnl_pct = (position.entry_price - current_price) / position.entry_price * 100 * position.leverage
        
        pnl_usdt = position.size_usdt * pnl_pct / 100
        return pnl_pct, pnl_usdt
    
    def close_position(
        self,
        position_id: int,
        exit_price: float,
        reason: str = "MANUAL"
    ) -> Optional[PaperPosition]:
        """Schließt eine Paper-Position."""
        position = self.get_position(position_id)
        if not position or position.status != "OPEN":
            return None
        
        # Berechne P&L
        pnl_pct, pnl_usdt = self.calculate_unrealized_pnl(position, exit_price)
        
        # Calculate exit fee (0.1% of position size)
        exit_fee = position.size_usdt * FEE_RATE
        pnl_usdt_after_fee = pnl_usdt - exit_fee
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        now = datetime.utcnow()
        
        # Update Position
        cursor.execute("""
            UPDATE paper_positions 
            SET exit_price = ?, realized_pnl_pct = ?, realized_pnl_usdt = ?,
                status = 'CLOSED', close_time = ?, close_reason = ?
            WHERE id = ?
        """, (exit_price, pnl_pct, pnl_usdt_after_fee, now, reason, position_id))
        
        # Update Account
        account = self.get_account(position.model_id)
        if account:
            # Margin zurück + P&L (nach Exit Fee)
            margin_returned = position.size_usdt / position.leverage
            new_balance = account.balance_usdt + margin_returned + pnl_usdt_after_fee
            
            # Update Trade Stats
            total_trades = account.total_trades + 1
            winning_trades = account.winning_trades + (1 if pnl_pct > 0 else 0)
            losing_trades = account.losing_trades + (1 if pnl_pct <= 0 else 0)
            
            # Update total fees paid
            new_total_fees = account.total_fees_paid + exit_fee
            
            cursor.execute("""
                UPDATE paper_accounts 
                SET balance_usdt = ?, total_equity = ?, updated_at = ?,
                    total_trades = ?, winning_trades = ?, losing_trades = ?, total_fees_paid = ?
                WHERE model_id = ?
            """, (new_balance, new_balance, now, total_trades, winning_trades, 
                  losing_trades, new_total_fees, position.model_id))
            
            print(f"[PaperTrading] Position closed: {position.symbol} {position.side}, PnL: {pnl_usdt:.4f} USDT, Exit Fee: {exit_fee:.4f} USDT, Net PnL: {pnl_usdt_after_fee:.4f} USDT")
        
        conn.commit()
        conn.close()
        
        return self.get_position(position_id)
    
    def update_positions_with_price(self, model_id: str, symbol: str, current_price: float) -> List[Dict]:
        """Aktualisiert alle Positionen mit aktuellem Preis und prüft TP/SL/Expiry."""
        positions = self.get_open_positions(model_id)
        closed_positions = []
        
        account = self.get_account(model_id)
        if not account:
            return []
        
        for pos in positions:
            # Prüfe TP/SL
            pnl_pct, pnl_usdt = self.calculate_unrealized_pnl(pos, current_price)
            
            should_close = False
            close_reason = None
            
            # Take Profit
            if pnl_pct >= pos.take_profit_pct:
                should_close = True
                close_reason = "TP"
            
            # Stop Loss
            elif pnl_pct <= pos.stop_loss_pct:
                should_close = True
                close_reason = "SL"
            
            # Zeitlimit prüfen
            time_open = datetime.utcnow() - pos.open_time
            if time_open.total_seconds() >= account.time_limit_hours * 3600:
                should_close = True
                close_reason = "EXPIRED"
            
            if should_close:
                closed_pos = self.close_position(pos.id, current_price, close_reason)
                if closed_pos:
                    closed_positions.append({
                        "position": asdict(closed_pos),
                        "pnl_pct": pnl_pct,
                        "pnl_usdt": pnl_usdt
                    })
        
        # Update Total Equity
        self._update_total_equity(model_id, current_price)
        
        return closed_positions
    
    def _update_total_equity(self, model_id: str, current_price: float):
        """Aktualisiert das Total Equity eines Accounts."""
        account = self.get_account(model_id)
        if not account:
            return
        
        positions = self.get_open_positions(model_id)
        total_unrealized_pnl = 0
        total_margin_used = 0
        
        for pos in positions:
            _, pnl_usdt = self.calculate_unrealized_pnl(pos, current_price)
            total_unrealized_pnl += pnl_usdt
            total_margin_used += (pos.size_usdt / pos.leverage)
        
        # Equity = Cash + Margin Used + Unrealized PnL
        total_equity = account.balance_usdt + total_margin_used + total_unrealized_pnl
        
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.utcnow()
        cursor.execute("""
            UPDATE paper_accounts 
            SET total_equity = ?, updated_at = ?
            WHERE model_id = ?
        """, (total_equity, now, model_id))
        conn.commit()
        conn.close()
        
        # Record equity history snapshot
        self._record_equity_snapshot(model_id, total_equity, account.balance_usdt, len(positions), total_unrealized_pnl)
    
    def _record_equity_snapshot(self, model_id: str, equity: float, balance: float, open_count: int, unrealized_pnl: float):
        """Records an equity history snapshot."""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.utcnow()
        
        # Check if we already have a record in the last minute (to avoid too many data points)
        cursor.execute("""
            SELECT timestamp FROM paper_equity_history 
            WHERE model_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (model_id,))
        
        row = cursor.fetchone()
        should_record = True
        
        if row:
            last_ts = datetime.fromisoformat(row["timestamp"])
            # Only record if at least 60 seconds have passed
            if (now - last_ts).total_seconds() < 60:
                should_record = False
        
        if should_record:
            cursor.execute("""
                INSERT INTO paper_equity_history 
                (model_id, timestamp, equity, balance, open_positions_count, unrealized_pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_id, now, equity, balance, open_count, unrealized_pnl))
            conn.commit()
        
        conn.close()
    
    def get_equity_history(self, model_id: str, limit: int = 500) -> List[Dict]:
        """Holt Equity-Verlauf für ein Modell."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_equity_history 
            WHERE model_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """, (model_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                "timestamp": row["timestamp"],
                "equity": row["equity"],
                "balance": row["balance"],
                "open_positions_count": row["open_positions_count"],
                "unrealized_pnl": row["unrealized_pnl"]
            })
        
        return result
    
    def get_closed_positions(self, model_id: str, limit: int = 50) -> List[PaperPosition]:
        """Holt geschlossene Positionen."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_positions 
            WHERE model_id = ? AND status = 'CLOSED'
            ORDER BY close_time DESC
            LIMIT ?
        """, (model_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_position(row) for row in rows]
    
    def get_performance_stats(self, model_id: str) -> Dict[str, Any]:
        """Holt Performance-Statistiken."""
        account = self.get_account(model_id)
        if not account:
            return {}
        
        open_positions = self.get_open_positions(model_id)
        closed_positions = self.get_closed_positions(model_id, limit=1000)
        
        # Berechne Stats
        total_return_pct = ((account.total_equity - account.initial_balance) / account.initial_balance) * 100
        win_rate = (account.winning_trades / account.total_trades * 100) if account.total_trades > 0 else 0
        
        # Durchschnittlicher P&L
        avg_pnl = 0
        avg_win = 0
        avg_loss = 0
        
        if closed_positions:
            pnls = [p.realized_pnl_pct for p in closed_positions if p.realized_pnl_pct is not None]
            if pnls:
                avg_pnl = sum(pnls) / len(pnls)
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p <= 0]
                avg_win = sum(wins) / len(wins) if wins else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Offene Exposure
        open_exposure = sum(pos.size_usdt for pos in open_positions)
        exposure_pct = (open_exposure / account.total_equity * 100) if account.total_equity > 0 else 0
        
        return {
            "model_id": model_id,
            "initial_balance": account.initial_balance,
            "current_balance": account.balance_usdt,
            "total_equity": account.total_equity,
            "total_return_pct": total_return_pct,
            "total_return_usdt": account.total_equity - account.initial_balance,
            "total_trades": account.total_trades,
            "winning_trades": account.winning_trades,
            "losing_trades": account.losing_trades,
            "win_rate": win_rate,
            "avg_pnl_pct": avg_pnl,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "open_positions_count": len(open_positions),
            "open_exposure_usdt": open_exposure,
            "open_exposure_pct": exposure_pct,
            "available_slots": account.max_positions - len(open_positions),
            "total_fees_paid": account.total_fees_paid
        }
    
    def get_daily_pnl(self, model_id: str, days: int = 30) -> List[Dict]:
        """Holt tägliche P&L-Daten."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM paper_daily_pnl 
            WHERE model_id = ?
            ORDER BY date DESC
            LIMIT ?
        """, (model_id, days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def reset_account(self, model_id: str, keep_history: bool = False) -> PaperAccount:
        """Resetet ein Paper-Trading Konto."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        now = datetime.utcnow()
        
        # Hard reset: Delete EVERYTHING for this model
        if not keep_history:
            cursor.execute("DELETE FROM paper_positions WHERE model_id = ?", (model_id,))
            cursor.execute("DELETE FROM paper_trade_history WHERE model_id = ?", (model_id,))
            cursor.execute("DELETE FROM paper_daily_pnl WHERE model_id = ?", (model_id,))
            # Also reset the account record completely
            cursor.execute("DELETE FROM paper_accounts WHERE model_id = ?", (model_id,))
        else:
            # Schließe nur offene Positionen
            cursor.execute("""
                UPDATE paper_positions 
                SET status = 'CLOSED', close_reason = 'RESET'
                WHERE model_id = ? AND status = 'OPEN'
            """, (model_id,))
            # Update reset_at timestamp even for soft reset
            cursor.execute("""
                UPDATE paper_accounts 
                SET reset_at = ?
                WHERE model_id = ?
            """, (now, model_id))
        
        conn.commit()
        conn.close()
        
        # Re-create fresh account with default values
        account = self.create_account(model_id)
        
        # Update reset_at to now for the new account
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE paper_accounts 
            SET reset_at = ?
            WHERE model_id = ?
        """, (now, model_id))
        conn.commit()
        conn.close()
        
        return self.get_account(model_id)


# Singleton Instance
_engine = None

def get_paper_engine() -> PaperTradingEngine:
    """Gibt die Paper Trading Engine Singleton zurück."""
    global _engine
    if _engine is None:
        _engine = PaperTradingEngine()
    return _engine


if __name__ == "__main__":
    # Test
    engine = get_paper_engine()
    
    # Erstelle Account
    account = engine.create_account("test_model")
    print(f"Created account: {account}")
    
    # Öffne Position
    pos = engine.open_position("test_model", "BTCUSDT", "Long", 50000.0, 1000.0, 7.0)
    print(f"Opened position: {pos}")
    
    # Aktualisiere mit Preis
    closed = engine.update_positions_with_price("test_model", "BTCUSDT", 51000.0)
    print(f"Closed positions: {closed}")
    
    # Stats
    stats = engine.get_performance_stats("test_model")
    print(f"Stats: {json.dumps(stats, indent=2, default=str)}")
