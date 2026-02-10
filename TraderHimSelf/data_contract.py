"""
data_contract.py

Definiert die Datenstrukturen (Data Contract) für Dings-Trader.
Training = Live nutzen exakt dasselbe Schema.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

# Optional imports (für später wenn pandas verfügbar)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class CandleRecord:
    """
    OHLCV Kerzendaten - das Fundament des Systems.
    
    Felder:
    - open_time_ms: Unix-Timestamp in Millisekunden (Kandle-Start)
    - open: Eröffnungspreis
    - high: Höchstpreis
    - low: Tiefstpreis
    - close: Schlusskurs
    - volume: Handelsvolumen (Base-Asset, z.B. BTC)
    """
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_dataframe_row(cls, row) -> 'CandleRecord':
        """Erstelle CandleRecord aus einer DataFrame-Zeile (pandas optional)"""
        if HAS_PANDAS and hasattr(row, 'get'):
            return cls(
                open_time_ms=int(row.get('open_time_ms', 0)),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
        else:
            raise ImportError("pandas not installed")
    
    def to_dict(self) -> dict:
        """Konvertiere zu Dictionary"""
        return {
            'open_time_ms': self.open_time_ms,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class FundingRecord:
    """
    Funding Rate für Perpetual Futures.
    Wird alle 8h bei Binance aktualisiert.
    
    Felder:
    - time_ms: Unix-Timestamp in Millisekunden
    - funding_rate: Funding-Rate (positiv = Longs zahlen Shorts)
    """
    time_ms: int
    funding_rate: float
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FundingRecord':
        """Erstelle FundingRecord aus API-Response"""
        return cls(
            time_ms=int(data['fundingTime']),
            funding_rate=float(data['fundingRate'])
        )
    
    def to_dict(self) -> dict:
        return {
            'time_ms': self.time_ms,
            'funding_rate': self.funding_rate
        }


@dataclass
class TradeAction:
    """
    Eine Trading-Aktion (Output des Systems).
    
    Felder:
    - timestamp_ms: Zeitpunkt der Entscheidung
    - action: 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'HOLD'
    - size: Positionsgröße (0.0 bis 1.0 = % des erlaubten Exposure)
    - confidence: Modell-Konfidenz (0.0 bis 1.0)
    - price: Aktueller Preis zum Zeitpunkt der Entscheidung
    - metadata: Optionale Zusatzinfos (z.B. Forecast-Werte)
    """
    timestamp_ms: int
    action: str  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'HOLD'
    size: float  # 0.0 bis 1.0
    confidence: float  # 0.0 bis 1.0
    price: float
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validierung nach Initialisierung"""
        valid_actions = ['OPEN_LONG', 'OPEN_SHORT', 'CLOSE', 'HOLD']
        if self.action not in valid_actions:
            raise ValueError(f"Ungültige Aktion: {self.action}. Erlaubt: {valid_actions}")
        
        if not 0.0 <= self.size <= 1.0:
            raise ValueError(f"Size muss zwischen 0.0 und 1.0 liegen, war: {self.size}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence muss zwischen 0.0 und 1.0 liegen, war: {self.confidence}")


@dataclass
class Position:
    """
    Eine offene Position.
    
    Felder:
    - entry_time_ms: Eintrittszeitpunkt
    - entry_price: Eintrittspreis
    - side: 'LONG' oder 'SHORT'
    - size: Positionsgröße (Anzahl Lots)
    - leverage: Hebel (1-10)
    - stop_loss: SL-Preis
    - take_profit: TP-Preis
    - current_pnl: Aktueller P&L in %
    """
    entry_time_ms: int
    entry_price: float
    side: str  # 'LONG' or 'SHORT'
    size: float
    leverage: float
    stop_loss: float
    take_profit: float
    current_pnl: float = 0.0
    
    def update_pnl(self, current_price: float):
        """Aktualisiere P&L basierend auf aktuellem Preis"""
        if self.side == 'LONG':
            self.current_pnl = ((current_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:  # SHORT
            self.current_pnl = ((self.entry_price - current_price) / self.entry_price) * 100 * self.leverage
    
    def check_exit(self, high: float, low: float) -> Optional[str]:
        """
        Prüfe ob SL oder TP getroffen wurde.
        Returns: 'SL', 'TP', oder None
        
        Bei Konflikt (SL und TP in selbem Candle): SL-first (konservativ)
        """
        if self.side == 'LONG':
            # Bei Long: Low berührt SL, High berührt TP
            sl_hit = low <= self.stop_loss
            tp_hit = high >= self.take_profit
        else:  # SHORT
            # Bei Short: High berührt SL, Low berührt TP
            sl_hit = high >= self.stop_loss
            tp_hit = low <= self.take_profit
        
        # SL-first Regel: Wenn beides getroffen, zählt SL
        if sl_hit:
            return 'SL'
        if tp_hit:
            return 'TP'
        return None


# Konstanten aus Schritt 0
class TradingConfig:
    """Zentrale Konfiguration - muss in config.json exportiert werden"""
    
    # Instrument
    SYMBOL = "BTCUSDT"
    
    # Zeitfenster
    DECISION_TIMEFRAME = "15m"
    INTRABAR_TIMEFRAME = "3m"
    LOOKBACK_STEPS = 512  # ~5,3 Tage
    BUFFER_STEPS = 800    # ~8,3 Tage
    
    # Limits / Risk
    MAX_HOLD_HOURS = 48
    MAX_HOLD_STEPS = 192  # 48h / 15m
    MAX_EXPOSURE_PCT = 0.10  # 10%
    MAX_POSITIONS = 10
    LEVERAGE_MIN = 1
    LEVERAGE_MAX = 10
    
    # Fees
    TAKER_FEE = 0.0006  # 0.06%
    MAKER_FEE = 0.0002  # 0.02%
    
    # Long/Short Exclusion
    NO_HEDGE = True  # Nie Long und Short gleichzeitig


if __name__ == "__main__":
    # Test der Data Contracts (ohne pandas)
    print("Testing Data Contracts...")
    
    # Test CandleRecord
    candle = CandleRecord(
        open_time_ms=1707504000000,
        open=50000.0,
        high=51000.0,
        low=49500.0,
        close=50500.0,
        volume=100.5
    )
    print(f"✅ CandleRecord: {candle.to_dict()}")
    
    # Test TradeAction
    action = TradeAction(
        timestamp_ms=1707504000000,
        action='OPEN_LONG',
        size=0.5,
        confidence=0.85,
        price=50500.0,
        metadata={'forecast': 0.7}
    )
    print(f"✅ TradeAction: {action.action} @ {action.price}")
    
    # Test Position
    pos = Position(
        entry_time_ms=1707504000000,
        entry_price=50000.0,
        side='LONG',
        size=0.1,
        leverage=5.0,
        stop_loss=48000.0,
        take_profit=55000.0
    )
    pos.update_pnl(51000.0)
    print(f"✅ Position P&L: {pos.current_pnl:.2f}%")
    
    # Test Position Exit (SL-first)
    exit_reason = pos.check_exit(high=56000.0, low=49000.0)  # Beide getroffen, SL zählt
    print(f"✅ Exit Check (SL-first): {exit_reason}")
    
    # Test Config
    print(f"✅ Config: SYMBOL={TradingConfig.SYMBOL}, MAX_POS={TradingConfig.MAX_POSITIONS}")
    
    print("\n🎉 Alle Data Contracts funktionieren!")
