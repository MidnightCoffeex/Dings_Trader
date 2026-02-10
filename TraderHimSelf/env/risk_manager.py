"""
risk_manager.py

Implementiert den Risk Manager Wrapper (Schritt 7 der Roadmap).
Dient als Sicherheitslayer zwischen Agenten-Entscheidung und Execution.
"""

from typing import List, Tuple, Any, Optional

class RiskManager:
    """
    Risk Manager für den Dings-Trader.
    
    Verantwortlichkeiten:
    1. Hard Caps erzwingen (Exposure, Max Positions, Leverage, SL/TP).
    2. No-Hedge Rule (v1) durchsetzen.
    3. Soft Controls bereitstellen (Entry Penalty Berechnung).
    
    Kann sowohl im Training (Gym Env) als auch in der Live-Execution genutzt werden.
    """
    
    def __init__(self, 
                 max_exposure_pct: float = 0.10, 
                 max_positions: int = 10,
                 entry_penalty_bps: float = 2.0):
        """
        Initialisiert den Risk Manager.
        
        Args:
            max_exposure_pct: Maximales Exposure in % der Equity (0.10 = 10%).
            max_positions: Maximale Anzahl offener Lots/Positionen.
            entry_penalty_bps: Penalty pro neuem Trade in Basispunkten (2.0 bps = 0.0002).
        """
        self.max_exposure_pct = max_exposure_pct
        self.max_positions = max_positions
        self.entry_penalty_factor = entry_penalty_bps / 10000.0
        
        # Hardcoded Limits aus Roadmap Schritt 7.1
        self.leverage_range = (1, 10)
        self.sl_mult_range = (0.5, 3.0)
        self.tp_mult_range = (0.5, 6.0)

    def validate_action(self, 
                       direction: str, 
                       leverage: int, 
                       sl_mult: float, 
                       tp_mult: float, 
                       equity: float, 
                       open_positions: List[Any]) -> Tuple[str, int, float, float]:
        """
        Validiert und korrigiert eine geplante Aktion des Agenten.
        
        Args:
            direction: 'long', 'short' oder 'flat'.
            leverage: Gewünschter Hebel (wird auf [1, 10] geclampt).
            sl_mult: SL-Multiplikator (wird auf [0.5, 3.0] geclampt).
            tp_mult: TP-Multiplikator (wird auf [0.5, 6.0] geclampt, muss >= SL sein).
            equity: Aktuelle Account-Equity.
            open_positions: Liste der offenen Positionen (Objekte mit .side und .margin_used).
            
        Returns:
            Ein Tupel (corrected_direction, corrected_leverage, corrected_sl, corrected_tp).
            Falls Limits verletzt werden, wird direction auf 'flat' gesetzt.
        """
        
        # 1. Parameter Clamping (passiert immer, auch wenn flat, für saubere Daten)
        # Leverage: [1, 10]
        c_leverage = max(self.leverage_range[0], min(self.leverage_range[1], int(leverage)))
        
        # SL/TP Limits
        c_sl = max(self.sl_mult_range[0], min(self.sl_mult_range[1], sl_mult))
        c_tp = max(self.tp_mult_range[0], min(self.tp_mult_range[1], tp_mult))
        
        # Rule: TP >= SL (Roadmap 7.1.4)
        if c_tp < c_sl:
            c_tp = c_sl

        # Wenn der Agent eh nichts tun will, sind wir fertig.
        if direction == 'flat':
            return 'flat', c_leverage, c_sl, c_tp

        # 2. Max Open Positions Check
        if len(open_positions) >= self.max_positions:
            return 'flat', c_leverage, c_sl, c_tp

        # 3. No-Hedge Rule (v1)
        # Wenn bereits Positionen offen sind, darf nur in dieselbe Richtung eröffnet werden.
        # Hedging (Long + Short gleichzeitig) ist deaktiviert.
        if open_positions:
            # Wir prüfen die Richtung der ersten Position (in v1 sollten alle gleich sein)
            existing_side = open_positions[0].side
            if direction != existing_side:
                return 'flat', c_leverage, c_sl, c_tp

        # 4. Exposure Cap Check
        # Berechne das bereits genutzte Exposure (Margin)
        exposure_open_margin = sum(p.margin_used for p in open_positions)
        
        # Berechne das noch verfügbare Exposure für das Limit
        available_exposure = (equity * self.max_exposure_pct) - exposure_open_margin
        
        # Wenn kein Platz mehr im Exposure-Limit ist, verbiete neuen Trade
        if available_exposure <= 0:
            return 'flat', c_leverage, c_sl, c_tp

        # Wenn alle Checks bestanden: Aktion erlauben (mit geclampten Parametern)
        return direction, c_leverage, c_sl, c_tp

    def get_entry_penalty(self, equity: float) -> float:
        """
        Berechnet die Entry Penalty (Soft Control).
        Diese Kosten fließen in den Reward ein, um Overtrading zu bestrafen.
        
        Formel: 0.0002 * Equity
        """
        return equity * self.entry_penalty_factor
