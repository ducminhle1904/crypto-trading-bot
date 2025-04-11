"""
Data models for trading positions and trade results.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class Position:
    """Trading position model."""
    side: str  # 'long' or 'short'
    entry: float
    size: float
    open_time: datetime
    trade_id: int
    trailing_stop: Optional[float] = None
    open_candles: int = 0
    strategy_name: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'side': self.side,
            'entry': self.entry,
            'size': self.size,
            'open_time': self.open_time,
            'trade_id': self.trade_id,
            'trailing_stop': self.trailing_stop,
            'open_candles': self.open_candles,
            'strategy_name': self.strategy_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary."""
        return cls(
            side=data['side'],
            entry=data['entry'],
            size=data['size'],
            open_time=data['open_time'],
            trade_id=data['trade_id'],
            trailing_stop=data.get('trailing_stop'),
            open_candles=data.get('open_candles', 0),
            strategy_name=data.get('strategy_name', 'default')
        )


@dataclass
class TradeResult:
    """Trading result model."""
    trade_id: int
    side: str
    entry_price: float
    exit_price: float
    size: float
    open_time: datetime
    close_time: datetime
    profit_amount: float
    profit_percent: float
    signals: List[str]
    strategy_name: str
    position_duration: float  # in minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade result to dictionary."""
        return {
            'trade_id': self.trade_id,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'open_time': self.open_time,
            'close_time': self.close_time,
            'profit_amount': self.profit_amount,
            'profit_percent': self.profit_percent,
            'signals': self.signals,
            'strategy_name': self.strategy_name,
            'position_duration': self.position_duration,
            'trade_status': 'PROFIT' if self.profit_amount > 0 else 'LOSS'
        } 