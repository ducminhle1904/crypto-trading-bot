"""
Logging utilities for trading bot.
"""
import csv
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

from telegram import Bot
from trading_bot.config import (
    logger, MAX_RETRIES, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SUMMARY_HEADERS, ACTIVE_STRATEGY, BASE_SUMMARY_FILE,
    get_strategy_summary_file, get_strategy_performance_file
)

# Initialize telegram bot
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Strategy performance metrics store
strategy_metrics = defaultdict(lambda: {
    'trades': 0,
    'wins': 0,
    'losses': 0,
    'total_profit': 0.0,
    'total_loss': 0.0,
    'profit_factor': 0.0,
    'win_rate': 0.0,
    'avg_profit': 0.0,
    'max_win': 0.0,
    'max_loss': 0.0,
    'current_balance': 1000.0,  # Default starting balance
    'peak_balance': 1000.0,
    'max_drawdown': 0.0,
    'long_trades': 0,
    'short_trades': 0,
    'long_wins': 0,
    'short_wins': 0,
    'last_updated': None
})

def setup_summary_logger():
    """Initialize the CSV summary log file with headers"""
    # Get the correct summary file based on active strategy
    summary_file = get_strategy_summary_file(ACTIVE_STRATEGY)
    
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(SUMMARY_HEADERS)
            logger.info(f"Created trade summary log file: {summary_file}")
    
    # Create a performance log file if it doesn't exist
    performance_file = get_strategy_performance_file(ACTIVE_STRATEGY)
    if not os.path.isfile(performance_file):
        save_performance_metrics()
        logger.info(f"Created strategy performance file: {performance_file}")

def log_trade_summary(data: Dict[str, Any]):
    """Log a trade event to the summary CSV file and update strategy metrics"""
    try:
        # Get the strategy name for file naming
        strategy_name = data.get('strategy', ACTIVE_STRATEGY or 'unknown')
        summary_file = get_strategy_summary_file(strategy_name)
        
        # Write to CSV
        with open(summary_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data.get(header, "") for header in SUMMARY_HEADERS])
        logger.debug(f"Logged trade summary: {data['action']} {data.get('side', '')}")
        
        # Update strategy performance metrics
        # Only process EXIT actions (completed trades)
        if data.get('action') == 'EXIT':
            profit_pct = data.get('profit_percent', 0.0)
            profit_amt = data.get('profit_amount', 0.0)
            side = data.get('side', 'UNKNOWN').upper()
            
            # Update strategy metrics
            strategy_metrics[strategy_name]['trades'] += 1
            
            # Track win/loss
            if profit_pct > 0:
                strategy_metrics[strategy_name]['wins'] += 1
                strategy_metrics[strategy_name]['total_profit'] += profit_pct
                strategy_metrics[strategy_name]['max_win'] = max(strategy_metrics[strategy_name]['max_win'], profit_pct)
                
                if side == 'LONG':
                    strategy_metrics[strategy_name]['long_wins'] += 1
                elif side == 'SHORT':
                    strategy_metrics[strategy_name]['short_wins'] += 1
            else:
                strategy_metrics[strategy_name]['losses'] += 1
                strategy_metrics[strategy_name]['total_loss'] += abs(profit_pct)
                strategy_metrics[strategy_name]['max_loss'] = min(strategy_metrics[strategy_name]['max_loss'], profit_pct)
            
            # Track trade sides
            if side == 'LONG':
                strategy_metrics[strategy_name]['long_trades'] += 1
            elif side == 'SHORT':
                strategy_metrics[strategy_name]['short_trades'] += 1
            
            # Update balance
            current_balance = data.get('balance', strategy_metrics[strategy_name]['current_balance'])
            strategy_metrics[strategy_name]['current_balance'] = current_balance
            
            # Track peak balance and drawdown
            if current_balance > strategy_metrics[strategy_name]['peak_balance']:
                strategy_metrics[strategy_name]['peak_balance'] = current_balance
            else:
                drawdown = (strategy_metrics[strategy_name]['peak_balance'] - current_balance) / strategy_metrics[strategy_name]['peak_balance'] * 100
                strategy_metrics[strategy_name]['max_drawdown'] = max(strategy_metrics[strategy_name]['max_drawdown'], drawdown)
            
            # Calculate derived metrics
            trades = strategy_metrics[strategy_name]['trades']
            wins = strategy_metrics[strategy_name]['wins']
            total_profit = strategy_metrics[strategy_name]['total_profit']
            total_loss = strategy_metrics[strategy_name]['total_loss']
            
            strategy_metrics[strategy_name]['win_rate'] = (wins / trades) if trades > 0 else 0.0
            strategy_metrics[strategy_name]['avg_profit'] = (total_profit - total_loss) / trades if trades > 0 else 0.0
            strategy_metrics[strategy_name]['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
            strategy_metrics[strategy_name]['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save metrics after updates
            save_performance_metrics(strategy_name)
            
    except Exception as e:
        logger.error(f"Error updating trade metrics: {e}")

def save_performance_metrics(strategy_name=None):
    """Save strategy performance metrics to a JSON file"""
    try:
        # Get appropriate file based on active strategy
        performance_file = get_strategy_performance_file(strategy_name or ACTIVE_STRATEGY)
        
        # If saving for a specific strategy, only include that strategy's data
        if strategy_name:
            metrics_dict = {strategy_name: dict(strategy_metrics[strategy_name])}
        else:
            # Convert defaultdict to regular dict for JSON serialization
            metrics_dict = {k: dict(v) for k, v in strategy_metrics.items()}
        
        with open(performance_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        logger.debug(f"Saved performance metrics to {performance_file}")
    except Exception as e:
        logger.error(f"Error saving performance metrics: {e}")

def load_performance_metrics(strategy_name=None):
    """Load strategy performance metrics from JSON file"""
    try:
        # Get appropriate file based on active strategy
        performance_file = get_strategy_performance_file(strategy_name or ACTIVE_STRATEGY)
        
        if os.path.exists(performance_file):
            with open(performance_file, 'r') as f:
                loaded_metrics = json.load(f)
                
            # Update the defaultdict with loaded values
            for strategy, metrics in loaded_metrics.items():
                strategy_metrics[strategy].update(metrics)
                
            logger.info(f"Loaded performance metrics from {performance_file}")
    except Exception as e:
        logger.error(f"Error loading performance metrics: {e}")

def load_previous_trades():
    """
    Load previous trades from the CSV file when restarting the bot.
    Returns a dictionary with the following information:
    - trades: Dictionary mapping strategy names to lists of trade results (profit percentages)
    - positions: Dictionary mapping strategy names to the last open position (if any)
    - balances: Dictionary mapping strategy names to the current balance
    - trade_id: The next trade ID to use
    """
    trades = defaultdict(list)
    positions = {}
    balances = defaultdict(lambda: 1000.0)  # Default starting balance
    trade_id = 1
    
    try:
        # Get appropriate file based on active strategy
        summary_file = get_strategy_summary_file(ACTIVE_STRATEGY)
        
        if not os.path.exists(summary_file):
            # If strategy-specific file doesn't exist, also check the default file
            if ACTIVE_STRATEGY and os.path.exists(BASE_SUMMARY_FILE):
                summary_file = BASE_SUMMARY_FILE
                logger.info(f"No strategy-specific trade file found, using default: {summary_file}")
            else:
                logger.info(f"No previous trade summary file found at {summary_file}")
                return {
                    'trades': dict(trades),
                    'positions': positions,
                    'balances': dict(balances),
                    'trade_id': trade_id
                }
        
        # Open positions tracking (to find the latest status of each open trade)
        open_positions = {}
        
        with open(summary_file, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            
            # Add default header for strategy if not in the file headers
            if 'strategy' not in reader.fieldnames and ACTIVE_STRATEGY:
                logger.warning(f"No strategy column in {summary_file}, will use active strategy: {ACTIVE_STRATEGY}")
                default_strategy = ACTIVE_STRATEGY
            else:
                default_strategy = 'unknown'
            
            for row in reader:
                # Try to get trade ID, otherwise skip
                try:
                    tid = int(row.get('trade_id', 0))
                    trade_id = max(trade_id, tid + 1)  # Next available trade ID
                except (ValueError, TypeError):
                    continue
                
                # Get strategy name, if not present, use active strategy
                strategy = row.get('strategy', default_strategy)
                
                # Handle ENTRY - store position info
                if row.get('action') == 'ENTRY':
                    try:
                        # Create position data
                        open_positions[tid] = {
                            'side': row.get('side', '').lower(),
                            'entry': float(row.get('entry_price', row.get('price', 0))),
                            'size': float(row.get('size', 0)),
                            'open_time': datetime.fromisoformat(row.get('timestamp', datetime.now().isoformat())),
                            'trade_id': tid,
                            'strategy_name': strategy
                        }
                    except (ValueError, TypeError):
                        logger.warning(f"Skipped malformed entry row for trade ID {tid}")
                
                # Handle EXIT - record completed trade
                elif row.get('action') == 'EXIT':
                    try:
                        profit_pct = float(row.get('profit_percent', 0))
                        trades[strategy].append(profit_pct)
                        # If trade is closed, remove from open positions
                        if tid in open_positions:
                            del open_positions[tid]
                        # Update strategy balance
                        if row.get('balance'):
                            balance = float(row.get('balance', 1000.0))
                            balances[strategy] = balance
                    except (ValueError, TypeError):
                        logger.warning(f"Skipped malformed exit row for trade ID {tid}")
        
        # Convert open positions to the format needed by the bot
        for trade_id, position_data in open_positions.items():
            # Make sure the position contains the minimum required data
            if all(k in position_data for k in ['side', 'entry', 'size', 'open_time', 'trade_id', 'strategy_name']):
                positions[position_data['strategy_name']] = position_data
            else:
                logger.warning(f"Skipped incomplete position data for trade ID {trade_id}")
        
        # Only log if there's actual data
        if trades or positions:
            logger.info(f"Loaded trading history from {summary_file}: {len(trades)} strategies, "
                       f"{sum(len(t) for t in trades.values())} completed trades, "
                       f"{len(positions)} open positions")
        
        return {
            'trades': dict(trades),
            'positions': positions,
            'balances': dict(balances),
            'trade_id': trade_id
        }
    
    except Exception as e:
        logger.error(f"Error loading previous trades: {e}")
        return {
            'trades': dict(trades),
            'positions': positions,
            'balances': dict(balances),
            'trade_id': trade_id
        }

async def generate_performance_report(strategies: Optional[List[str]] = None) -> str:
    """Generate a formatted performance report for all or specific strategies"""
    if not strategies:
        strategies = list(strategy_metrics.keys())
    
    report = "📊 <b>STRATEGY PERFORMANCE REPORT</b> 📊\n\n"
    
    for strategy in strategies:
        if strategy in strategy_metrics and strategy_metrics[strategy]['trades'] > 0:
            metrics = strategy_metrics[strategy]
            
            report += f"<b>{strategy.upper()}</b>\n"
            report += f"Trades: {metrics['trades']} (🟢 {metrics['wins']} | 🔴 {metrics['losses']})\n"
            report += f"Win Rate: {metrics['win_rate']:.2%}\n"
            report += f"Avg Profit: {metrics['avg_profit']:.2f}%\n"
            report += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            report += f"Max Win: {metrics['max_win']:.2f}% | Max Loss: {metrics['max_loss']:.2f}%\n"
            report += f"Balance: ${metrics['current_balance']:.2f}\n"
            report += f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            
            # Direction bias
            long_win_rate = metrics['long_wins'] / metrics['long_trades'] if metrics['long_trades'] > 0 else 0
            short_win_rate = metrics['short_wins'] / metrics['short_trades'] if metrics['short_trades'] > 0 else 0
            
            report += f"Long: {metrics['long_trades']} trades ({long_win_rate:.2%} win)\n"
            report += f"Short: {metrics['short_trades']} trades ({short_win_rate:.2%} win)\n"
            report += f"Last Updated: {metrics['last_updated']}\n\n"
    
    # Add summary comparison
    if len(strategies) > 1:
        report += "<b>STRATEGY RANKING (by profit factor)</b>\n"
        ranked_strategies = sorted(
            [(s, strategy_metrics[s]['profit_factor']) for s in strategies if strategy_metrics[s]['trades'] > 0],
            key=lambda x: x[1], reverse=True
        )
        
        for i, (strategy, pf) in enumerate(ranked_strategies, 1):
            report += f"{i}. {strategy}: {pf:.2f}\n"
    
    return report
            
async def send_performance_report(strategies: Optional[List[str]] = None):
    """Generate and send performance report via Telegram"""
    report = await generate_performance_report(strategies)
    return await send_telegram_message(report)
    
async def send_telegram_message(message: str) -> bool:
    """Send a message to Telegram. Returns True if successful, False otherwise."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Missing Telegram credentials. Message not sent.")
        return False
        
    retries = 0
    while retries < MAX_RETRIES:
        try:
            await telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            return True
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                logger.error(f"Failed to send Telegram message after {MAX_RETRIES} attempts: {e}")
                return False
            await asyncio.sleep(2 ** retries)  # Exponential backoff
    
    return False 