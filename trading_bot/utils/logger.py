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
    SUMMARY_LOG_FILE, SUMMARY_HEADERS
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
    file_exists = os.path.isfile(SUMMARY_LOG_FILE)
    with open(SUMMARY_LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(SUMMARY_HEADERS)
            logger.info(f"Created trade summary log file: {SUMMARY_LOG_FILE}")
    
    # Create a performance log file if it doesn't exist
    performance_file = "strategy_performance.json"
    if not os.path.isfile(performance_file):
        save_performance_metrics()
        logger.info(f"Created strategy performance file: {performance_file}")

def log_trade_summary(data: Dict[str, Any]):
    """Log a trade event to the summary CSV file and update strategy metrics"""
    try:
        # Write to CSV
        with open(SUMMARY_LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data.get(header, "") for header in SUMMARY_HEADERS])
        logger.debug(f"Logged trade summary: {data['action']} {data.get('side', '')}")
        
        # Update strategy performance metrics
        strategy_name = data.get('strategy', 'unknown')
        
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
            save_performance_metrics()
            
    except Exception as e:
        logger.error(f"Error updating trade metrics: {e}")

def save_performance_metrics():
    """Save strategy performance metrics to a JSON file"""
    try:
        # Convert defaultdict to regular dict for JSON serialization
        metrics_dict = {k: dict(v) for k, v in strategy_metrics.items()}
        
        with open("strategy_performance.json", 'w') as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving performance metrics: {e}")

def load_performance_metrics():
    """Load strategy performance metrics from JSON file"""
    try:
        if os.path.exists("strategy_performance.json"):
            with open("strategy_performance.json", 'r') as f:
                loaded_metrics = json.load(f)
                
            # Update the defaultdict with loaded values
            for strategy, metrics in loaded_metrics.items():
                strategy_metrics[strategy].update(metrics)
                
            logger.info(f"Loaded performance metrics for {len(loaded_metrics)} strategies")
    except Exception as e:
        logger.error(f"Error loading performance metrics: {e}")

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

async def send_telegram_message(message: str, retries: int = 0) -> bool:
    """Send message to Telegram chat"""
    try:
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        logger.info(f"Sent Telegram message: {message}")
        return True
    except Exception as e:
        if retries < MAX_RETRIES:
            logger.warning(f"Error sending Telegram message, retrying ({retries+1}/{MAX_RETRIES}): {e}")
            await asyncio.sleep(2 ** retries)
            return await send_telegram_message(message, retries + 1)
        else:
            logger.error(f"Failed to send Telegram message after {MAX_RETRIES} attempts: {e}")
            return False
            
async def send_performance_report(strategies: Optional[List[str]] = None):
    """Generate and send performance report via Telegram"""
    report = await generate_performance_report(strategies)
    return await send_telegram_message(report)

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
        if not os.path.exists(SUMMARY_LOG_FILE):
            logger.info(f"No previous trade summary file found at {SUMMARY_LOG_FILE}")
            return {
                'trades': dict(trades),
                'positions': positions,
                'balances': dict(balances),
                'trade_id': trade_id
            }
        
        # Open positions tracking (to find the latest status of each open trade)
        open_positions = {}  # {trade_id: position_data}
        
        with open(SUMMARY_LOG_FILE, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows with missing critical data
                if not row.get('timestamp') or not row.get('action'):
                    continue
                
                # Get the strategy name, defaulting to 'unknown'
                strategy = row.get('strategy', 'unknown')
                
                # Parse trade_id and update the next available ID
                if row.get('trade_id') and row['trade_id'].isdigit():
                    current_trade_id = int(row['trade_id'])
                    trade_id = max(trade_id, current_trade_id + 1)
                
                # Track trades and balances
                if row['action'] == 'EXIT':
                    # Add the trade result to the strategy's trade history
                    if row.get('profit_percent') and row['profit_percent']:
                        try:
                            profit_pct = float(row['profit_percent'])
                            trades[strategy].append(profit_pct)
                        except ValueError:
                            pass
                    
                    # Update the balance
                    if row.get('balance') and row['balance']:
                        try:
                            balances[strategy] = float(row['balance'])
                        except ValueError:
                            pass
                    
                    # Remove the closed position from open positions
                    if row.get('trade_id') in open_positions:
                        del open_positions[row['trade_id']]
                
                elif row['action'] == 'ENTRY':
                    # Track the open position
                    try:
                        # Parse the entry price and timestamp
                        entry_price = float(row['entry_price']) if row.get('entry_price') else float(row['price'])
                        timestamp = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
                        
                        # Create a position object for tracking
                        position_data = {
                            'side': row['side'].lower(),
                            'entry': entry_price,
                            'size': float(row['size']) if row.get('size') else 0.0,
                            'open_time': timestamp,
                            'trade_id': int(row['trade_id']) if row['trade_id'].isdigit() else 1,
                            'strategy_name': strategy,
                            'signals': row.get('signals', '')
                        }
                        
                        open_positions[row['trade_id']] = position_data
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error parsing position data: {e}")
                
                # For UPDATE actions, we just want to keep track of the latest balance
                elif row['action'] == 'UPDATE':
                    if row.get('balance') and row['balance']:
                        try:
                            balances[strategy] = float(row['balance'])
                        except ValueError:
                            pass
        
        # Convert open positions to the format needed by the bot
        for trade_id, position_data in open_positions.items():
            # Make sure the position contains the minimum required data
            if all(k in position_data for k in ['side', 'entry', 'size', 'open_time', 'trade_id', 'strategy_name']):
                positions[position_data['strategy_name']] = position_data
            else:
                logger.warning(f"Skipped incomplete position data for trade ID {trade_id}")
        
        # Only log if there's actual data
        if trades or positions:
            logger.info(f"Loaded trading history from {SUMMARY_LOG_FILE}: {len(trades)} strategies, "
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