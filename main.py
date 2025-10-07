from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
import mysql.connector
from uuid import uuid4
from enum import Enum
from typing import List, Optional
from decimal import Decimal, ROUND_HALF_UP
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------------- Database Connection ----------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Naman@321",
        database="stock_exchange",
        autocommit=False
    )

# ---------------- Models ----------------
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    LIMIT = "limit"

class OrderStatus(str, Enum):
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"

class PortfolioItem(BaseModel):
    symbol: str
    quantity: int
    avg_price: float

class CreateUserRequest(BaseModel):
    username: str
    walletBalance: float = 10000.00
    portfolio: Optional[List[PortfolioItem]] = None

class OrderRequest(BaseModel):
    username: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: float
    quantity: int

    class Config:
        use_enum_values = True

class CancelOrderRequest(BaseModel):
    username: str

# ---------------- Valid Symbols ----------------
VALID_SYMBOLS = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'S']

def validate_symbol(symbol: str):
    if symbol not in VALID_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Invalid symbol. Valid symbols: {VALID_SYMBOLS}")

# ---------------- Decimal Precision Helpers ----------------
def to_decimal(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

# ---------------- Wallet Helpers ----------------
def get_wallet_balance(conn, username: str) -> Decimal:
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT balance FROM user_wallets WHERE username = %s", (username,))
    row = cursor.fetchone()
    return to_decimal(row["balance"]) if row else to_decimal(0.0)

def update_wallet_balance(conn, username: str, new_balance: Decimal):
    cursor = conn.cursor()
    cursor.execute("UPDATE user_wallets SET balance = %s WHERE username = %s", (float(new_balance), username))

def record_wallet_transaction(conn, username: str, amount: Decimal, trans_type: str,
                              order_id: str = None, trade_id: str = None):
    cursor = conn.cursor(dictionary=True)
    transaction_id = str(uuid4())
    
    # Get current balance after the transaction
    current_balance = get_wallet_balance(conn, username)

    cursor.execute("""
        INSERT INTO wallet_transactions 
        (transaction_id, username, amount, transaction_type, related_order_id, related_trade_id, balance_after)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (transaction_id, username, float(amount), trans_type, order_id, trade_id, float(current_balance)))

# ---------------- Portfolio Helpers ----------------
def get_portfolio_quantity(conn, username: str, symbol: str) -> int:
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT quantity FROM portfolio WHERE username = %s AND symbol = %s", (username, symbol))
    row = cursor.fetchone()
    return row["quantity"] if row else 0

def get_available_shares_for_selling(conn, username: str, symbol: str) -> int:
    """Get shares available for selling (total owned minus shares in open sell orders)"""
    cursor = conn.cursor(dictionary=True)
    
    # Get total shares owned
    cursor.execute("SELECT quantity FROM portfolio WHERE username = %s AND symbol = %s", (username, symbol))
    portfolio = cursor.fetchone()
    total_owned = portfolio["quantity"] if portfolio else 0
    
    # Get shares locked in open sell orders
    cursor.execute("""
        SELECT COALESCE(SUM(remaining_quantity), 0) as locked_shares 
        FROM orders 
        WHERE username = %s AND symbol = %s AND side = 'sell' AND status = 'open'
    """, (username, symbol))
    result = cursor.fetchone()
    locked_shares = result["locked_shares"] if result else 0
    
    available_shares = total_owned - locked_shares
    return max(0, available_shares)

def update_portfolio(conn, username: str, symbol: str, quantity_change: int, price: Decimal):
    cursor = conn.cursor(dictionary=True)
    
    # Get current portfolio entry
    cursor.execute("SELECT quantity, avg_price FROM portfolio WHERE username = %s AND symbol = %s", (username, symbol))
    current = cursor.fetchone()
    
    if current:
        current_qty = current["quantity"]
        current_avg = to_decimal(current["avg_price"])
        
        if quantity_change > 0:  # Buying - calculate new average price
            total_cost = (current_qty * current_avg) + (quantity_change * price)
            new_quantity = current_qty + quantity_change
            new_avg = total_cost / new_quantity
        else:  # Selling - average price remains same
            new_quantity = current_qty + quantity_change
            new_avg = current_avg
            
        if new_quantity == 0:
            cursor.execute("DELETE FROM portfolio WHERE username = %s AND symbol = %s", (username, symbol))
        else:
            cursor.execute("""
                UPDATE portfolio SET quantity = %s, avg_price = %s 
                WHERE username = %s AND symbol = %s
            """, (new_quantity, float(new_avg), username, symbol))
    else:
        if quantity_change > 0:  # Only create new entry if buying
            cursor.execute("""
                INSERT INTO portfolio (username, symbol, quantity, avg_price)
                VALUES (%s, %s, %s, %s)
            """, (username, symbol, quantity_change, float(price)))
        else:
            raise Exception(f"Cannot sell {symbol} that you don't own")

# ---------------- Order Helpers ----------------
def get_order(conn, order_id: str):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
    return cursor.fetchone()

def update_order_filled_quantity(conn, order_id: str, new_filled_quantity: int):
    """Update filled_quantity - trigger will handle status update"""
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE orders 
        SET filled_quantity = %s 
        WHERE order_id = %s
    """, (new_filled_quantity, order_id))

def get_order_remaining_quantity(conn, order_id: str) -> int:
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT remaining_quantity FROM orders WHERE order_id = %s", (order_id,))
    result = cursor.fetchone()
    return result["remaining_quantity"] if result else 0

def reserve_funds_for_buy_order(conn, username: str, total_cost: Decimal, order_id: str):
    """Reserve funds for a buy order by deducting from wallet"""
    current_balance = get_wallet_balance(conn, username)
    
    if current_balance < total_cost:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient funds. Balance: ${current_balance:.2f}, Required: ${total_cost:.2f}"
        )
    
    # Deduct the total cost from balance
    new_balance = current_balance - total_cost
    update_wallet_balance(conn, username, new_balance)
    # Transaction recording moved to place_order function after order creation
    
    return new_balance

def refund_unused_funds(conn, username: str, order_id: str, original_price: Decimal, 
                       original_quantity: int, filled_quantity: int):
    """Refund unused funds for partially filled or cancelled buy orders"""
    unused_quantity = original_quantity - filled_quantity
    if unused_quantity > 0:
        refund_amount = original_price * unused_quantity
        current_balance = get_wallet_balance(conn, username)
        new_balance = current_balance + refund_amount
        update_wallet_balance(conn, username, new_balance)
        record_wallet_transaction(conn, username, refund_amount, "trade_refund", order_id)
        logger.info(f"Refunded ${refund_amount:.2f} to {username} for order {order_id}")

# ---------------- Trading Engine Core ----------------
class TradingEngine:
    def __init__(self):
        self.processing_orders = set()
    
    async def process_order(self, order_id: str):
        """Main order processing function with locking to prevent duplicate processing"""
        if order_id in self.processing_orders:
            return
        
        self.processing_orders.add(order_id)
        try:
            conn = get_db_connection()
            try:
                # Get the order with FOR UPDATE lock
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM orders WHERE order_id = %s FOR UPDATE", (order_id,))
                order = cursor.fetchone()
                
                if not order or order["status"] in ["filled", "canceled"]:
                    return
                
                # Match orders based on side
                if order["side"] == "buy":
                    await self.match_buy_order(conn, order)
                else:
                    await self.match_sell_order(conn, order)
                    
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error processing order {order_id}: {str(e)}")
                raise
            finally:
                conn.close()
                
        finally:
            self.processing_orders.discard(order_id)
    
    async def match_buy_order(self, conn, buy_order):
        """Match a buy order with available sell orders"""
        cursor = conn.cursor(dictionary=True)
        
        # Find matching sell orders (price <= buy_order price, sorted by price ascending then time)
        cursor.execute("""
            SELECT * FROM orders 
            WHERE symbol = %s AND side = 'sell' AND status = 'open'
            AND price <= %s
            ORDER BY price ASC, created_at ASC
            FOR UPDATE
        """, (buy_order["symbol"], buy_order["price"]))
        
        sell_orders = cursor.fetchall()
        
        remaining_quantity = buy_order["remaining_quantity"]
        total_filled = buy_order["filled_quantity"]
        
        for sell_order in sell_orders:
            if remaining_quantity <= 0:
                break
                
            sell_remaining = sell_order["remaining_quantity"]
            trade_quantity = min(remaining_quantity, sell_remaining)
            
            if trade_quantity > 0:
                # Execute trade
                await self.execute_trade(conn, buy_order, sell_order, trade_quantity)
                remaining_quantity -= trade_quantity
                total_filled += trade_quantity
        
        # Update buy order filled quantity
        if total_filled > buy_order["filled_quantity"]:
            update_order_filled_quantity(conn, buy_order["order_id"], total_filled)
            
            # Refund any unused reserved funds (for limit orders where trade happened at lower price)
            actual_cost = sum([
                to_decimal(trade["price"]) * trade["quantity"] 
                for trade in self.get_order_trades(conn, buy_order["order_id"])
            ])
            reserved_cost = to_decimal(buy_order["price"]) * buy_order["quantity"]
            
            if actual_cost < reserved_cost:
                refund_amount = reserved_cost - actual_cost
                current_balance = get_wallet_balance(conn, buy_order["username"])
                new_balance = current_balance + refund_amount
                update_wallet_balance(conn, buy_order["username"], new_balance)
                record_wallet_transaction(conn, buy_order["username"], refund_amount, 
                                       "price_difference_refund", buy_order["order_id"])
    
    async def match_sell_order(self, conn, sell_order):
        """Match a sell order with available buy orders"""
        cursor = conn.cursor(dictionary=True)
        
        # Find matching buy orders (price >= sell_order price, sorted by price descending then time)
        cursor.execute("""
            SELECT * FROM orders 
            WHERE symbol = %s AND side = 'buy' AND status = 'open'
            AND price >= %s
            ORDER BY price DESC, created_at ASC
            FOR UPDATE
        """, (sell_order["symbol"], sell_order["price"]))
        
        buy_orders = cursor.fetchall()
        
        remaining_quantity = sell_order["remaining_quantity"]
        total_filled = sell_order["filled_quantity"]
        
        for buy_order in buy_orders:
            if remaining_quantity <= 0:
                break
                
            buy_remaining = buy_order["remaining_quantity"]
            trade_quantity = min(remaining_quantity, buy_remaining)
            
            if trade_quantity > 0:
                # Execute trade
                await self.execute_trade(conn, buy_order, sell_order, trade_quantity)
                remaining_quantity -= trade_quantity
                total_filled += trade_quantity
        
        # Update sell order filled quantity
        if total_filled > sell_order["filled_quantity"]:
            update_order_filled_quantity(conn, sell_order["order_id"], total_filled)
    
    async def execute_trade(self, conn, buy_order, sell_order, quantity: int):
        """Execute a single trade between buy and sell orders"""
        # Determine trade price (use the price of the order that was placed first)
        if buy_order["created_at"] <= sell_order["created_at"]:
            trade_price = to_decimal(buy_order["price"])
        else:
            trade_price = to_decimal(sell_order["price"])
        
        trade_id = str(uuid4())
        cursor = conn.cursor()
        
        try:
            # Create trade record
            cursor.execute("""
                INSERT INTO trades (trade_id, buy_order_id, sell_order_id, symbol, price, quantity)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (trade_id, buy_order["order_id"], sell_order["order_id"], 
                  buy_order["symbol"], float(trade_price), quantity))
            
            # Update portfolios
            update_portfolio(conn, buy_order["username"], buy_order["symbol"], quantity, trade_price)
            update_portfolio(conn, sell_order["username"], sell_order["symbol"], -quantity, trade_price)
            
            # Credit seller's wallet
            total_amount = trade_price * quantity
            seller_balance = get_wallet_balance(conn, sell_order["username"])
            new_seller_balance = seller_balance + total_amount
            update_wallet_balance(conn, sell_order["username"], new_seller_balance)
            record_wallet_transaction(conn, sell_order["username"], total_amount, "trade_receipt", 
                                    sell_order["order_id"], trade_id)
            
            logger.info(f"Trade executed: {quantity} {buy_order['symbol']} at ${trade_price:.2f} "
                       f"between {buy_order['username']}(buyer) and {sell_order['username']}(seller)")
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise
    
    def get_order_trades(self, conn, order_id: str):
        """Get all trades for an order"""
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM trades 
            WHERE buy_order_id = %s OR sell_order_id = %s
            ORDER BY executed_at
        """, (order_id, order_id))
        return cursor.fetchall()

# Initialize trading engine
trading_engine = TradingEngine()

# ---------------- API Endpoints ----------------
@app.post("/api/users")
def create_user(user_data: CreateUserRequest):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT username FROM users WHERE username = %s", (user_data.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")

        cursor.execute("INSERT INTO users (username) VALUES (%s)", (user_data.username,))
        cursor.execute("INSERT INTO user_wallets (username, balance) VALUES (%s, %s)",
                       (user_data.username, user_data.walletBalance))

        # Record initial deposit
        record_wallet_transaction(conn, user_data.username, to_decimal(user_data.walletBalance), "deposit")

        # Add portfolio
        if user_data.portfolio:
            for item in user_data.portfolio:
                validate_symbol(item.symbol)
                cursor.execute("""
                    INSERT INTO portfolio (username, symbol, quantity, avg_price)
                    VALUES (%s, %s, %s, %s)
                """, (user_data.username, item.symbol, item.quantity, item.avg_price))

        conn.commit()
        return {
            "username": user_data.username,
            "walletBalance": user_data.walletBalance,
            "message": f"User {user_data.username} created successfully",
            "portfolio": user_data.portfolio if user_data.portfolio else []
        }

    except mysql.connector.Error as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn: conn.close()
@app.post("/api/orders")
async def place_order(order: OrderRequest, background_tasks: BackgroundTasks):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        validate_symbol(order.symbol)

        # Verify user exists
        cursor.execute("SELECT username FROM users WHERE username = %s", (order.username,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        order_id = str(uuid4())
        order_price = to_decimal(order.price)
        total_cost = order_price * order.quantity

        # Pre-trade validations
        if order.side == "buy":
            # Check balance
            current_balance = get_wallet_balance(conn, order.username)
            if current_balance < total_cost:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient funds. Balance: ${current_balance:.2f}, Required: ${total_cost:.2f}"
                )
        else:  # sell order
            # Check available shares
            available_shares = get_available_shares_for_selling(conn, order.username, order.symbol)
            if available_shares < order.quantity:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient shares. Available: {available_shares} {order.symbol}, Trying to sell: {order.quantity}"
                )

        # Insert order FIRST
        cursor.execute("""
            INSERT INTO orders (order_id, username, symbol, side, type, price, quantity, status, filled_quantity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', 0)
        """, (order_id, order.username, order.symbol, order.side, order.type, 
              float(order_price), order.quantity))

        # NOW handle wallet transactions for buy orders
        if order.side == "buy":
            # Deduct funds
            new_balance = current_balance - total_cost
            update_wallet_balance(conn, order.username, new_balance)
            # Record transaction AFTER order is created
            record_wallet_transaction(conn, order.username, -total_cost, "trade_payment", order_id)

        conn.commit()
        
        # Process order matching in background
        background_tasks.add_task(trading_engine.process_order, order_id)
        
        return {
            "message": "Order placed successfully", 
            "orderId": order_id,
            "status": "open",
            "remaining_quantity": order.quantity
        }

    except mysql.connector.Error as e:
        if conn: 
            conn.rollback()
        logger.error(f"Database error in place_order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        if conn: 
            conn.rollback()
        logger.error(f"Unexpected error in place_order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn: 
            conn.close()
@app.post("/api/orders/{order_id}/cancel")
def cancel_order(order_id: str, cancel_request: CancelOrderRequest):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get order details with FOR UPDATE to lock the row
        cursor.execute("SELECT * FROM orders WHERE order_id = %s FOR UPDATE", (order_id,))
        order = cursor.fetchone()
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        if order["username"] != cancel_request.username:
            raise HTTPException(status_code=403, detail="Not authorized to cancel this order")
        
        if order["status"] in ["filled", "canceled"]:
            raise HTTPException(status_code=400, detail=f"Cannot cancel order that is {order['status']}")
        
        # Refund unused funds for buy orders
        if order["side"] == "buy":
            filled_quantity = order.get("filled_quantity", 0)
            refund_unused_funds(conn, order["username"], order_id, 
                              to_decimal(order["price"]), order["quantity"], filled_quantity)
        
        # Update order status to canceled
        cursor.execute("UPDATE orders SET status = 'canceled' WHERE order_id = %s", (order_id,))
        
        conn.commit()
        return {
            "message": "Order cancelled successfully",
            "orderId": order_id,
            "filled_quantity": order.get("filled_quantity", 0),
            "refunded_quantity": order["quantity"] - order.get("filled_quantity", 0)
        }
        
    except mysql.connector.Error as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn: conn.close()

@app.get("/api/orders/{order_id}")
def get_order_details(order_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get order details
        cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
        order = cursor.fetchone()
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        # Get associated trades
        cursor.execute("""
            SELECT * FROM trades 
            WHERE buy_order_id = %s OR sell_order_id = %s
            ORDER BY executed_at
        """, (order_id, order_id))
        trades = cursor.fetchall()
        
        # Calculate average execution price
        avg_price = None
        total_executed = 0
        if trades:
            total_quantity = sum(trade["quantity"] for trade in trades)
            total_value = sum(to_decimal(trade["price"]) * trade["quantity"] for trade in trades)
            avg_price = float(total_value / total_quantity) if total_quantity > 0 else None
            total_executed = total_quantity
        
        return {
            "order": order,
            "trades": trades,
            "execution_summary": {
                "filled_quantity": order.get("filled_quantity", 0),
                "remaining_quantity": order["remaining_quantity"],
                "total_executed": total_executed,
                "average_price": avg_price
            }
        }
    finally:
        if conn: conn.close()

@app.get("/api/orders")
def get_user_orders(username: str = Query(...), status: Optional[str] = Query(None)):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        if status:
            cursor.execute("""
                SELECT * FROM orders 
                WHERE username = %s AND status = %s 
                ORDER BY created_at DESC
            """, (username, status))
        else:
            cursor.execute("""
                SELECT * FROM orders 
                WHERE username = %s 
                ORDER BY created_at DESC
            """, (username,))
        
        orders = cursor.fetchall()
        return orders
    finally:
        if conn: conn.close()

@app.get("/api/orderbook/{symbol}")
def get_orderbook(symbol: str):
    validate_symbol(symbol)
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get buy orders (grouped by price)
        cursor.execute("""
            SELECT price, SUM(remaining_quantity) as total_quantity,
                   COUNT(*) as order_count
            FROM orders 
            WHERE symbol = %s AND side = 'buy' AND status = 'open'
            GROUP BY price 
            ORDER BY price DESC
            LIMIT 20
        """, (symbol,))
        buy_orders = cursor.fetchall()

        # Get sell orders (grouped by price)
        cursor.execute("""
            SELECT price, SUM(remaining_quantity) as total_quantity,
                   COUNT(*) as order_count
            FROM orders 
            WHERE symbol = %s AND side = 'sell' AND status = 'open'
            GROUP BY price 
            ORDER BY price ASC
            LIMIT 20
        """, (symbol,))
        sell_orders = cursor.fetchall()

        return {
            "symbol": symbol, 
            "buy_orders": buy_orders, 
            "sell_orders": sell_orders,
            "timestamp": datetime.now().isoformat()
        }
    finally:
        if conn: conn.close()

@app.get("/api/portfolio/{username}")
def get_portfolio(username: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get portfolio
        cursor.execute("""
            SELECT symbol, quantity, avg_price,
                   (quantity * avg_price) as total_value
            FROM portfolio 
            WHERE username = %s AND quantity > 0
            ORDER BY total_value DESC
        """, (username,))
        portfolio = cursor.fetchall()
        
        # Get locked shares in open sell orders
        cursor.execute("""
            SELECT symbol, SUM(remaining_quantity) as locked_quantity
            FROM orders 
            WHERE username = %s AND side = 'sell' AND status = 'open'
            GROUP BY symbol
        """, (username,))
        locked_shares = {row["symbol"]: row["locked_quantity"] for row in cursor.fetchall()}
        
        # Enhance portfolio with available shares
        for item in portfolio:
            symbol = item["symbol"]
            locked = locked_shares.get(symbol, 0)
            item["available_quantity"] = max(0, item["quantity"] - locked)
            item["locked_quantity"] = locked
        
        # Calculate totals
        total_value = sum(item["total_value"] for item in portfolio)
        
        return {
            "portfolio": portfolio,
            "summary": {
                "total_items": len(portfolio),
                "total_value": float(total_value) if total_value else 0
            }
        }
    finally:
        if conn: conn.close()

@app.get("/api/wallet/{username}/balance")
def get_wallet_balance_endpoint(username: str):
    conn = None
    try:
        conn = get_db_connection()
        balance = get_wallet_balance(conn, username)
        
        # Get reserved funds (money locked in open buy orders)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT COALESCE(SUM(price * remaining_quantity), 0) as reserved_funds
            FROM orders 
            WHERE username = %s AND side = 'buy' AND status = 'open'
        """, (username,))
        result = cursor.fetchone()
        reserved_funds = to_decimal(result["reserved_funds"]) if result else to_decimal(0.0)
        
        available_balance = balance - reserved_funds
        
        return {
            "username": username,
            "total_balance": float(balance),
            "reserved_funds": float(reserved_funds),
            "available_balance": float(available_balance)
        }
    finally:
        if conn: conn.close()

@app.get("/api/trades/{symbol}")
def get_trades_by_symbol(symbol: str, limit: int = Query(100, ge=1, le=1000)):
    validate_symbol(symbol)
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT t.*, 
                   bo.username as buyer, 
                   so.username as seller
            FROM trades t
            JOIN orders bo ON t.buy_order_id = bo.order_id
            JOIN orders so ON t.sell_order_id = so.order_id
            WHERE t.symbol = %s 
            ORDER BY t.executed_at DESC 
            LIMIT %s
        """, (symbol, limit))
        return cursor.fetchall()
    finally:
        if conn: conn.close()

@app.get("/api/trades")
def get_trades_by_user(username: str = Query(...)):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT t.*, 
                   bo.username as buyer,
                   so.username as seller,
                   CASE 
                     WHEN bo.username = %s THEN 'buy'
                     WHEN so.username = %s THEN 'sell'
                   END as user_side
            FROM trades t
            JOIN orders bo ON t.buy_order_id = bo.order_id
            JOIN orders so ON t.sell_order_id = so.order_id
            WHERE bo.username = %s OR so.username = %s
            ORDER BY t.executed_at DESC
        """, (username, username, username, username))
        return cursor.fetchall()
    finally:
        if conn: conn.close()

# Remove this part:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# And add this instead:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)