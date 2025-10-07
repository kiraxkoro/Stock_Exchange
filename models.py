from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import mysql.connector
from uuid import uuid4
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import logging
from threading import Lock

# ---------------- Setup ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# In-memory storage (thread-safe with a lock)
wallet_lock = Lock()
user_wallets: Dict[str, float] = {}                  # available cash
user_reserved_total: Dict[str, float] = {}           # total reserved across orders
user_reserved_by_order: Dict[str, float] = {}        # order_id -> reserved amount
user_initial_shares: Dict[str, Dict[str, int]] = {}  # username -> {symbol: qty}

# ---------------- Database Connection ---------------- #
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Naman@321",
            database="stock_exchange",
            autocommit=False  # we'll manage transactions explicitly
        )
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# ---------------- Models ---------------- #
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    LIMIT = "limit"

class PortfolioItem(BaseModel):
    symbol: str
    quantity: int
    avg_price: float

class CreateUserRequest(BaseModel):
    username: str
    walletBalance: float = 10000.00
    portfolio: Optional[List[PortfolioItem]] = None

class OrderRequest(BaseModel):
    userId: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: float
    quantity: int

    class Config:
        use_enum_values = True

# ---------------- Helper Functions ---------------- #
def get_username_from_id(user_id: str) -> Optional[str]:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT username FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        return user["username"] if user else None
    except mysql.connector.Error as e:
        logger.error(f"Error getting username for {user_id}: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def update_portfolio_buy(cursor, user_id: str, symbol: str, buy_qty: int, buy_price: float):
    """
    Update or insert portfolio row for a buyer: compute new avg_price in Python to avoid SQL expression issues.
    Uses the provided cursor and does NOT commit.
    """
    cursor.execute("SELECT quantity, avg_price FROM portfolio WHERE user_id = %s AND symbol = %s", (user_id, symbol))
    existing = cursor.fetchone()
    if existing and existing.get("quantity", 0) > 0:
        old_qty = existing["quantity"]
        old_avg = existing["avg_price"] or 0.0
        new_qty = old_qty + buy_qty
        new_avg = ((old_avg * old_qty) + (buy_price * buy_qty)) / new_qty
        cursor.execute("""
            UPDATE portfolio
            SET quantity = %s, avg_price = %s
            WHERE user_id = %s AND symbol = %s
        """, (new_qty, new_avg, user_id, symbol))
    else:
        cursor.execute("""
            INSERT INTO portfolio (user_id, symbol, quantity, avg_price)
            VALUES (%s, %s, %s, %s)
        """, (user_id, symbol, buy_qty, buy_price))

def update_portfolio_sell(cursor, user_id: str, symbol: str, sell_qty: int):
    """
    Subtract sell_qty from seller's portfolio. Remove row if quantity <= 0.
    """
    cursor.execute("SELECT quantity FROM portfolio WHERE user_id = %s AND symbol = %s", (user_id, symbol))
    existing = cursor.fetchone()
    existing_qty = existing["quantity"] if existing else 0
    new_qty = existing_qty - sell_qty
    if new_qty > 0:
        cursor.execute("""
            UPDATE portfolio
            SET quantity = %s
            WHERE user_id = %s AND symbol = %s
        """, (new_qty, user_id, symbol))
    else:
        cursor.execute("""
            DELETE FROM portfolio
            WHERE user_id = %s AND symbol = %s
        """, (user_id, symbol))

def reserve_funds_for_order(order_id: str, username: str, amount: float):
    with wallet_lock:
        user_reserved_by_order[order_id] = amount
        user_reserved_total[username] = user_reserved_total.get(username, 0.0) + amount
        # user_wallets is expected to have been reduced already by 'amount' at reservation time

def consume_reserved_for_order(order_id: str, username: str, amount: float):
    """
    Consume 'amount' from the reservation for order_id (for a matching trade).
    Decrements both per-order and per-user reserved totals.
    """
    with wallet_lock:
        remaining_for_order = user_reserved_by_order.get(order_id, 0.0)
        use_amount = min(remaining_for_order, amount)
        user_reserved_by_order[order_id] = remaining_for_order - use_amount
        user_reserved_total[username] = user_reserved_total.get(username, 0.0) - use_amount
        if user_reserved_by_order[order_id] <= 0:
            user_reserved_by_order.pop(order_id, None)
        # we do NOT add to user_wallets here (that is done for refunds or seller payments)

def refund_reserved_for_order(order_id: str, username: str):
    """
    Refund any remaining reservation for order_id back to the user's available wallet.
    """
    with wallet_lock:
        remaining = user_reserved_by_order.pop(order_id, 0.0)
        if remaining:
            # reduce reserved total and add to available wallet
            user_reserved_total[username] = max(0.0, user_reserved_total.get(username, 0.0) - remaining)
            user_wallets[username] = user_wallets.get(username, 0.0) + remaining
            logger.info(f"Refunded ${remaining:.2f} to {username} for order {order_id}")

# ---------------- Trade Matching Engine ---------------- #
def execute_trades(symbol: str):
    """
    Match buy and sell limit orders for the provided symbol in a loop until no match is possible.
    Uses DB transactions and updates in-memory wallet/reservation maps safely.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        while True:
            # Get the best buy order (highest price, earliest)
            cursor.execute("""
                SELECT * FROM orders
                WHERE symbol = %s AND side = 'buy' AND status = 'open'
                ORDER BY price DESC, created_at ASC
                LIMIT 1
            """, (symbol,))
            buy_order = cursor.fetchone()

            # Get the best sell order (lowest price, earliest)
            cursor.execute("""
                SELECT * FROM orders
                WHERE symbol = %s AND side = 'sell' AND status = 'open'
                ORDER BY price ASC, created_at ASC
                LIMIT 1
            """, (symbol,))
            sell_order = cursor.fetchone()

            # No match possible
            if not buy_order or not sell_order:
                break

            # Prices must cross (buyer willing to pay >= seller ask)
            if buy_order["price"] < sell_order["price"]:
                break

            # Determine trade quantity and trade price (price-time priority; use seller price)
            trade_qty = min(buy_order["quantity"], sell_order["quantity"])
            trade_price = sell_order["price"]
            trade_total = trade_qty * trade_price

            # Resolve usernames
            buyer_username = get_username_from_id(buy_order["user_id"])
            seller_username = get_username_from_id(sell_order["user_id"])

            if not buyer_username or not seller_username:
                # If usernames cannot be resolved, cancel the orders to avoid inconsistency
                logger.error(f"Could not resolve usernames for orders {buy_order['order_id']} or {sell_order['order_id']}. Cancelling them.")
                cursor.execute("UPDATE orders SET status = 'canceled' WHERE order_id IN (%s, %s)", (buy_order["order_id"], sell_order["order_id"]))
                conn.commit()
                break

            # Ensure the buyer actually has reserved funds for this order (defence in depth)
            with wallet_lock:
                reserved_for_buy_order = user_reserved_by_order.get(buy_order["order_id"], 0.0)

            if reserved_for_buy_order <= 0 and buy_order["price"] * buy_order["quantity"] > 0:
                # no reservation; cancel buy order
                logger.info(f"Buy order {buy_order['order_id']} has no reservation; cancelling.")
                cursor.execute("UPDATE orders SET status = 'canceled' WHERE order_id = %s", (buy_order["order_id"],))
                conn.commit()
                break

            # Insert trade record
            trade_id = str(uuid4())
            cursor.execute("""
                INSERT INTO trades (trade_id, buy_order_id, sell_order_id, symbol, price, quantity, executed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (trade_id, buy_order["order_id"], sell_order["order_id"], symbol, trade_price, trade_qty, datetime.utcnow()))

            # Update order remaining quantities
            cursor.execute("UPDATE orders SET quantity = quantity - %s WHERE order_id = %s", (trade_qty, buy_order["order_id"]))
            cursor.execute("UPDATE orders SET quantity = quantity - %s WHERE order_id = %s", (trade_qty, sell_order["order_id"]))

            # Possibly mark orders as filled if remaining quantity is 0
            cursor.execute("UPDATE orders SET status = 'filled' WHERE order_id = %s AND quantity = 0", (buy_order["order_id"],))
            cursor.execute("UPDATE orders SET status = 'filled' WHERE order_id = %s AND quantity = 0", (sell_order["order_id"],))

            # Update buyer portfolio (buyer gains shares)
            update_portfolio_buy(cursor, buy_order["user_id"], symbol, trade_qty, trade_price)

            # Update seller portfolio (seller loses shares)
            update_portfolio_sell(cursor, sell_order["user_id"], symbol, trade_qty)

            # Transfer money:
            # - Consume reserved funds for buyer by trade_total (reservation tracked per order)
            consume_reserved_for_order(buy_order["order_id"], buyer_username, trade_total)

            # - Pay seller by adding to their available wallet (seller receives cash immediately)
            with wallet_lock:
                user_wallets[seller_username] = user_wallets.get(seller_username, 0.0) + trade_total

            conn.commit()
            logger.info(f"Trade executed: {trade_qty} {symbol} at {trade_price}, Total: ${trade_total:.2f}")

            # After committing, if buy order is fully filled, refund any leftover reservation for that order:
            cursor.execute("SELECT status, quantity FROM orders WHERE order_id = %s", (buy_order["order_id"],))
            updated_buy = cursor.fetchone()
            if updated_buy and updated_buy["status"] == "filled":
                # Refund remaining reservation (if any) to buyer wallet
                refund_reserved_for_order(buy_order["order_id"], buyer_username)

            # If sell order filled, nothing to refund (seller never reserved money), but ensure DB cleanup already done.

        cursor.close()

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Trade execution error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# ---------------- API Endpoints ---------------- #

# Create User with initial portfolio
@app.post("/api/users")
def create_user(user_data: CreateUserRequest):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if username already exists (case sensitive check as example)
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (user_data.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")

        user_id = str(uuid4())
        cursor.execute("INSERT INTO users (user_id, username, created_at) VALUES (%s, %s, %s)", (user_id, user_data.username, datetime.utcnow()))

        # Initialize in-memory wallet and reserved maps
        with wallet_lock:
            user_wallets[user_data.username] = float(user_data.walletBalance)
            user_reserved_total[user_data.username] = 0.0
            user_initial_shares[user_data.username] = {}

        # Store initial shares in DB if provided
        if user_data.portfolio:
            # add initial shares in DB using helper (open a new connection inside helper)
            # We'll use the same connection/cursor for efficiency
            cursor_dict = conn.cursor(dictionary=True)
            for share in user_data.portfolio:
                # Insert / update portfolio using same logic as update_portfolio_buy
                update_portfolio_buy(cursor_dict, user_id, share.symbol, share.quantity, share.avg_price)
                # track initial shares in-memory
                user_initial_shares[user_data.username][share.symbol] = share.quantity
            cursor_dict.close()

        conn.commit()

        response = {
            "user_id": user_id,
            "username": user_data.username,
            "walletBalance": user_data.walletBalance,
            "message": f"User created with ${user_data.walletBalance:.2f} initial balance"
        }
        if user_data.portfolio:
            response["portfolio"] = [item.dict() for item in user_data.portfolio]
            response["message"] += f" and {len(user_data.portfolio)} portfolio items"

        cursor.close()
        return response

    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error in create_user: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Get User by ID
@app.get("/api/users/{user_id}")
def get_user(user_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT user_id, username, created_at FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        balance = user_wallets.get(user["username"], 0.00)

        cursor.execute("SELECT symbol, quantity, avg_price FROM portfolio WHERE user_id = %s", (user_id,))
        portfolio = cursor.fetchall()

        cursor.close()
        return {
            "user": user,
            "walletBalance": balance,
            "portfolio": portfolio
        }

    except mysql.connector.Error as e:
        logger.error(f"Database error in get_user: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Place Order
@app.post("/api/orders")
def place_order(order: OrderRequest):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE user_id = %s", (order.userId,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        username = user["username"]

        # Generate order_id early so we can use it to track reservation
        order_id = str(uuid4())

        # If BUY: check and reserve money (deduct from available wallet, track reservation per order)
        if order.side == OrderSide.BUY.value:
            total_cost = float(order.price) * int(order.quantity)
            with wallet_lock:
                available = user_wallets.get(username, 0.0)
                if available < total_cost:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Not enough money. You have ${available:.2f}, need ${total_cost:.2f}"
                    )
                # deduct from available wallet and create a reservation entry
                user_wallets[username] = available - total_cost
                reserve_funds_for_order(order_id, username, total_cost)

        # If SELL: ensure user has enough shares in portfolio
        if order.side == OrderSide.SELL.value:
            cursor.execute("""
                SELECT COALESCE(SUM(quantity), 0) as total_quantity
                FROM portfolio
                WHERE user_id = %s AND symbol = %s
            """, (order.userId, order.symbol))
            portfolio = cursor.fetchone()
            available_shares = int(portfolio['total_quantity']) if portfolio else 0
            if available_shares < order.quantity:
                raise HTTPException(
                    status_code=400,
                    detail=f"Not enough shares to sell. You have {available_shares} shares of {order.symbol}"
                )

        # Insert order with open status
        cursor.execute("""
            INSERT INTO orders (order_id, user_id, symbol, side, type, price, quantity, status, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s)
        """, (order_id, order.userId, order.symbol, order.side, order.type, order.price, order.quantity, datetime.utcnow()))

        conn.commit()
        cursor.close()

        # Run matching engine (this will consume reserved funds and transfer money)
        execute_trades(order.symbol)

        return {"message": "Order placed", "orderId": order_id}

    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        # If it was a buy order that failed before DB insert or afterwards, refund the reserved money if present
        if order.side == OrderSide.BUY.value:
            # Try to refund based on order_id reservation
            # Note: place_order generates order_id before reservation, but in case reservation succeeded and DB failed:
            # refund by scanning reserved_by_order (best-effort).
            # We can't access the local order_id here if exception occurred before being defined; guard with try.
            try:
                if order_id in user_reserved_by_order:
                    # figure username
                    uname = None
                    try:
                        uname = get_username_from_id(order.userId)
                    except Exception:
                        uname = None
                    if uname:
                        refund_reserved_for_order(order_id, uname)
            except Exception:
                pass
        logger.error(f"Database error in place_order: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except HTTPException:
        # re-raise known HTTP errors
        raise
    finally:
        if conn:
            conn.close()

# Get Order by ID
@app.get("/api/orders/{order_id}")
def get_order(order_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
        order = cursor.fetchone()
        cursor.close()

        if not order:
            raise HTTPException(status_code=404, detail="Order not found")

        return order

    except mysql.connector.Error as e:
        logger.error(f"Database error in get_order: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Get Orders by User ID
@app.get("/api/orders")
def get_orders_by_user(userId: str = Query(..., description="User ID to filter orders")):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM orders WHERE user_id = %s ORDER BY created_at DESC", (userId,))
        orders = cursor.fetchall()
        cursor.close()
        return orders

    except mysql.connector.Error as e:
        logger.error(f"Database error in get_orders_by_user: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Get Order Book for a Symbol
@app.get("/api/orderbook/{symbol}")
def get_orderbook(symbol: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT * FROM orders
            WHERE symbol = %s AND side = 'buy' AND status = 'open'
            ORDER BY price DESC, created_at ASC
        """, (symbol,))
        buy_orders = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM orders
            WHERE symbol = %s AND side = 'sell' AND status = 'open'
            ORDER BY price ASC, created_at ASC
        """, (symbol,))
        sell_orders = cursor.fetchall()

        cursor.close()
        return {
            "symbol": symbol,
            "buy_orders": buy_orders,
            "sell_orders": sell_orders
        }

    except mysql.connector.Error as e:
        logger.error(f"Database error in get_orderbook: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Get Trades by Symbol
@app.get("/api/trades/{symbol}")
def get_trades_by_symbol(symbol: str, limit: int = Query(100, ge=1, le=1000)):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT * FROM trades
            WHERE symbol = %s
            ORDER BY executed_at DESC
            LIMIT %s
        """, (symbol, limit))
        trades = cursor.fetchall()
        cursor.close()
        return trades

    except mysql.connector.Error as e:
        logger.error(f"Database error in get_trades_by_symbol: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Get Trades by User ID
@app.get("/api/trades")
def get_trades_by_user(userId: str = Query(..., description="User ID to filter trades")):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT t.*
            FROM trades t
            JOIN orders bo ON t.buy_order_id = bo.order_id
            JOIN orders so ON t.sell_order_id = so.order_id
            WHERE bo.user_id = %s OR so.user_id = %s
            ORDER BY t.executed_at DESC
        """, (userId, userId))
        trades = cursor.fetchall()
        cursor.close()
        return trades

    except mysql.connector.Error as e:
        logger.error(f"Database error in get_trades_by_user: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Get Portfolio by User ID
@app.get("/api/portfolio/{user_id}")
def get_portfolio(user_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT symbol, quantity, avg_price
            FROM portfolio
            WHERE user_id = %s
        """, (user_id,))
        portfolio = cursor.fetchall()
        cursor.close()
        return portfolio

    except mysql.connector.Error as e:
        logger.error(f"Database error in get_portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Cancel an order
@app.post("/api/orders/{order_id}/cancel")
def cancel_order(order_id: str):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get order with username
        cursor.execute("""
            SELECT o.*, u.username
            FROM orders o
            JOIN users u ON o.user_id = u.user_id
            WHERE o.order_id = %s AND o.status = 'open'
        """, (order_id,))
        order = cursor.fetchone()

        if not order:
            raise HTTPException(status_code=404, detail="Order not found or not cancellable")

        username = order["username"]

        # If it's a buy order, refund the reserved money (using per-order reservation map)
        if order["side"] == "buy":
            # Prefer refund via reservation map
            refund_reserved_for_order(order_id, username)

        # Cancel the order in DB
        cursor.execute("""
            UPDATE orders
            SET status = 'canceled'
            WHERE order_id = %s
        """, (order_id,))

        conn.commit()
        cursor.close()
        return {"message": "Order cancelled successfully"}

    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error in cancel_order: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
