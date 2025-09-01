from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import mysql.connector
from uuid import uuid4
from datetime import datetime

# Add this function to your main.py
def execute_trades(symbol: str, price: float):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Find matching buy and sell orders
        cursor.execute("""
            SELECT buy.order_id as buy_id, sell.order_id as sell_id, 
                   buy.user_id as buyer_id, sell.user_id as seller_id,
                   LEAST(buy.quantity, sell.quantity) as trade_quantity,
                   buy.price as trade_price
            FROM orders buy, orders sell
            WHERE buy.symbol = %s AND sell.symbol = %s
            AND buy.side = 'buy' AND sell.side = 'sell'
            AND buy.price >= sell.price  -- Buy price should be >= sell price
            AND buy.status = 'open' AND sell.status = 'open'
            AND buy.user_id != sell.user_id
            ORDER BY buy.created_at, sell.created_at
            LIMIT 1
        """, (symbol, symbol))
        
        match = cursor.fetchone()
        
        if match:
            # Create trade
            trade_id = str(uuid4())
            cursor.execute("""
                INSERT INTO trades (trade_id, buy_order_id, sell_order_id, symbol, price, quantity)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (trade_id, match['buy_id'], match['sell_id'], symbol, match['trade_price'], match['trade_quantity']))
            
            # Update order statuses to 'filled'
            cursor.execute("UPDATE orders SET status = 'filled' WHERE order_id IN (%s, %s)", 
                          (match['buy_id'], match['sell_id']))
            
            # Update buyer's portfolio
            cursor.execute("""
                INSERT INTO portfolio (user_id, symbol, quantity, avg_price) 
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                quantity = quantity + VALUES(quantity),
                avg_price = ((avg_price * quantity) + (VALUES(avg_price) * VALUES(quantity))) / (quantity + VALUES(quantity))
            """, (match['buyer_id'], symbol, match['trade_quantity'], match['trade_price']))
            
            # Update seller's portfolio (remove stocks)
            cursor.execute("""
                UPDATE portfolio 
                SET quantity = quantity - %s
                WHERE user_id = %s AND symbol = %s AND quantity >= %s
            """, (match['trade_quantity'], match['seller_id'], symbol, match['trade_quantity']))
            
            conn.commit()
            print(f"Trade executed: {match['trade_quantity']} {symbol} at {match['trade_price']}")
            
        conn.close()
        
    except mysql.connector.Error as e:
        print(f"Trade execution error: {str(e)}")

app = FastAPI()

# Database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Naman@321",
            database="stock_exchange"
        )
        return connection
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# Pydantic models
class Order(BaseModel):
    userId: str
    symbol: str
    side: str
    type: str
    price: float
    quantity: int

class Trade(BaseModel):
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    symbol: str
    price: float
    quantity: int
    executed_at: datetime

# ADD USER CREATION ENDPOINT
@app.post("/api/users")
def create_user(username: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = str(uuid4())
        cursor.execute(
            "INSERT INTO users (user_id, username) VALUES (%s, %s)",
            (user_id, username)
        )
        conn.commit()
        conn.close()
        return {"user_id": user_id, "username": username}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# KEEP ONLY THIS ORDERS ENDPOINT (with user creation logic)
@app.post("/api/orders")
def place_order(order: Order):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # CHECK IF USER EXISTS, IF NOT CREATE THEM
        cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (order.userId,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users (user_id, username) VALUES (%s, %s)",
                (order.userId, f"user_{order.userId[:8]}")
            )
        
        # PLACE THE ORDER
        order_id = str(uuid4())
        cursor.execute(
            "INSERT INTO orders (order_id, user_id, symbol, side, type, price, quantity) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (order_id, order.userId, order.symbol, order.side, order.type, order.price, order.quantity)
        )
        
        conn.commit()
        conn.close()
        
        # 🚨 ADD THIS LINE - Execute trades after placing order
        execute_trades(order.symbol, order.price)
        
        return {"message": "Order placed", "orderId": order_id}
        
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ... KEEP ALL YOUR OTHER GET ENDPOINTS AS THEY ARE ...
# GET /api/orders/{orderId}
# GET /api/orders?userId=...
# GET /api/orderbook/{symbol}
# GET /api/trades/{symbol}
# GET /api/trades?userId=...
# GET /api/portfolio/{userId}

# GET /api/orders/{orderId} - Get order by ID
@app.get("/api/orders/{order_id}")
def get_order(order_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
        order = cursor.fetchone()
        conn.close()
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        return order
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# GET /api/orders?userId=... - Get orders by user
@app.get("/api/orders")
def get_orders_by_user(user_id: str = Query(..., description="User ID to filter orders")):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM orders WHERE user_id = %s", (user_id,))
        orders = cursor.fetchall()
        conn.close()
        return orders
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# GET /api/orderbook/{symbol} - Get order book for a symbol (simplified)
@app.get("/api/orderbook/{symbol}")
def get_orderbook(symbol: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM orders WHERE symbol = %s AND status = 'open' ORDER BY price, created_at", (symbol,))
        orders = cursor.fetchall()
        conn.close()
        buy_orders = [o for o in orders if o['side'] == 'buy']
        sell_orders = [o for o in orders if o['side'] == 'sell']
        return {"symbol": symbol, "buy_orders": buy_orders, "sell_orders": sell_orders}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# GET /api/trades/{symbol} - Get trades by symbol
@app.get("/api/trades/{symbol}")
def get_trades_by_symbol(symbol: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM trades WHERE symbol = %s", (symbol,))
        trades = cursor.fetchall()
        conn.close()
        return trades
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# GET /api/trades?userId=... - Get trades by user
@app.get("/api/trades")
def get_trades_by_user(user_id: str = Query(..., description="User ID to filter trades")):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT t.* FROM trades t
            JOIN orders o ON t.buy_order_id = o.order_id OR t.sell_order_id = o.order_id
            WHERE o.user_id = %s
        """, (user_id,))
        trades = cursor.fetchall()
        conn.close()
        return trades
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# GET /api/portfolio/{userId} - Get portfolio by user
@app.get("/api/portfolio/{user_id}")
def get_portfolio(user_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM portfolio WHERE user_id = %s", (user_id,))
        portfolio = cursor.fetchall()
        conn.close()
        return portfolio
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")