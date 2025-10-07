CREATE DATABASE stock_exchange;
USE stock_exchange;

-- Users table
CREATE TABLE users (
    username VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Wallet table
CREATE TABLE user_wallets (
    username VARCHAR(50) PRIMARY KEY,
    balance DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

-- Orders
CREATE TABLE orders (
    order_id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side ENUM('buy', 'sell') NOT NULL,
    type ENUM('limit') NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    quantity INT NOT NULL,
    status ENUM('open', 'filled', 'canceled') DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_quantity INT DEFAULT 0,
    remaining_quantity INT GENERATED ALWAYS AS (quantity - filled_quantity) STORED,
    FOREIGN KEY (username) REFERENCES users(username)
);

-- Create index for better performance
CREATE INDEX idx_orders_symbol_side_status ON orders(symbol, side, status, price, created_at);

DELIMITER //

CREATE TRIGGER update_order_status_to_filled
AFTER UPDATE ON orders
FOR EACH ROW
BEGIN
    -- Check if the remaining_quantity is 0 and the order isn't already marked as 'filled'
    -- Use NEW.remaining_quantity which now has the updated generated value
    IF NEW.remaining_quantity = 0 AND NEW.status != 'filled' THEN
        -- Update the status to 'filled' in a separate statement
        UPDATE orders SET status = 'filled' WHERE order_id = NEW.order_id;
    END IF;
END;

//
DELIMITER ;

-- Trades
CREATE TABLE trades (
    trade_id VARCHAR(36) PRIMARY KEY,
    buy_order_id VARCHAR(36) NOT NULL,
    sell_order_id VARCHAR(36) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    quantity INT NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (buy_order_id) REFERENCES orders(order_id),
    FOREIGN KEY (sell_order_id) REFERENCES orders(order_id)
);

-- Wallet transactions
CREATE TABLE wallet_transactions (
    transaction_id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    transaction_type ENUM('deposit', 'withdrawal', 'trade_payment', 'trade_receipt', 'trade_refund', 'price_difference_refund') NOT NULL,
    related_order_id VARCHAR(36) NULL,
    related_trade_id VARCHAR(36) NULL,
    balance_after DECIMAL(15, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES users(username),
    FOREIGN KEY (related_order_id) REFERENCES orders(order_id) ON DELETE SET NULL,
    FOREIGN KEY (related_trade_id) REFERENCES trades(trade_id) ON DELETE SET NULL
);

-- Portfolio
CREATE TABLE portfolio (
    username VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    quantity INT NOT NULL,
    avg_price DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (username, symbol),
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);
